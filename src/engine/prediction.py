"""计算异常分数、阈值和最终预测报告。"""


import json

import time


from tqdm import tqdm


from src.data.utils import *

from src.engine.eval_methods import *

from src.models.physical_response import compute_numpy_physical_response_errors


def _autocast_context(device):

    enabled = device == "cuda"

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):

        return torch.amp.autocast("cuda", enabled=enabled)

    return torch.cuda.amp.autocast(enabled=enabled)


class Predictor:

    """MTAD-GAT 预测器类。


    :param model: MTAD-GAT model (pre-trained) used to forecast and reconstruct

    :param window_size: Length of the input sequence

    :param n_features: Number of input features

    :param pred_args: params for thresholding and predicting anomalies


    """


    def __init__(self, model, window_size, n_features, pred_args, summary_file_name="summary.txt"):

        self.model = model

        self.window_size = window_size

        self.n_features = n_features

        self.dataset = pred_args["dataset"]

        self.target_dims = pred_args["target_dims"]

        self.score_dims = pred_args.get("score_dims")

        if self.score_dims is None:

            self.score_dims = get_score_dims(self.dataset, self.target_dims)

        if self.score_dims is not None:

            self.score_dims = [int(index) for index in self.score_dims]

        self.scale_scores = pred_args["scale_scores"]

        self.q = pred_args["q"]

        self.level = pred_args["level"]

        self.normal_threshold_quantile = float(np.clip(
            pred_args.get("normal_threshold_quantile", 0.99), 0.5, 0.9999
        ))

        self.dynamic_pot = pred_args["dynamic_pot"]

        self.use_mov_av = pred_args["use_mov_av"]

        self.gamma = pred_args["gamma"]

        self.score_fusion_mode = pred_args.get("score_fusion_mode", "fixed")

        self.use_event_consistency = pred_args.get("use_event_consistency", False)

        self.event_low_ratio = pred_args.get("event_low_ratio", 0.5)

        self.event_min_length = pred_args.get("event_min_length", 3)

        self.use_physical_response_score = pred_args.get("use_physical_response_score", False)

        self.physical_response_config = pred_args.get("physical_response_config")

        self.physical_response_max_weight = float(pred_args.get("physical_response_max_weight", 0.35))

        self.use_physical_consistency_score = bool(
            pred_args.get("use_physical_consistency_head", False)
        )

        self.physical_consistency_score_max_weight = float(np.clip(
            pred_args.get("physical_consistency_score_max_weight", 0.35), 0.0, 0.5
        ))

        self.use_relation_change_score = bool(pred_args.get("use_relation_change_score", False))

        self.relation_change_weight = float(np.clip(
            pred_args.get("relation_change_weight", 0.2), 0.0, 0.5
        ))

        self.relation_change_fusion_mode = str(
            pred_args.get("relation_change_fusion_mode", "linear_legacy")
        )

        self.relation_change_mode = str(
            pred_args.get("relation_change_mode", "consecutive_js")
        )

        if self.relation_change_fusion_mode not in {"linear_legacy", "residual_gated"}:

            raise ValueError(f"Unsupported relation-change fusion mode: {self.relation_change_fusion_mode}")

        if self.relation_change_mode not in {"consecutive_js", "normal_transition_residual"}:

            raise ValueError(f"Unsupported relation-change mode: {self.relation_change_mode}")

        self._relation_change_calibration = None

        self._relation_transition_params = None

        self._last_relation_attention_segments = None

        self.last_sample_value_only_scores = {}

        self.last_sample_relation_raw = {}

        self._physical_fusion_calibration = None

        self._physical_consistency_fusion_calibration = None

        self.last_physical_response_term_summary = {}

        self.last_scoring_stats = {}

        # Formal runs pass the training-side runtime snapshot through prediction
        # arguments so the standard report contains one comparable efficiency block.
        self.training_runtime = dict(pred_args.get("training_runtime", {}))
        self.preprocessing_seconds = float(pred_args.get("preprocessing_seconds", 0.0))
        self.model_parameters = int(
            pred_args.get(
                "model_parameters",
                sum(parameter.numel() for parameter in self.model.parameters()),
            )
        )

        self._fusion_weights = None

        self.reg_level = pred_args["reg_level"]

        self.save_path = pred_args["save_path"]

        self.batch_size = int(pred_args.get("predict_batch_size", 256))

        self.window_stride = max(1, int(pred_args.get("window_stride", 1)))

        default_workers = 2 if os.name == "nt" else 4

        self.num_workers = max(0, int(pred_args.get("predict_num_workers", default_workers)))

        self.pin_memory = bool(pred_args.get("predict_pin_memory", True))

        self.use_cuda = bool(pred_args.get("use_cuda", True))

        self.pred_args = pred_args

        self.summary_file_name = summary_file_name


    def _score_view(self, values):

        """Select response dimensions that are allowed to raise the global alarm."""

        if self.score_dims is None:

            return values

        if values.ndim < 2:

            raise ValueError("score dimension selection requires a 2-D array")

        invalid = [index for index in self.score_dims if index < 0 or index >= values.shape[1]]

        if invalid:

            raise ValueError(

                f"score_dims {invalid} exceed model output dimension {values.shape[1]}"

            )

        if not self.score_dims:

            raise ValueError("No configured response dimension is present in model outputs")

        return values[:, self.score_dims]


    def _aggregate_output_scores(self, values):

        return np.mean(self._score_view(values), axis=1)


    @staticmethod

    def _branch_stability(error_series, eps=1e-6):

        median = np.median(error_series)

        mad = np.median(np.abs(error_series - median))

        scale = np.median(np.abs(error_series)) + eps

        return mad / scale


    def _compute_fusion_weights(self, pred_errors, recon_errors):

        if self.score_fusion_mode != "quality_aware":

            pred_weights = np.ones(pred_errors.shape[1], dtype=np.float32)

            recon_weights = np.full(pred_errors.shape[1], float(self.gamma), dtype=np.float32)

            return pred_weights, recon_weights

        if self._fusion_weights is not None:

            return self._fusion_weights


        pred_weights = []

        recon_weights = []

        for feature_idx in range(pred_errors.shape[1]):

            pred_stability = self._branch_stability(pred_errors[:, feature_idx])

            recon_stability = self._branch_stability(recon_errors[:, feature_idx])


            pred_quality = 1.0 / (1e-6 + pred_stability)

            recon_quality = float(self.gamma) / (1e-6 + recon_stability)

            quality_sum = pred_quality + recon_quality + 1e-6


            pred_weights.append(pred_quality / quality_sum)

            recon_weights.append(recon_quality / quality_sum)


        self._fusion_weights = (

            np.asarray(pred_weights, dtype=np.float32),

            np.asarray(recon_weights, dtype=np.float32),

        )

        return self._fusion_weights


    def _restore_fusion_weights(self, score_df):

        if self.score_fusion_mode != "quality_aware" or self._fusion_weights is not None or score_df.empty:

            return

        out_dim = self.n_features if self.target_dims is None else len(self.target_dims)

        pred_columns = [f"Pred_Weight_{index}" for index in range(out_dim)]

        recon_columns = [f"Recon_Weight_{index}" for index in range(out_dim)]

        if all(column in score_df.columns for column in pred_columns + recon_columns):

            self._fusion_weights = (

                score_df[pred_columns].iloc[0].to_numpy(dtype=np.float32),

                score_df[recon_columns].iloc[0].to_numpy(dtype=np.float32),

            )


    def _compute_physical_response_score(self, actual, recons, segment_lengths=None):

        config = self.physical_response_config

        if not self.use_physical_response_score or not config:

            return None

        if segment_lengths is None:
            segment_lengths = [len(actual)]
        if sum(segment_lengths) != len(actual):
            raise ValueError("Physical-response segment lengths do not match score arrays")

        score_parts = []
        summary_values = {}
        offset = 0
        for segment_length in segment_lengths:
            segment_slice = slice(offset, offset + segment_length)
            term_errors = compute_numpy_physical_response_errors(
                actual[segment_slice],
                recons[segment_slice],
                config,
            )
            if not term_errors:
                return None
            score_parts.append(np.mean(np.stack(list(term_errors.values()), axis=1), axis=1))
            for name, values in term_errors.items():
                summary_values.setdefault(name, []).append(np.asarray(values))
            offset += segment_length

        if not score_parts:

            return None

        self.last_physical_response_term_summary = {
            name: float(np.mean(np.concatenate(values)))
            for name, values in summary_values.items()
        }
        return np.concatenate(score_parts).astype(np.float32)


    def _fuse_physical_response(self, model_scores, physical_scores):

        if self._physical_fusion_calibration is None:

            model_center = float(np.median(model_scores))

            model_scale = float(np.median(np.abs(model_scores - model_center)) + 1e-6)

            physical_center = float(np.median(physical_scores))

            physical_scale = float(np.median(np.abs(physical_scores - physical_center)) + 1e-6)

            model_stability = self._branch_stability(model_scores)

            physical_stability = self._branch_stability(physical_scores)

            physical_quality = 1.0 / (physical_stability + 1e-6)

            model_quality = 1.0 / (model_stability + 1e-6)

            weight = physical_quality / (physical_quality + model_quality)

            weight = float(np.clip(weight, 0.0, self.physical_response_max_weight))

            self._physical_fusion_calibration = {

                "model_center": model_center,

                "model_scale": model_scale,

                "physical_center": physical_center,

                "physical_scale": physical_scale,

                "weight": weight,

            }

        calibration = self._physical_fusion_calibration

        physical_standardized = np.maximum(

            0.0,

            (physical_scores - calibration["physical_center"]) / calibration["physical_scale"],

        )

        physical_aligned = calibration["model_center"] + calibration["model_scale"] * physical_standardized

        weight = calibration["weight"]

        return (1.0 - weight) * model_scores + weight * physical_aligned, weight


    def get_calibration_summary(self):

        """Expose validation-fitted score weights for reproducible reporting."""

        summary = {

            "score_fusion_mode": self.score_fusion_mode,

            "global_score_dims": self.score_dims,

        }

        if self._fusion_weights is not None:

            pred_weights, recon_weights = self._fusion_weights

            summary["prediction_weights"] = [float(value) for value in pred_weights]

            summary["reconstruction_weights"] = [float(value) for value in recon_weights]

        if self._physical_fusion_calibration is not None:

            summary["physical_response_weight"] = float(self._physical_fusion_calibration["weight"])

            summary["physical_response_max_weight"] = float(self.physical_response_max_weight)

            summary["calibration_source"] = "normal_validation_or_training_reference"

        if self._physical_consistency_fusion_calibration is not None:

            summary["physical_consistency"] = {

                **{
                    key: float(value)
                    for key, value in self._physical_consistency_fusion_calibration.items()
                },

                "max_weight": float(self.physical_consistency_score_max_weight),

                "calibration_source": "normal_training_reference",

            }

        if self._relation_change_calibration is not None:

            summary["relation_change"] = {

                **{key: float(value) for key, value in self._relation_change_calibration.items()},

                "weight": float(self.relation_change_weight),

                "distance": self.relation_change_mode,

                "fusion_mode": self.relation_change_fusion_mode,

                "calibration_source": "normal_calibration_reference",

            }

        return summary


    def _fit_physical_consistency_calibration(self, train_df):

        required = {"A_Score_Global", "Physical_Consistency_Score"}

        if not required.issubset(train_df.columns):

            missing = sorted(required - set(train_df.columns))

            raise KeyError(f"C4 score calibration is missing columns: {missing}")

        model_scores = train_df["A_Score_Global"].to_numpy(dtype=np.float32)

        physical_scores = train_df["Physical_Consistency_Score"].to_numpy(dtype=np.float32)

        model_center = float(np.median(model_scores))

        physical_center = float(np.median(physical_scores))

        model_scale = float(max(1.4826 * np.median(np.abs(model_scores - model_center)), 1e-7))

        physical_scale = float(max(
            1.4826 * np.median(np.abs(physical_scores - physical_center)), 1e-7
        ))

        model_quality = 1.0 / (self._branch_stability(model_scores) + 1e-6)

        physical_quality = 1.0 / (self._branch_stability(physical_scores) + 1e-6)

        quality_weight = physical_quality / (model_quality + physical_quality)

        self._physical_consistency_fusion_calibration = {

            "model_center": model_center,

            "model_scale": model_scale,

            "physical_center": physical_center,

            "physical_scale": physical_scale,

            "weight": float(np.clip(
                quality_weight, 0.0, self.physical_consistency_score_max_weight
            )),

        }


    def _apply_physical_consistency_fusion(self, score_df):

        calibration = self._physical_consistency_fusion_calibration

        if calibration is None:

            raise RuntimeError("C4 consistency calibration must be fitted on normal training data")

        model_scores = score_df["A_Score_Global"].to_numpy(dtype=np.float32)

        physical_scores = score_df["Physical_Consistency_Score"].to_numpy(dtype=np.float32)

        physical_excess = np.maximum(

            0.0,

            (physical_scores - calibration["physical_center"]) / calibration["physical_scale"],

        )

        physical_aligned = (

            calibration["model_center"] + calibration["model_scale"] * physical_excess

        )

        weight = calibration["weight"]

        score_df["A_Score_Backbone"] = model_scores

        score_df["Physical_Consistency_Aligned"] = physical_aligned.astype(np.float32)

        score_df["Physical_Consistency_Weight"] = float(weight)

        score_df["A_Score_Global"] = (

            (1.0 - weight) * model_scores + weight * physical_aligned

        ).astype(np.float32)

        return score_df


    @staticmethod

    def _relation_change_from_attention(attention, previous_attention=None):

        """Return per-window JS change of row-normalized Feature-GAT relations."""

        eps = 1e-7

        attention = attention.float().clamp_min(eps)

        if previous_attention is None:

            first = attention[:1]

        else:

            first = previous_attention.unsqueeze(0).float().clamp_min(eps)

        prior = torch.cat((first, attention[:-1]), dim=0)

        mixture = 0.5 * (attention + prior)

        divergence = 0.5 * (

            attention * (attention.log() - mixture.log())

            + prior * (prior.log() - mixture.log())

        )

        return divergence.mean(dim=(1, 2)), attention[-1].detach()


    def _fit_relation_transition_model(self):

        """Fit A[t+1] = mean + alpha * (A[t] - mean) on normal attention windows."""

        segments = self._last_relation_attention_segments

        if not segments:

            raise RuntimeError("Normal attention windows are required before fitting relation transitions")

        usable = [segment.float() for segment in segments if len(segment) >= 2]

        if not usable:

            raise RuntimeError("At least two normal attention windows are required per relation segment")

        all_attention = torch.cat(usable, dim=0)

        mean_attention = all_attention.mean(dim=0)

        numerator = all_attention.new_tensor(0.0)

        denominator = all_attention.new_tensor(0.0)

        for segment in usable:

            previous = segment[:-1] - mean_attention

            current = segment[1:] - mean_attention

            numerator += (previous * current).sum()

            denominator += previous.square().sum()

        alpha = (numerator / denominator.clamp_min(1e-12)).clamp(0.0, 1.0)

        self._relation_transition_params = {

            "mean_attention": mean_attention.detach().cpu(),

            "alpha": float(alpha.detach().cpu()),

        }


    def _relation_transition_residual_from_attention(self, attention):

        """Per-window residual under a normal-fitted first-order attention transition."""

        if self._relation_transition_params is None:

            # Only used provisionally while extracting the normal fitting sequence.
            mean_attention = attention.mean(dim=0)

            alpha = 1.0

        else:

            mean_attention = self._relation_transition_params["mean_attention"].to(attention.device)

            alpha = self._relation_transition_params["alpha"]

        previous = torch.cat((attention[:1], attention[:-1]), dim=0)

        predicted = mean_attention.unsqueeze(0) + alpha * (previous - mean_attention.unsqueeze(0))

        residual = (attention - predicted).square().mean(dim=(1, 2))

        if len(residual) > 1:

            residual[0] = residual[1]

        return residual


    def _relation_raw_from_segments(self, segments):

        raw_parts = []

        for segment in segments:

            if self.relation_change_mode == "normal_transition_residual":

                raw = self._relation_transition_residual_from_attention(segment)

            else:

                raw, _ = self._relation_change_from_attention(segment)

            raw = raw.numpy().astype(np.float32)

            if len(raw) > 1 and raw[0] == 0.0:

                raw[0] = raw[1]

            raw_parts.append(raw)

        return np.concatenate(raw_parts)


    def _fit_relation_change_calibration(self, train_df):

        values = train_df["Relation_Change_Raw"].to_numpy(dtype=np.float32)

        center = float(np.median(values))

        mad = float(np.median(np.abs(values - center)))

        model_values = train_df["A_Score_Global"].to_numpy(dtype=np.float32)

        model_center = float(np.median(model_values))

        model_mad = float(np.median(np.abs(model_values - model_center)))

        self._relation_change_calibration = {

            "center": center,

            "scale": max(1.4826 * mad, 1e-7),

            "model_center": model_center,

            "model_scale": max(1.4826 * model_mad, 1e-7),

            "model_p95": float(np.percentile(model_values, 95)),

        }


    def _apply_relation_change_fusion(self, score_df):

        calibration = self._relation_change_calibration

        if calibration is None:

            raise RuntimeError("Relation-change calibration must be fitted on normal training data")

        raw = score_df["Relation_Change_Raw"].to_numpy(dtype=np.float32)

        excess = np.maximum(0.0, (raw - calibration["center"]) / calibration["scale"])

        model_score = score_df["A_Score_Global"].to_numpy(dtype=np.float32)

        weight = self.relation_change_weight

        if self.relation_change_fusion_mode == "residual_gated":

            bounded_excess = np.minimum(excess, 3.0)

            gate_logit = np.clip(

                (model_score - calibration["model_p95"]) / calibration["model_scale"],

                -20.0, 20.0,

            )

            value_gate = 1.0 / (1.0 + np.exp(-gate_logit))

            relation_increment = (

                weight * calibration["model_scale"] * bounded_excess * value_gate

            )

            aligned = model_score + relation_increment

            fused = aligned

        else:

            aligned = calibration["model_center"] + calibration["model_scale"] * excess

            fused = (1.0 - weight) * model_score + weight * aligned

        score_df["A_Score_Value_Only"] = model_score

        score_df["Relation_Change_Score"] = aligned.astype(np.float32)

        score_df["Relation_Change_Weight"] = float(weight)

        score_df["A_Score_Global"] = fused.astype(np.float32)

        return score_df


    @staticmethod

    def _to_serializable_dict(metrics):

        serializable = {}

        for k, v in metrics.items():

            if isinstance(v, list):

                serializable[k] = v

            else:

                serializable[k] = float(v)

        return serializable


    def _save_standard_reports(self, e_eval, p_eval, bf_eval, feature_thresholds, global_threshold, extra_reports=None):

        summary = {"epsilon_result": e_eval, "pot_result": p_eval, "bf_result": bf_eval}

        if extra_reports:

            summary.update(extra_reports)

        with open(f"{self.save_path}/{self.summary_file_name}", "w") as f:

            json.dump(summary, f, indent=2)


        with open(f"{self.save_path}/summary_metrics.json", "w") as f:

            json.dump(summary, f, indent=2)


        thresholds = {

            "global_threshold": float(global_threshold),

            "feature_thresholds": {k: float(v) for k, v in feature_thresholds.items()},

        }

        if extra_reports and "event_consistency_result" in extra_reports:

            event_report = extra_reports["event_consistency_result"]

            thresholds["event_high_threshold"] = float(event_report["high_threshold"])

            thresholds["event_low_threshold"] = float(event_report["low_threshold"])

        with open(f"{self.save_path}/thresholds.json", "w") as f:

            json.dump(thresholds, f, indent=2)


    @staticmethod

    def _split_segment_slices(score_df):

        if "Segment_ID" not in score_df.columns or len(score_df) == 0:

            return [slice(0, len(score_df))]


        segment_ids = score_df["Segment_ID"].to_numpy()

        boundaries = np.where(segment_ids[1:] != segment_ids[:-1])[0] + 1

        start = 0

        slices = []

        for boundary in boundaries:

            slices.append(slice(start, boundary))

            start = boundary

        slices.append(slice(start, len(score_df)))

        return slices


    def _compute_event_low_threshold(self, train_scores, high_threshold):

        ratio = float(np.clip(self.event_low_ratio, 0.0, 1.0))

        train_median = float(np.median(train_scores))

        if high_threshold <= train_median:

            return float(high_threshold)

        return float(train_median + ratio * (high_threshold - train_median))


    def _apply_hysteresis_to_segment(self, scores, high_threshold, low_threshold):

        preds = np.zeros(len(scores), dtype=np.int32)

        event_start = None


        for idx, score in enumerate(scores):

            if event_start is None:

                if score >= high_threshold:

                    event_start = idx

            elif score < low_threshold:

                if idx - event_start >= max(1, int(self.event_min_length)):

                    preds[event_start:idx] = 1

                event_start = None


        if event_start is not None and len(scores) - event_start >= max(1, int(self.event_min_length)):

            preds[event_start:] = 1


        return preds


    def _apply_event_consistency(self, score_df, high_threshold, low_threshold):

        refined = np.zeros(len(score_df), dtype=np.int32)

        scores = score_df["A_Score_Global"].to_numpy(dtype=np.float32)

        for segment_slice in self._split_segment_slices(score_df):

            refined[segment_slice] = self._apply_hysteresis_to_segment(

                scores[segment_slice],

                high_threshold=high_threshold,

                low_threshold=low_threshold,

            )

        return refined


    @staticmethod

    def _evaluate_binary_predictions(pred, true_anomalies):

        metrics = {

            "positive_count": int(np.sum(pred)),

        }

        if true_anomalies is None:

            return metrics


        pred_eval, latency = adjust_predicts(None, true_anomalies, 0.0, pred=pred.copy(), calc_latency=True)

        point_metrics = calc_point2point(pred_eval, true_anomalies)

        metrics.update({

            "f1": float(point_metrics[0]),

            "precision": float(point_metrics[1]),

            "recall": float(point_metrics[2]),

            "TP": float(point_metrics[3]),

            "TN": float(point_metrics[4]),

            "FP": float(point_metrics[5]),

            "FN": float(point_metrics[6]),

            "latency": float(latency),

            "metric_scope": "legacy_point_adjusted",

            "point_adjustment": True,

        })

        return metrics


    @staticmethod

    def _binary_intervals(values):

        values = np.asarray(values, dtype=np.int32)

        padded = np.pad(values, (1, 1), mode="constant")

        changes = np.diff(padded)

        starts = np.where(changes == 1)[0]

        ends = np.where(changes == -1)[0]

        return list(zip(starts.tolist(), ends.tolist()))


    def _evaluate_raw_events(self, pred, true_anomalies, score_df=None):

        """Evaluate unadjusted interval detection and first-hit delay."""

        if true_anomalies is None:

            return {}

        predictions = np.asarray(pred, dtype=np.int32)

        labels = np.asarray(true_anomalies, dtype=np.int32)

        if len(predictions) != len(labels):

            raise ValueError("Event metric predictions and labels must have the same length")

        segment_slices = self._split_segment_slices(score_df) if score_df is not None else [slice(0, len(labels))]

        true_event_count = 0

        detected_true_events = 0

        predicted_event_count = 0

        overlapping_predicted_events = 0

        delays = []

        for segment_slice in segment_slices:

            segment_pred = predictions[segment_slice]

            segment_labels = labels[segment_slice]

            true_intervals = self._binary_intervals(segment_labels)

            predicted_intervals = self._binary_intervals(segment_pred)

            true_event_count += len(true_intervals)

            predicted_event_count += len(predicted_intervals)

            for start, end in true_intervals:

                hits = np.where(segment_pred[start:end] == 1)[0]

                if hits.size:

                    detected_true_events += 1

                    delays.append(int(hits[0]))

            for pred_start, pred_end in predicted_intervals:

                if any(pred_start < true_end and pred_end > true_start for true_start, true_end in true_intervals):

                    overlapping_predicted_events += 1

        event_precision = overlapping_predicted_events / predicted_event_count if predicted_event_count else 0.0

        event_recall = detected_true_events / true_event_count if true_event_count else 0.0

        event_f1 = (

            2.0 * event_precision * event_recall / (event_precision + event_recall)

            if event_precision + event_recall > 0

            else 0.0

        )

        result = {

            "metric_scope": "raw_event",

            "point_adjustment": False,

            "true_event_count": int(true_event_count),

            "detected_true_event_count": int(detected_true_events),

            "missed_true_event_count": int(true_event_count - detected_true_events),

            "predicted_event_count": int(predicted_event_count),

            "overlapping_predicted_event_count": int(overlapping_predicted_events),

            "event_precision": float(event_precision),

            "event_recall": float(event_recall),

            "event_f1": float(event_f1),

            "delay_unit": "scored_step",

            "source_timestep_multiplier": int(self.window_stride),

        }

        if delays:

            result.update({

                "mean_first_hit_delay_scored_steps": float(np.mean(delays)),

                "median_first_hit_delay_scored_steps": float(np.median(delays)),

                "max_first_hit_delay_scored_steps": int(np.max(delays)),

                "mean_first_hit_delay_source_timesteps": float(np.mean(delays) * self.window_stride),

                "median_first_hit_delay_source_timesteps": float(np.median(delays) * self.window_stride),

            })

        else:

            result.update({

                "mean_first_hit_delay_scored_steps": None,

                "median_first_hit_delay_scored_steps": None,

                "max_first_hit_delay_scored_steps": None,

                "mean_first_hit_delay_source_timesteps": None,

                "median_first_hit_delay_source_timesteps": None,

            })

        return result


    @staticmethod

    def _evaluate_raw_scores(scores, pred, true_anomalies):

        """Compute point metrics without point adjustment or threshold search."""

        if true_anomalies is None:

            return {}

        from sklearn.metrics import auc, average_precision_score, precision_recall_curve, precision_recall_fscore_support, roc_auc_score

        labels = np.asarray(true_anomalies, dtype=np.int32)

        predictions = np.asarray(pred, dtype=np.int32)

        precision, recall, f1, _ = precision_recall_fscore_support(

            labels,

            predictions,

            average="binary",

            zero_division=0,

        )

        average_precision = float(average_precision_score(labels, scores))
        precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
        trapezoidal_pr_auc = float(auc(recall_curve, precision_curve))

        result = {

            "pr_auc": average_precision,
            "pr_auc_trapezoid": trapezoidal_pr_auc,

            "average_precision": average_precision,

            "auprc": average_precision,

            "point_precision": float(precision),

            "point_recall": float(recall),

            "point_f1": float(f1),

            "threshold_source": "normal_calibration_reference_epsilon",

            "point_adjustment": False,

        }

        if np.unique(labels).size > 1:

            result["auroc"] = float(roc_auc_score(labels, scores))

        return result


    @staticmethod

    def _add_segment_metadata(score_df, segment_id, segment_source_length, global_start_index):

        score_df = score_df.copy()

        score_df["Segment_ID"] = int(segment_id)

        score_df["Segment_Pos"] = np.arange(len(score_df), dtype=np.int32)

        score_df["Segment_Source_Length"] = int(segment_source_length)

        score_df["Global_Pos"] = np.arange(global_start_index, global_start_index + len(score_df), dtype=np.int32)

        return score_df


    @staticmethod

    def _collect_segment_boundaries(score_df):

        if "Segment_ID" not in score_df.columns or len(score_df) == 0:

            return []


        boundaries = []

        current_segment = None

        for idx, segment_id in enumerate(score_df["Segment_ID"].values):

            if current_segment is None:

                current_segment = segment_id

                continue

            if segment_id != current_segment:

                boundaries.append(idx)

                current_segment = segment_id

        return boundaries


    @staticmethod

    def _build_segment_summary(score_df):

        if "Segment_ID" not in score_df.columns or len(score_df) == 0:

            return pd.DataFrame()


        summary_rows = []

        for segment_id, segment_df in score_df.groupby("Segment_ID", sort=True):

            pred_count = int(segment_df["A_Pred_Global"].sum()) if "A_Pred_Global" in segment_df.columns else 0

            summary_rows.append({

                "segment_id": int(segment_id),

                "score_length": int(len(segment_df)),

                "score_start_index": int(segment_df.index[0]),

                "score_end_index": int(segment_df.index[-1]),

                "global_start_pos": int(segment_df["Global_Pos"].iloc[0]),

                "global_end_pos": int(segment_df["Global_Pos"].iloc[-1]),

                "global_mid_pos": float((segment_df["Global_Pos"].iloc[0] + segment_df["Global_Pos"].iloc[-1]) / 2.0),

                "source_length": int(segment_df["Segment_Source_Length"].iloc[0]),

                "max_score": float(segment_df["A_Score_Global"].max()),

                "mean_score": float(segment_df["A_Score_Global"].mean()),

                "pred_anomaly_count": pred_count,

            })

        return pd.DataFrame(summary_rows)


    def _save_segment_metadata(self, score_df, file_name):

        if "Segment_ID" not in score_df.columns or len(score_df) == 0:

            return


        metadata_df = self._build_segment_summary(score_df)

        metadata_df.to_csv(f"{self.save_path}/{file_name}", index=False)


    def _plot_segment_overview(self, score_df, threshold, file_name, title, top_k=5):

        if "Segment_ID" not in score_df.columns or len(score_df) == 0:

            return


        x_axis = score_df["Global_Pos"].values if "Global_Pos" in score_df.columns else np.arange(len(score_df))

        y_axis = score_df["A_Score_Global"].values

        pred_axis = score_df["A_Pred_Global"].values if "A_Pred_Global" in score_df.columns else None

        segment_summary = self._build_segment_summary(score_df)


        fig, ax = plt.subplots(figsize=(14, 4.5))

        ax.plot(x_axis, y_axis, color="tab:red", linewidth=1.1, label="Global Score")

        ax.axhline(threshold, color="tab:blue", linestyle="--", linewidth=1.0, label="Threshold")


        if pred_axis is not None and np.any(pred_axis > 0):

            anomaly_x = x_axis[pred_axis > 0]

            anomaly_y = y_axis[pred_axis > 0]

            ax.scatter(anomaly_x, anomaly_y, color="black", s=10, alpha=0.7, label="Predicted Anomaly")


        for boundary in self._collect_segment_boundaries(score_df):

            boundary_x = x_axis[boundary]

            ax.axvline(boundary_x, color="gray", linestyle=":", linewidth=0.9, alpha=0.8)


        if not segment_summary.empty:

            top_segments = segment_summary.sort_values(

                ["max_score", "pred_anomaly_count", "mean_score"],

                ascending=False

            ).head(top_k)

            top_segment_ids = set(top_segments["segment_id"].tolist())


            y_upper = max(float(np.max(y_axis)), float(threshold)) if len(y_axis) > 0 else float(threshold)

            text_y = y_upper * 1.03 if y_upper > 0 else 0.05


            for _, row in segment_summary.iterrows():

                start_x = row["global_start_pos"]

                end_x = row["global_end_pos"]

                mid_x = row["global_mid_pos"]

                segment_id = int(row["segment_id"])

                is_top_segment = segment_id in top_segment_ids


                if is_top_segment:

                    ax.axvspan(start_x, end_x, color="gold", alpha=0.10)


                label_text = f"S{segment_id:02d}"

                if is_top_segment:

                    label_text += f" max={row['max_score']:.3f}"

                ax.text(

                    mid_x,

                    text_y,

                    label_text,

                    ha="center",

                    va="bottom",

                    fontsize=8,

                    color="black" if is_top_segment else "dimgray",

                    rotation=0,

                    clip_on=False,

                )


        ax.set_title(title)

        ax.set_xlabel("Score Index")

        ax.set_ylabel("Global Anomaly Score")

        ax.legend(loc="upper right")

        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

        plt.tight_layout()

        plt.savefig(f"{self.save_path}/{file_name}", bbox_inches="tight", dpi=300)

        plt.close(fig)


    def _plot_segment_detail_figures(self, score_df, threshold, output_dir_name):

        if "Segment_ID" not in score_df.columns or len(score_df) == 0:

            return


        output_dir = os.path.join(self.save_path, output_dir_name)

        os.makedirs(output_dir, exist_ok=True)


        for segment_id, segment_df in score_df.groupby("Segment_ID", sort=True):

            x_axis = segment_df["Segment_Pos"].values if "Segment_Pos" in segment_df.columns else np.arange(len(segment_df))

            y_axis = segment_df["A_Score_Global"].values

            pred_axis = segment_df["A_Pred_Global"].values if "A_Pred_Global" in segment_df.columns else None

            segment_max_score = float(segment_df["A_Score_Global"].max())

            segment_mean_score = float(segment_df["A_Score_Global"].mean())

            pred_count = int(segment_df["A_Pred_Global"].sum()) if "A_Pred_Global" in segment_df.columns else 0


            fig, ax = plt.subplots(figsize=(12, 4))

            ax.plot(x_axis, y_axis, color="tab:red", linewidth=1.2, label="Global Score")

            ax.axhline(threshold, color="tab:blue", linestyle="--", linewidth=1.0, label="Threshold")


            if pred_axis is not None and np.any(pred_axis > 0):

                anomaly_x = x_axis[pred_axis > 0]

                anomaly_y = y_axis[pred_axis > 0]

                ax.scatter(anomaly_x, anomaly_y, color="black", s=12, alpha=0.7, label="Predicted Anomaly")


            ax.set_title(

                f"Segment {int(segment_id):02d} Global Score "

                f"(len={len(segment_df)}, source_len={int(segment_df['Segment_Source_Length'].iloc[0])}, "

                f"max={segment_max_score:.3f}, mean={segment_mean_score:.3f}, pred={pred_count})"

            )

            ax.set_xlabel("Position Within Segment Score")

            ax.set_ylabel("Global Anomaly Score")

            ax.legend(loc="upper right")

            ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

            plt.tight_layout()

            plt.savefig(

                os.path.join(output_dir, f"segment_{int(segment_id):02d}_score.png"),

                bbox_inches="tight",

                dpi=300,

            )

            plt.close(fig)


    def _save_segment_visualizations(self, train_pred_df, test_pred_df, threshold):

        if "Segment_ID" not in test_pred_df.columns:

            return


        self._save_segment_metadata(test_pred_df, "test_segment_metadata.csv")

        self._plot_segment_overview(

            test_pred_df,

            threshold,

            "test_score_segment_overview.png",

            "Test Global Anomaly Score with Segment Boundaries",

            top_k=5,

        )

        self._plot_segment_detail_figures(test_pred_df, threshold, "test_segments")


        if "Segment_ID" in train_pred_df.columns:

            self._save_segment_metadata(train_pred_df, "train_segment_metadata.csv")

            self._plot_segment_overview(

                train_pred_df,

                threshold,

                "train_score_segment_overview.png",

                "Train Global Anomaly Score with Segment Boundaries",

                top_k=5,

            )


    def get_score(self, values):

        """使用给定模型和数据计算异常分数的方法。

        :param values: 2D array of multivariate time series data, shape (N, k)

        :return np array of anomaly scores + dataframe with prediction for each channel and global anomalies

        """


        print("Predicting and calculating anomaly scores..")

        data = SlidingWindowDataset(

            values,

            self.window_size,

            self.target_dims,

            stride=self.window_stride,

        )


        loader = torch.utils.data.DataLoader(

            data, batch_size=self.batch_size, shuffle=False,

            num_workers=self.num_workers,

            pin_memory=self.pin_memory and torch.cuda.is_available(),

        )


        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"


        self.model.eval()

        preds = []

        recons = []

        consistency_errors = []

        relation_changes = []

        relation_attention_parts = []

        c3_joint_scores = []
        c3_value_residuals = []
        c3_relation_residuals = []

        previous_attention = None

        batch_latencies = []
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        inference_started = time.perf_counter()

        with torch.no_grad():

            for x, y in tqdm(loader):

                batch_started = time.perf_counter()

                x = x.to(device)

                y = y.to(device)


                with _autocast_context(device):

                    y_hat, _ = self.model(x)

                    # C4 的独立控制—响应头对应当前输入窗口 x；随后的 recon_x
                    # 前向会覆盖该缓存，因此必须在这里立即取出其窗口级残差。
                    base_model = self.model.module if hasattr(self.model, "module") else self.model
                    consistency_prediction = getattr(
                        base_model, "_physical_consistency_prediction", None
                    )
                    if consistency_prediction is not None:
                        response_dims = base_model.physical_consistency_target_dims
                        consistency_target = x[:, :, response_dims]
                        consistency_errors.append(
                            torch.mean(
                                torch.abs(consistency_target - consistency_prediction), dim=(1, 2)
                            ).detach().cpu().numpy()
                        )


                    if getattr(base_model, "use_c3_joint_relation", False):
                        c3_components = base_model.c3_joint_components(x, y, y_hat)
                        window_recon = c3_components["shifted_reconstruction"]
                        c3_joint_scores.append(
                            c3_components["joint_score"].detach().cpu().numpy()
                        )
                        c3_value_residuals.append(
                            c3_components["value_residual"].detach().cpu().numpy()
                        )
                        c3_relation_residuals.append(
                            c3_components["relation_residual"].detach().cpu().numpy()
                        )
                    else:
                        # 用真实下一步 y 拼接重构窗口
                        recon_x = torch.cat((x[:, 1:, :], y), dim=1)
                        _, window_recon = self.model(recon_x)

                if self.use_relation_change_score:

                    attention = getattr(self.model, "_feature_attention_weights", None)

                    if attention is None:

                        raise RuntimeError("Relation-change scoring requires Feature-GAT attention")

                    change, previous_attention = self._relation_change_from_attention(

                        attention.detach(), previous_attention

                    )

                    relation_changes.append(change.cpu().numpy())

                    relation_attention_parts.append(attention.detach().cpu())


                preds.append(y_hat.detach().cpu().numpy())

                # 保存窗口最后一步的重构结果

                recons.append(window_recon[:, -1, :].detach().cpu().numpy())

                batch_latencies.append(time.perf_counter() - batch_started)


        preds = np.concatenate(preds, axis=0)

        recons = np.concatenate(recons, axis=0)

        actual_indices = self.window_size + np.arange(len(data)) * self.window_stride

        actual = values.detach().cpu().numpy()[actual_indices]

        if device == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started
        total_windows = int(len(data))
        self.last_scoring_stats = {
            "device": device,
            "window_count": total_windows,
            "inference_seconds": float(inference_seconds),
            "windows_per_second": float(total_windows / max(inference_seconds, 1e-9)),
            "milliseconds_per_window": float(1000.0 * inference_seconds / max(total_windows, 1)),
            "batch_count": int(len(batch_latencies)),
            "batch_latency_p50_ms": float(np.percentile(batch_latencies, 50) * 1000.0)
            if batch_latencies else 0.0,
            "batch_latency_p95_ms": float(np.percentile(batch_latencies, 95) * 1000.0)
            if batch_latencies else 0.0,
            "peak_cuda_memory_mb": (
                float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if device == "cuda" else 0.0
            ),
            "peak_cuda_reserved_mb": (
                float(torch.cuda.max_memory_reserved() / (1024 ** 2)) if device == "cuda" else 0.0
            ),
        }


        if self.target_dims is not None:

            actual = actual[:, self.target_dims]

        # 待办：补充更细粒度的目标维度异常分数计算

        anomaly_scores = np.zeros_like(actual)

        pred_errors = np.zeros_like(actual)

        recon_errors = np.zeros_like(actual)

        df_dict = {}

        for i in range(preds.shape[1]):

            df_dict[f"Forecast_{i}"] = preds[:, i]

            df_dict[f"Recon_{i}"] = recons[:, i]

            df_dict[f"True_{i}"] = actual[:, i]

            pred_error = np.sqrt((preds[:, i] - actual[:, i]) ** 2)

            recon_error = np.sqrt((recons[:, i] - actual[:, i]) ** 2)

            pred_errors[:, i] = pred_error

            recon_errors[:, i] = recon_error

            df_dict[f"Pred_Error_{i}"] = pred_error

            df_dict[f"Recon_Error_{i}"] = recon_error


        pred_weights, recon_weights = self._compute_fusion_weights(pred_errors, recon_errors)


        for i in range(preds.shape[1]):

            pred_error = pred_errors[:, i]

            recon_error = recon_errors[:, i]

            a_score = pred_weights[i] * pred_error + recon_weights[i] * recon_error


            if self.scale_scores:

                q75, q25 = np.percentile(a_score, [75, 25])

                iqr = q75 - q25

                median = np.median(a_score)

                a_score = (a_score - median) / (1 + iqr)


            anomaly_scores[:, i] = a_score

            df_dict[f"A_Score_{i}"] = a_score

            df_dict[f"Pred_Weight_{i}"] = np.full_like(a_score, pred_weights[i], dtype=np.float32)

            df_dict[f"Recon_Weight_{i}"] = np.full_like(a_score, recon_weights[i], dtype=np.float32)


        df = pd.DataFrame(df_dict)

        if consistency_errors:
            df["Physical_Consistency_Score"] = np.concatenate(consistency_errors).astype(np.float32)

        if self.use_relation_change_score:

            relation_change = np.concatenate(relation_changes).astype(np.float32)

            if len(relation_change) > 1 and relation_change[0] == 0.0:

                relation_change[0] = relation_change[1]

            df["Relation_Change_Raw"] = relation_change

            self._last_relation_attention_segments = [torch.cat(relation_attention_parts, dim=0)]

        df['Pred_Error_Global'] = self._aggregate_output_scores(pred_errors)

        df['Recon_Error_Global'] = self._aggregate_output_scores(recon_errors)


        global_scores = self._aggregate_output_scores(anomaly_scores)

        physical_scores = self._compute_physical_response_score(actual, recons)

        if physical_scores is not None:

            global_scores, physical_weight = self._fuse_physical_response(global_scores, physical_scores)

            df['Physical_Response_Score'] = physical_scores

            df['Physical_Response_Weight'] = physical_weight

        if c3_joint_scores:
            global_scores = np.concatenate(c3_joint_scores).astype(np.float32)
            df["C3_Value_Residual"] = np.concatenate(c3_value_residuals).astype(np.float32)
            df["C3_Relation_Residual"] = np.concatenate(c3_relation_residuals).astype(np.float32)
            df["C3_Joint_NLL"] = global_scores

        df['A_Score_Global'] = global_scores

        selected_pred_weights = self._score_view(pred_weights[None, :])[0]

        selected_recon_weights = self._score_view(recon_weights[None, :])[0]

        df['Pred_Weight_Global'] = float(np.mean(selected_pred_weights))

        df['Recon_Weight_Global'] = float(np.mean(selected_recon_weights))


        return df


    def get_sample_map_scores(self, values_map, *, calibrate_relation=False):

        """Score many equal-role snippets in one DataLoader and split scores by sample."""

        sample_ids = []

        datasets = []

        counts = []

        actual_parts = []

        for sample_id, values in values_map.items():

            dataset = SlidingWindowDataset(

                values,

                self.window_size,

                self.target_dims,

                stride=self.window_stride,

            )

            if len(dataset) == 0:

                continue

            indices = self.window_size + np.arange(len(dataset)) * self.window_stride

            actual = values.detach().cpu().numpy()[indices]

            if self.target_dims is not None:

                actual = actual[:, self.target_dims]

            sample_ids.append(sample_id)

            datasets.append(dataset)

            counts.append(len(dataset))

            actual_parts.append(actual)

        if not datasets:

            raise ValueError("No Tsinghua snippet is longer than the configured lookback")

        loader = torch.utils.data.DataLoader(

            torch.utils.data.ConcatDataset(datasets),

            batch_size=self.batch_size,

            shuffle=False,

            num_workers=self.num_workers,

            pin_memory=self.pin_memory and torch.cuda.is_available(),

        )

        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

        self.model.eval()

        predictions = []

        reconstructions = []

        attention_parts = []

        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        inference_started = time.perf_counter()

        with torch.no_grad():

            for x, y in tqdm(loader, desc="Scoring snippets"):

                x = x.to(device)

                y = y.to(device)

                with _autocast_context(device):

                    y_hat, _ = self.model(x)

                    recon_x = torch.cat((x[:, 1:, :], y), dim=1)

                    _, window_recon = self.model(recon_x)

                predictions.append(y_hat.detach().cpu().numpy())

                reconstructions.append(window_recon[:, -1, :].detach().cpu().numpy())

                if self.use_relation_change_score:

                    attention = getattr(self.model, "_feature_attention_weights", None)

                    if attention is None:

                        raise RuntimeError("Relation-change scoring requires Feature-GAT attention")

                    attention_parts.append(attention.detach().cpu())

        if device == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started
        total_windows = int(sum(counts))
        self.last_scoring_stats = {
            "device": device,
            "window_count": total_windows,
            "inference_seconds": float(inference_seconds),
            "windows_per_second": float(total_windows / max(inference_seconds, 1e-9)),
            "milliseconds_per_window": float(1000.0 * inference_seconds / max(total_windows, 1)),
            "peak_cuda_memory_mb": (
                float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if device == "cuda" else 0.0
            ),
            "peak_cuda_reserved_mb": (
                float(torch.cuda.max_memory_reserved() / (1024 ** 2)) if device == "cuda" else 0.0
            ),
        }

        predictions = np.concatenate(predictions, axis=0)

        reconstructions = np.concatenate(reconstructions, axis=0)

        actual = np.concatenate(actual_parts, axis=0)

        pred_errors = np.abs(predictions - actual)

        recon_errors = np.abs(reconstructions - actual)

        pred_weights, recon_weights = self._compute_fusion_weights(pred_errors, recon_errors)

        anomaly_scores = pred_errors * pred_weights[None, :] + recon_errors * recon_weights[None, :]

        if self.scale_scores:

            q75, q25 = np.percentile(anomaly_scores, [75, 25], axis=0)

            median = np.median(anomaly_scores, axis=0)

            anomaly_scores = (anomaly_scores - median) / (1.0 + q75 - q25)

        global_scores = self._aggregate_output_scores(anomaly_scores)

        physical_scores = self._compute_physical_response_score(actual, reconstructions, segment_lengths=counts)

        if physical_scores is not None:

            global_scores, _ = self._fuse_physical_response(global_scores, physical_scores)

        value_only_scores = np.asarray(global_scores, dtype=np.float32).copy()

        relation_change = None

        if self.use_relation_change_score:

            all_attention = torch.cat(attention_parts, dim=0)

            attention_segments = []

            attention_offset = 0

            for count in counts:

                segment_attention = all_attention[attention_offset:attention_offset + count]

                attention_segments.append(segment_attention)

                attention_offset += count

            self._last_relation_attention_segments = attention_segments

            relation_change = self._relation_raw_from_segments(attention_segments)

            score_df = pd.DataFrame({

                "A_Score_Global": value_only_scores,

                "Relation_Change_Raw": relation_change,

            })

            if calibrate_relation:

                self._fit_relation_change_calibration(score_df)

            if self._relation_change_calibration is None:

                raise RuntimeError(

                    "Sample-map relation scoring requires a normal calibration call first"

                )

            score_df = self._apply_relation_change_fusion(score_df)

            global_scores = score_df["A_Score_Global"].to_numpy(dtype=np.float32)

        result = {}

        value_only_result = {}

        relation_raw_result = {}

        offset = 0

        for sample_id, count in zip(sample_ids, counts):

            result[sample_id] = global_scores[offset:offset + count]

            value_only_result[sample_id] = value_only_scores[offset:offset + count]

            if relation_change is not None:

                relation_raw_result[sample_id] = relation_change[offset:offset + count]

            offset += count

        self.last_sample_value_only_scores = value_only_result

        self.last_sample_relation_raw = relation_raw_result

        return result


    def get_score_for_sequences(self, values):

        if isinstance(values, dict):

            score_dfs = []

            attention_segments = []

            for _, entity_values in values.items():

                score_dfs.append(self.get_score_for_sequences(entity_values))

                if self.use_relation_change_score and self._last_relation_attention_segments:

                    attention_segments.extend(self._last_relation_attention_segments)

            if not score_dfs:

                raise ValueError("No sequence data provided for scoring")

            if self.use_relation_change_score:

                self._last_relation_attention_segments = attention_segments

            return pd.concat(score_dfs, axis=0, ignore_index=True)

        if is_sequence_container(values):

            score_dfs = []

            attention_segments = []

            global_start_index = 0

            for segment_id, seq_values in enumerate(ensure_sequence_list(values)):

                if len(seq_values) <= self.window_size:

                    continue

                segment_score_df = self.get_score(seq_values)

                if self.use_relation_change_score and self._last_relation_attention_segments:

                    attention_segments.extend(self._last_relation_attention_segments)

                segment_score_df = self._add_segment_metadata(

                    segment_score_df,

                    segment_id=segment_id,

                    segment_source_length=len(seq_values),

                    global_start_index=global_start_index,

                )

                global_start_index += len(segment_score_df)

                score_dfs.append(segment_score_df)

            if not score_dfs:

                raise ValueError("No valid sequence is longer than the window size")

            if self.use_relation_change_score:

                self._last_relation_attention_segments = attention_segments

            return pd.concat(score_dfs, axis=0, ignore_index=True)

        return self.get_score(values)


    def _prepare_true_anomalies(self, true_anomalies, expected_length):

        if true_anomalies is None:

            return None


        if isinstance(true_anomalies, dict):

            aligned_parts = []

            for _, entity_labels in true_anomalies.items():

                aligned_entity = self._prepare_true_anomalies(entity_labels, expected_length=None)

                if aligned_entity is not None and len(aligned_entity) > 0:

                    aligned_parts.append(aligned_entity)

            true_anomalies = np.concatenate(aligned_parts, axis=0) if aligned_parts else None

        elif is_sequence_container(true_anomalies):

            aligned_parts = []

            for seq_labels in ensure_sequence_list(true_anomalies):

                seq_labels = np.asarray(seq_labels)

                if len(seq_labels) > self.window_size:

                    aligned_parts.append(
                        seq_labels[self.window_size::self.window_stride]
                    )

            true_anomalies = np.concatenate(aligned_parts, axis=0) if aligned_parts else None

        else:

            true_anomalies = np.asarray(true_anomalies)


        if true_anomalies is None:

            return None


        if expected_length is None:

            return true_anomalies


        if len(true_anomalies) == expected_length:

            return true_anomalies


        if len(true_anomalies) == expected_length + self.window_size:

            return true_anomalies[self.window_size:]

        # Raw point labels need the same lookback and stride selection as the
        # SlidingWindowDataset used to produce anomaly scores.
        aligned = true_anomalies[self.window_size::self.window_stride]

        if len(aligned) == expected_length:

            return aligned


        raise ValueError(

            f"True anomaly labels length {len(true_anomalies)} does not match expected score length {expected_length}"

        )


    def predict_anomalies(self, train, test, true_anomalies=None, load_scores=False, save_output=True,

                          scale_scores=False, cached_train_pred_df=None):

        """ 预测异常。


        :param train: 2D array of train multivariate time series data (normal data used to establish baseline)

        :param test: 2D array of test multivariate time series data (data to be evaluated for anomalies)

        :param true_anomalies: true anomalies of test set, None if not available (for unsupervised setting)

        :param save_scores: Whether to save anomaly scores of train and test

        :param load_scores: Whether to load anomaly scores instead of calculating them

        :param save_output: Whether to save output dataframe

        :param scale_scores: Whether to feature-wise scale anomaly scores

        :param cached_train_pred_df: Pre-computed train_pred_df to skip redundant model inference

        """


        train_scoring_stats = {}
        test_scoring_stats = {}

        if load_scores:

            print("Loading anomaly scores")


            train_pred_df = pd.read_pickle(f"{self.save_path}/train_output.pkl")

            test_pred_df = pd.read_pickle(f"{self.save_path}/test_output.pkl")


            train_anomaly_scores = train_pred_df['A_Score_Global'].values

            test_anomaly_scores = test_pred_df['A_Score_Global'].values


        else:

            if cached_train_pred_df is not None:

                train_pred_df = cached_train_pred_df.copy()

                self._restore_fusion_weights(train_pred_df)

                print("Using cached training data scores, skipping redundant model inference")

            else:

                train_pred_df = self.get_score_for_sequences(train)
                train_scoring_stats = dict(self.last_scoring_stats)

            if self.use_relation_change_score and self.relation_change_mode == "normal_transition_residual":

                self._fit_relation_transition_model()

                train_pred_df["Relation_Change_Raw"] = self._relation_raw_from_segments(
                    self._last_relation_attention_segments
                )

            test_pred_df = self.get_score_for_sequences(test)
            test_scoring_stats = dict(self.last_scoring_stats)

            if self.use_physical_consistency_score:

                self._fit_physical_consistency_calibration(train_pred_df)

                train_pred_df = self._apply_physical_consistency_fusion(train_pred_df)

                test_pred_df = self._apply_physical_consistency_fusion(test_pred_df)

            if self.use_relation_change_score:

                self._fit_relation_change_calibration(train_pred_df)

                train_pred_df = self._apply_relation_change_fusion(train_pred_df)

                test_pred_df = self._apply_relation_change_fusion(test_pred_df)


            train_anomaly_scores = train_pred_df['A_Score_Global'].values

            test_anomaly_scores = test_pred_df['A_Score_Global'].values


            train_anomaly_scores = adjust_anomaly_scores(train_anomaly_scores, self.dataset, True, self.window_size)

            test_anomaly_scores = adjust_anomaly_scores(test_anomaly_scores, self.dataset, False, self.window_size)


            # 写入结果数据帧

            train_pred_df['A_Score_Global'] = train_anomaly_scores

            test_pred_df['A_Score_Global'] = test_anomaly_scores


        true_anomalies = self._prepare_true_anomalies(true_anomalies, expected_length=len(test_pred_df))


        if self.use_mov_av:

            smoothing_window = int(self.batch_size * self.window_size * 0.05)

            train_anomaly_scores = pd.DataFrame(train_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()

            test_anomaly_scores = pd.DataFrame(test_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()

            train_pred_df["A_Score_Global"] = train_anomaly_scores

            test_pred_df["A_Score_Global"] = test_anomaly_scores


        # 计算每个特征的异常预测和阈值

        out_dim = self.n_features if self.target_dims is None else len(self.target_dims)

        all_preds = np.zeros((len(test_pred_df), out_dim))

        feature_thresholds = {}

        for i in range(out_dim):

            train_feature_anom_scores = train_pred_df[f"A_Score_{i}"].values

            test_feature_anom_scores = test_pred_df[f"A_Score_{i}"].values

            epsilon = find_epsilon(train_feature_anom_scores, reg_level=2)

            feature_thresholds[f"feature_{i}"] = epsilon


            train_feature_anom_preds = (train_feature_anom_scores >= epsilon).astype(int)

            test_feature_anom_preds = (test_feature_anom_scores >= epsilon).astype(int)


            train_pred_df[f"A_Pred_{i}"] = train_feature_anom_preds

            test_pred_df[f"A_Pred_{i}"] = test_feature_anom_preds


            train_pred_df[f"Thresh_{i}"] = epsilon

            test_pred_df[f"Thresh_{i}"] = epsilon


            all_preds[:, i] = test_feature_anom_preds


        # 全局异常只使用全局分数评估

        # 特征级预测已在前面的循环中保存

        # 使用不同阈值方法评估：暴力搜索、epsilon 和 POT

        e_eval = epsilon_eval(train_anomaly_scores, test_anomaly_scores, true_anomalies, reg_level=self.reg_level)

        # 仅在存在真实标签时计算 POT 指标
        if true_anomalies is not None:

            p_eval = pot_eval(train_anomaly_scores, test_anomaly_scores, true_anomalies,

                              q=self.q, level=self.level, dynamic=self.dynamic_pot)

        else:
            # 在无监督设置下仅使用 epsilon 方法
            p_eval = {"threshold": np.percentile(train_anomaly_scores, self.level * 100) if self.level else 95}


        if true_anomalies is not None:

            bf_eval = bf_search(test_anomaly_scores, true_anomalies, start=0.01, end=2, step_num=100, verbose=False)

        else:

            bf_eval = {}


        print(f"Results using epsilon method:\n {e_eval}")

        if true_anomalies is not None:

            print(f"Results using peak-over-threshold method:\n {p_eval}")

        print(f"Results using best f1 score search:\n {bf_eval}")


        e_eval = self._to_serializable_dict(e_eval)

        p_eval = self._to_serializable_dict(p_eval)

        bf_eval = self._to_serializable_dict(bf_eval)

        if true_anomalies is None and str(self.dataset).upper() == "BMS":

            global_epsilon = float(np.quantile(
                train_anomaly_scores, self.normal_threshold_quantile
            ))

            e_eval["threshold"] = global_epsilon

            e_eval["threshold_quantile"] = self.normal_threshold_quantile

        else:

            global_epsilon = (
                e_eval["threshold"]
                if "threshold" in e_eval
                else np.percentile(train_anomaly_scores, 95)
            )

        event_low_threshold = self._compute_event_low_threshold(train_anomaly_scores, global_epsilon)


        raw_train_preds = (train_anomaly_scores >= global_epsilon).astype(int)

        raw_test_preds = (test_anomaly_scores >= global_epsilon).astype(int)

        event_train_preds = self._apply_event_consistency(train_pred_df, global_epsilon, event_low_threshold)

        event_test_preds = self._apply_event_consistency(test_pred_df, global_epsilon, event_low_threshold)


        event_report = {

            "enabled": bool(self.use_event_consistency),

            "high_threshold": float(global_epsilon),

            "low_threshold": float(event_low_threshold),

            "low_ratio": float(self.event_low_ratio),

            "min_event_length": int(self.event_min_length),

            "legacy_point_adjusted_raw_threshold_result": self._evaluate_binary_predictions(

                raw_test_preds, true_anomalies

            ),

            "legacy_point_adjusted_persistence_result": self._evaluate_binary_predictions(

                event_test_preds, true_anomalies

            ),

            "raw_event_result": self._evaluate_raw_events(

                raw_test_preds, true_anomalies, score_df=test_pred_df

            ),

            "persistence_filtered_raw_event_result": self._evaluate_raw_events(

                event_test_preds, true_anomalies, score_df=test_pred_df

            ),

        }

        raw_metric_report = self._evaluate_raw_scores(test_anomaly_scores, raw_test_preds, true_anomalies)

        # AP is the ranking metric shared with the Tsinghua protocol.  Print
        # it explicitly so external-run logs do not expose threshold-F1 only.
        if raw_metric_report:
            ap = raw_metric_report["average_precision"]
            auroc = raw_metric_report.get("auroc")
            suffix = "" if auroc is None else f", AUROC={auroc:.6f}"
            print(f"Raw point-level ranking metrics: AP={ap:.6f}{suffix}", flush=True)
        else:
            print(
                "Raw point-level ranking metrics unavailable: no usable binary anomaly labels.",
                flush=True,
            )

        self._save_standard_reports(

            e_eval,

            p_eval,

            bf_eval,

            feature_thresholds,

            global_epsilon,

            extra_reports={

                "event_consistency_result": event_report,

                "raw_point_result": raw_metric_report,

                "model_parameters": int(self.model_parameters),

                "inference_efficiency": {
                    "train": train_scoring_stats,
                    "test": test_scoring_stats,
                    "total_inference_seconds": float(
                        train_scoring_stats.get("inference_seconds", 0.0)
                        + test_scoring_stats.get("inference_seconds", 0.0)
                    ),
                },

                "runtime": {
                    "model_parameters": int(self.model_parameters),
                    "preprocessing_seconds": float(self.preprocessing_seconds),
                    "training": self.training_runtime,
                    "inference": {
                        "train": train_scoring_stats,
                        "test": test_scoring_stats,
                    },
                },

            },

        )


        # 保存使用 epsilon 方法得到的异常预测（可改为 POT 或暴力搜索方法）

        if save_output:

            if true_anomalies is not None:

                test_pred_df["A_True_Global"] = true_anomalies

            train_pred_df["Thresh_Global"] = global_epsilon

            test_pred_df["Thresh_Global"] = global_epsilon

            train_pred_df["Thresh_Global_Low"] = event_low_threshold

            test_pred_df["Thresh_Global_Low"] = event_low_threshold

            train_pred_df["A_Pred_Global_Raw"] = raw_train_preds

            test_pred_df["A_Pred_Global_Raw"] = raw_test_preds

            train_pred_df["A_Pred_Global_Event"] = event_train_preds

            test_pred_df["A_Pred_Global_Event"] = event_test_preds


            train_pred_df[f"A_Pred_Global"] = event_train_preds if self.use_event_consistency else raw_train_preds

            test_preds_global = event_test_preds.copy() if self.use_event_consistency else raw_test_preds.copy()


            # 仅在存在真实标签时计算评估指标

            if true_anomalies is not None:

                test_pred_df["A_Pred_Global_PointAdjusted"] = adjust_predicts(

                    None,

                    true_anomalies,

                    global_epsilon,

                    pred=test_preds_global.copy(),

                )

            test_pred_df[f"A_Pred_Global"] = test_preds_global


            print(f"Saving output to {self.save_path}/<train/test>_output.pkl")

            train_pred_df.to_pickle(f"{self.save_path}/train_output.pkl")

            test_pred_df.to_pickle(f"{self.save_path}/test_output.pkl")

            self._save_segment_visualizations(train_pred_df, test_pred_df, global_epsilon)


        return test_pred_df
