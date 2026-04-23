import json
from tqdm import tqdm
from eval_methods import *
from utils import *


def _autocast_context(device):
    enabled = device == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


class Predictor:
    """MTAD-GAT predictor class.

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
        self.scale_scores = pred_args["scale_scores"]
        self.q = pred_args["q"]
        self.level = pred_args["level"]
        self.dynamic_pot = pred_args["dynamic_pot"]
        self.use_mov_av = pred_args["use_mov_av"]
        self.gamma = pred_args["gamma"]
        self.reg_level = pred_args["reg_level"]
        self.save_path = pred_args["save_path"]
        self.batch_size = 256
        self.use_cuda = True
        self.pred_args = pred_args
        self.summary_file_name = summary_file_name

    @staticmethod
    def _to_serializable_dict(metrics):
        serializable = {}
        for k, v in metrics.items():
            if isinstance(v, list):
                serializable[k] = v
            else:
                serializable[k] = float(v)
        return serializable

    def _save_standard_reports(self, e_eval, p_eval, bf_eval, feature_thresholds, global_threshold):
        summary = {"epsilon_result": e_eval, "pot_result": p_eval, "bf_result": bf_eval}
        with open(f"{self.save_path}/{self.summary_file_name}", "w") as f:
            json.dump(summary, f, indent=2)

        with open(f"{self.save_path}/summary_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

        thresholds = {
            "global_threshold": float(global_threshold),
            "feature_thresholds": {k: float(v) for k, v in feature_thresholds.items()},
        }
        with open(f"{self.save_path}/thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=2)

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
        """Method that calculates anomaly score using given model and data
        :param values: 2D array of multivariate time series data, shape (N, k)
        :return np array of anomaly scores + dataframe with prediction for each channel and global anomalies
        """

        print("Predicting and calculating anomaly scores..")
        data = SlidingWindowDataset(values, self.window_size, self.target_dims)
        
        # 优化 DataLoader
        num_workers = 2 if os.name == 'nt' else 4
        pin_memory = torch.cuda.is_available()
        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
        
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

        self.model.eval()
        preds = []
        recons = []
        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(device)
                y = y.to(device)

                with _autocast_context(device):
                    y_hat, _ = self.model(x)

                    # Shifting input to include the observed value (y) when doing the reconstruction
                    recon_x = torch.cat((x[:, 1:, :], y), dim=1)
                    _, window_recon = self.model(recon_x)

                preds.append(y_hat.detach().cpu().numpy())
                # Extract last reconstruction only
                recons.append(window_recon[:, -1, :].detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        recons = np.concatenate(recons, axis=0)
        actual = values.detach().cpu().numpy()[self.window_size:]

        if self.target_dims is not None:
            actual = actual[:, self.target_dims]
        #TODO 计算异常分数需要加图结构权重吗？
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
            a_score = pred_error + self.gamma * recon_error

            if self.scale_scores:
                q75, q25 = np.percentile(a_score, [75, 25])
                iqr = q75 - q25
                median = np.median(a_score)
                a_score = (a_score - median) / (1+iqr)

            anomaly_scores[:, i] = a_score
            pred_errors[:, i] = pred_error
            recon_errors[:, i] = recon_error
            df_dict[f"Pred_Error_{i}"] = pred_error
            df_dict[f"Recon_Error_{i}"] = recon_error
            df_dict[f"A_Score_{i}"] = a_score

        df = pd.DataFrame(df_dict)
        df['Pred_Error_Global'] = np.mean(pred_errors, axis=1)
        df['Recon_Error_Global'] = np.mean(recon_errors, axis=1)
        anomaly_scores = np.mean(anomaly_scores, 1)
        df['A_Score_Global'] = anomaly_scores

        return df

    def get_score_for_sequences(self, values):
        if isinstance(values, dict):
            score_dfs = []
            for _, entity_values in values.items():
                score_dfs.append(self.get_score_for_sequences(entity_values))
            if not score_dfs:
                raise ValueError("No sequence data provided for scoring")
            return pd.concat(score_dfs, axis=0, ignore_index=True)
        if is_sequence_container(values):
            score_dfs = []
            global_start_index = 0
            for segment_id, seq_values in enumerate(ensure_sequence_list(values)):
                if len(seq_values) <= self.window_size:
                    continue
                segment_score_df = self.get_score(seq_values)
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
                    aligned_parts.append(seq_labels[self.window_size:])
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

        raise ValueError(
            f"True anomaly labels length {len(true_anomalies)} does not match expected score length {expected_length}"
        )

    def predict_anomalies(self, train, test, true_anomalies=None, load_scores=False, save_output=True,
                          scale_scores=False):
        """ Predicts anomalies

        :param train: 2D array of train multivariate time series data (normal data used to establish baseline)
        :param test: 2D array of test multivariate time series data (data to be evaluated for anomalies)
        :param true_anomalies: true anomalies of test set, None if not available (for unsupervised setting)
        :param save_scores: Whether to save anomaly scores of train and test
        :param load_scores: Whether to load anomaly scores instead of calculating them
        :param save_output: Whether to save output dataframe
        :param scale_scores: Whether to feature-wise scale anomaly scores
        """

        if load_scores:
            print("Loading anomaly scores")

            train_pred_df = pd.read_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df = pd.read_pickle(f"{self.save_path}/test_output.pkl")

            train_anomaly_scores = train_pred_df['A_Score_Global'].values
            test_anomaly_scores = test_pred_df['A_Score_Global'].values

        else:
            train_pred_df = self.get_score_for_sequences(train)
            test_pred_df = self.get_score_for_sequences(test)

            train_anomaly_scores = train_pred_df['A_Score_Global'].values
            test_anomaly_scores = test_pred_df['A_Score_Global'].values

            train_anomaly_scores = adjust_anomaly_scores(train_anomaly_scores, self.dataset, True, self.window_size)
            test_anomaly_scores = adjust_anomaly_scores(test_anomaly_scores, self.dataset, False, self.window_size)

            # Update df
            train_pred_df['A_Score_Global'] = train_anomaly_scores
            test_pred_df['A_Score_Global'] = test_anomaly_scores

        true_anomalies = self._prepare_true_anomalies(true_anomalies, expected_length=len(test_pred_df))

        if self.use_mov_av:
            smoothing_window = int(self.batch_size * self.window_size * 0.05)
            train_anomaly_scores = pd.DataFrame(train_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()
            test_anomaly_scores = pd.DataFrame(test_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()

        # Find threshold and predict anomalies at feature-level (for plotting and diagnosis purposes)
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

        # Global anomalies (entity-level) are predicted using aggregation of anomaly scores across all features
        # These predictions are used to evaluate performance, as true anomalies are labeled at entity-level
        # Evaluate using different threshold methods: brute-force, epsilon and peaks-over-treshold
        e_eval = epsilon_eval(train_anomaly_scores, test_anomaly_scores, true_anomalies, reg_level=self.reg_level)
        # 只在有真实标签时才运行POT方法，避免在无标签数据上出现数值问题
        if true_anomalies is not None:
            p_eval = pot_eval(train_anomaly_scores, test_anomaly_scores, true_anomalies,
                              q=self.q, level=self.level, dynamic=self.dynamic_pot)
        else:
            # 在无监督设置中，只使用epsilon方法
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
        global_epsilon = e_eval["threshold"] if "threshold" in e_eval else np.percentile(train_anomaly_scores, 95)
        self._save_standard_reports(e_eval, p_eval, bf_eval, feature_thresholds, global_epsilon)

        # Save anomaly predictions made using epsilon method (could be changed to pot or bf-method)
        if save_output:
            if true_anomalies is not None:
                test_pred_df["A_True_Global"] = true_anomalies
            train_pred_df["Thresh_Global"] = global_epsilon
            test_pred_df["Thresh_Global"] = global_epsilon
            train_pred_df[f"A_Pred_Global"] = (train_anomaly_scores >= global_epsilon).astype(int)
            test_preds_global = (test_anomaly_scores >= global_epsilon).astype(int)
            # Adjust predictions according to evaluation strategy
            if true_anomalies is not None:
                test_preds_global = adjust_predicts(None, true_anomalies, global_epsilon, pred=test_preds_global)
            test_pred_df[f"A_Pred_Global"] = test_preds_global

            print(f"Saving output to {self.save_path}/<train/test>_output.pkl")
            train_pred_df.to_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df.to_pickle(f"{self.save_path}/test_output.pkl")
            self._save_segment_visualizations(train_pred_df, test_pred_df, global_epsilon)

        print("-- Done.")
