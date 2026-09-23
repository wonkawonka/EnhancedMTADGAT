"""Load a frozen RCR bundle and score raw or already normalized windows."""
import json
from pathlib import Path
import numpy as np
import torch
from src.models.rcr import RCRBackbone, RCRLite, MaskedResidualGraphNet, EmpiricalCDF, condition_features


class FrozenRCRDetector:
    def __init__(self, root, unit='net_continuous_mlp', device='cpu'):
        self.root, self.unit, self.device = Path(root), unit, torch.device(device)
        spec = json.loads((self.root/'protocol.json').read_text())
        mapping = json.loads((self.root/'mapping.json').read_text())
        self.columns = mapping['feature_columns']
        self.control = mapping['control_indices']
        self.backbone = RCRBackbone(len(self.columns), feature_graph=spec['feature_graph'], temporal=spec['temporal']).to(self.device)
        self.backbone.load_state_dict(torch.load(self.root/'backbone_best.pt', map_location=self.device, weights_only=True))
        self.backbone.eval().requires_grad_(False)
        self.head = None
        directory = self.root/unit
        assert (directory/'complete.json').exists()
        if unit.startswith('net_'):
            config = json.loads((directory/'protocol.json').read_text())
            self.mode = config['condition']
            self.lite = RCRLite.load(self.root/('lite_'+self.mode))
            self.head = MaskedResidualGraphNet(len(self.columns), condition_mode=config['modulation'],
                        condition_dim=len(self.lite.calibration.q_mean)).to(self.device)
            self.head.load_state_dict(torch.load(directory/'best.pt', map_location=self.device, weights_only=True))
            self.head.eval().requires_grad_(False)
            with np.load(directory/'score_calibration.npz') as calibration:
                self.relation_cdf = EmpiricalCDF().fit(calibration['relation_reference'])
        else:
            self.mode = unit.removeprefix('lite_')
            self.lite = RCRLite.load(directory)
        scaler = self.root.parent/'input_normalization.npz'
        self.input_scaler = None
        if scaler.exists():
            with np.load(scaler) as data:
                assert data['feature_columns'].tolist() == self.columns
                self.input_scaler = {k: data[k].copy() for k in ['scale', 'offset']}

    @torch.inference_mode()
    def score_windows(self, x, *, normalized=True, cycles=None, details=False):
        # Match contiguous training batches even when pandas supplied F-strided views.
        values = np.array(x, dtype=np.float32, copy=True, order='C')
        if values.ndim != 3 or values.shape[1:] != (32, len(self.columns)) or not np.isfinite(values).all():
            raise ValueError('Finite complete windows [B,32,D] required')
        if not normalized:
            if self.input_scaler is None:
                raise RuntimeError('Input MinMax archive required for raw-window inference')
            values *= self.input_scaler['scale']
            values += self.input_scaler['offset']
        xt = torch.as_tensor(values, device=self.device)
        residual = (xt-self.backbone(xt)).cpu().numpy()
        q = condition_features(xt, *self.control).cpu().numpy()
        if self.mode == 'none':
            q = np.zeros_like(q)
        elif self.mode == 'cycle':
            if cycles is None or len(cycles) != len(values) or not set(cycles) <= {'charge', 'discharge'}:
                raise ValueError('One-hot ablation requires explicit cycle metadata')
            q = np.eye(2, dtype=np.float32)[[int(c == 'discharge') for c in cycles]]
        result = self.lite.score(q, residual, return_residual=details or self.head is not None)
        if self.head is not None:
            z = result['standardized_residual'].astype(np.float32)
            cq = ((q-self.lite.calibration.q_mean)/self.lite.calibration.q_std).astype(np.float32)
            zt = torch.as_tensor(z, device=self.device).flatten(0, 1)
            ct = torch.as_tensor(cq, device=self.device)[:, None].expand(-1, 32, -1).flatten(0, 1)
            error = (zt-self.head.leave_one_out(zt, ct)).square().reshape(z.shape).cpu().numpy()
            result['relation'] = error.mean((1, 2), dtype=np.float64)
            result['relation_percentile'] = self.relation_cdf.transform(result['relation'])
            result['combined'] = .5*(result['magnitude_percentile']+result['relation_percentile'])
            if details:
                result['loo_squared_error'] = error
        if not details:
            result.pop('residual', None); result.pop('standardized_residual', None)
        return result
