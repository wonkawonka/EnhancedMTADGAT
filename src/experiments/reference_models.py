"""Factories for pinned reference implementations; no timing-runner dependency."""
import ast
import importlib
import importlib.util
import math
import sys
import types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.models.rcr import RCRBackbone, LinearResidualCalibration, condition_features
from src.models.omni_anomaly_port import OmniAnomalyPort, OmniConfig

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'external_baselines/reference_snapshots'


def extract(path, name, namespace):
    node = next(n for n in ast.parse(Path(path).read_text()).body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build(method, device, architecture):
    checks = {}
    if method == 'rcr':
        names = ('hidden', 'node_dim', 'feature_graph', 'temporal', 'layers', 'heads', 'ffn_dim', 'dropout')
        model = RCRBackbone(7, **{k: architecture[k] for k in names})
    elif method == 'omni':
        model = OmniAnomalyPort(OmniConfig(), semantics='release')
    elif method == 'svdd':
        namespace = {'nn': nn}
        source = ROOT/'src/models/unified_external_baselines.py'
        extract(source, 'DeepSVDDEncoder', namespace)
        cls = extract(source, 'DeepSVDDAutoEncoder', namespace)
        model = cls(architecture['window']*7, 64, 32)
    elif method == 'tranad':
        module = file_module('formal_tranad_dlutils', SOURCE/'tranad/src/dlutils.py')
        encoder_forward = module.TransformerEncoderLayer.forward
        decoder_forward = module.TransformerDecoderLayer.forward
        def encoder(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
            assert not is_causal
            return encoder_forward(self, src, src_mask, src_key_padding_mask)
        def decoder(self, tgt, memory, tgt_mask=None, memory_mask=None,
                    tgt_key_padding_mask=None, memory_key_padding_mask=None,
                    tgt_is_causal=False, memory_is_causal=False):
            assert not tgt_is_causal and not memory_is_causal
            return decoder_forward(self, tgt, memory, tgt_mask, memory_mask,
                                   tgt_key_padding_mask, memory_key_padding_mask)
        module.TransformerEncoderLayer.forward = encoder
        module.TransformerDecoderLayer.forward = decoder
        ns = dict(nn=nn, torch=torch, math=math, lr=1e-4,
                  TransformerEncoder=nn.TransformerEncoder, TransformerDecoder=nn.TransformerDecoder,
                  PositionalEncoding=module.PositionalEncoding,
                  TransformerEncoderLayer=module.TransformerEncoderLayer,
                  TransformerDecoderLayer=module.TransformerDecoderLayer)
        model = extract(SOURCE/'tranad/src/models.py', 'TranAD', ns)(7).double().to(device)
        layer = model.transformer_encoder.layers[0]
        layer.eval()
        x = torch.linspace(0, 1, 280, dtype=torch.float64, device=device).reshape(10, 2, 14)
        a, b = encoder_forward(layer, x), encoder(layer, x, is_causal=False)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        ga = torch.autograd.grad(a.sum(), tuple(layer.parameters()))
        gb = torch.autograd.grad(b.sum(), tuple(layer.parameters()))
        for u, v in zip(ga, gb):
            torch.testing.assert_close(u, v, atol=0, rtol=0)
        checks['legacy_encoder_forward_gradient'] = 'passed'
    elif method == 'mtad':
        file_module('modules', SOURCE/'mtad/modules.py')
        module = file_module('formal_mtad', SOURCE/'mtad/mtad_gat.py')
        model = module.MTAD_GAT(7, 100, 7, kernel_size=7, use_gatv2=True,
                               gru_n_layers=1, gru_hid_dim=150, forecast_n_layers=3,
                               forecast_hid_dim=150, recon_n_layers=1, recon_hid_dim=150,
                               dropout=.3, alpha=.2)
    elif method in ('at', 'dc'):
        name = 'formal_' + method
        package = types.ModuleType(name)
        package.__path__ = [str(SOURCE/method/'model')]
        sys.modules[name] = package
        if method == 'at':
            cls = importlib.import_module(name+'.AnomalyTransformer').AnomalyTransformer
            model = cls(100, 7, 7, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                        dropout=0., activation='gelu', output_attention=True)
        else:
            cls = importlib.import_module(name+'.DCdetector').DCdetector
            model = cls(105, 7, 7, n_heads=1, d_model=256, e_layers=3,
                        patch_size=[3, 5, 7], channel=7, dropout=0.)
    else:
        raise ValueError(method)
    return model.to(device), checks


def noise(model, x, generator=None):
    return torch.cat([model._noise(group, generator) for group in x.split(50)], 0)


def kl_for(method):
    return extract(SOURCE/method/'solver.py', 'my_kl_loss', {'torch': torch})


def association(series, prior, kl):
    sl, pl = 0., 0.
    for s, p in zip(series, prior):
        p = p / p.sum(-1, keepdim=True)
        sl = sl + kl(s, p.detach()).mean() + kl(p.detach(), s).mean()
        pl = pl + kl(p, s.detach()).mean() + kl(s.detach(), p).mean()
    return sl/len(prior), pl/len(prior)


def terms(method, model, x, epoch, kl=None, center=None, stage='main', generator=None):
    if method == 'rcr':
        return [F.huber_loss(model(x), x, delta=1.)]
    if method == 'svdd':
        if stage == 'pretrain':
            return [F.mse_loss(model(x), x)]
        return [(model.encoder(x)-center).square().sum(-1).mean()]
    if method == 'mtad':
        forecast, rec = model(x[:, :100])
        return [F.mse_loss(forecast, x[:, 100]).sqrt()+F.mse_loss(rec, x[:, :100]).sqrt()]
    if method == 'tranad':
        seq = x.transpose(0, 1)
        first, second = model(seq, seq[-1:])
        return [F.mse_loss(first, seq[-1:])/epoch + (1-1/epoch)*F.mse_loss(second, seq[-1:])]
    if method == 'omni':
        return [model.loss(model(x, noise=noise(model, x, generator)))]
    if method in ('at', 'dc'):
        if method == 'at':
            rec, series, prior, _ = model(x)
        else:
            series, prior = model(x)
        sl, pl = association(series, prior, kl)
        if method == 'dc':
            return [pl-sl]
        mse = F.mse_loss(rec, x)
        return [mse-3*sl, mse+3*pl]
    raise ValueError(method)


def scores(method, model, x, kl=None, calibration=None, center=None, generator=None):
    if method == 'rcr':
        r = (x-model(x)).cpu().numpy().astype(np.float64)
        if calibration is None:
            values = np.square(r).mean((1, 2))
        else:
            q = condition_features(x, 1, 2).cpu().numpy()
            values = np.square(calibration.transform(q, r)).mean((1, 2))
        return torch.as_tensor(values, device=x.device)
    if method == 'svdd':
        return (model.encoder(x)-center).square().sum(-1)
    if method == 'mtad':
        forecast, _ = model(x[:, :100])
        _, rec = model(x[:, 1:])
        return ((forecast-x[:, 100]).abs()+(rec[:, -1]-x[:, 100]).abs()).mean(-1)
    if method == 'tranad':
        seq = x.transpose(0, 1)
        _, second = model(seq, seq[-1:])
        return (second-seq[-1:]).square().mean(-1).reshape(-1)
    if method == 'omni':
        return model.anomaly_score(model(x, noise=noise(model, x, generator)))
    if method == 'at':
        rec, series, prior, _ = model(x)
    else:
        series, prior = model(x)
    forward, reverse = 0., 0.
    for s, p in zip(series, prior):
        p = p/p.sum(-1, keepdim=True)
        forward = forward+50*kl(s, p.detach())
        reverse = reverse+50*kl(p, s.detach())
    weights = torch.softmax(-forward-reverse, dim=-1)
    return weights*(rec-x).square().mean(-1) if method == 'at' else weights


def fit_scale(q, moments, floor=.05):
    # Use the existing estimator verbatim, not another fitted scale definition.
    return extract(ROOT/'src/runners/run_rcr_revalidation.py', 'fit_scale',
                   {'np': np, 'LinearResidualCalibration': LinearResidualCalibration})(q, moments, floor)
