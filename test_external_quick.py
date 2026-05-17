"""
Quick validation: each external baseline does 1 forward + 1 backward pass
with tiny synthetic data. Runs inline (no subprocess) to avoid path issues.
Usage: python test_external_quick.py
"""
import sys, importlib
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent
errors = []

def _clean():
    keys = [k for k in list(sys.modules.keys()) if "models" in k or "model" in k
            or "GANF" in k or "TranAD" in k or "GDN" in k or "DCdetector" in k
            or "AnomalyTransformer" in k or "dgl" in k or "torch_geometric" in k]
    for k in keys:
        del sys.modules[k]
    sys.path = [p for p in sys.path if "external_baselines" not in p]

def ok(name):
    print(f"  [PASS] {name}")

def fail(name, e):
    print(f"  [FAIL] {name}: {e}")
    errors.append(name)

print("=" * 60)
print("Validating External Baselines (tiny data)")
print(f"Python: {sys.executable}")
print("=" * 60)

# ── 1. GANF ────────────────────────────────────
print("\n--- GANF ---")
_clean()
try:
    sys.path.insert(0, str(ROOT / "external_baselines" / "GANF"))
    import models.GANF as ganf_mod
    importlib.reload(ganf_mod)
    from models.GANF import GANF
    B, K, L, D = 4, 5, 10, 1
    x = torch.randn(B, K, L, D)
    A = torch.eye(K).float()
    m = GANF(n_blocks=2, input_size=1, hidden_size=8, n_hidden=1, model="MAF")
    loss = m(x, A); loss.backward()
    ok("GANF forward + backward")
except Exception as e:
    fail("GANF", e)

# ── 2. TranAD ──────────────────────────────────
print("\n--- TranAD ---")
_clean()
try:
    sys.path.insert(0, str(ROOT / "external_baselines" / "TranAD"))
    from src.models import TranAD
    m = TranAD(4)
    bs, nw = 2, 10
    src = torch.randn(bs, nw, 4); tgt = torch.randn(bs, nw, 4)
    x1, x2 = m(src, tgt)
    loss = (x1.mean() + x2.mean()) / 2; loss.backward()
    ok("TranAD forward + backward")
except Exception as e:
    fail("TranAD", e)

# ── 3. Anomaly-Transformer ─────────────────────
print("\n--- Anomaly-Transformer ---")
_clean()
try:
    sys.path.insert(0, str(ROOT / "external_baselines" / "Anomaly-Transformer"))
    from model.AnomalyTransformer import AnomalyTransformer
    m = AnomalyTransformer(win_size=10, enc_in=8, c_out=8, d_model=64, e_layers=1)
    x = torch.randn(2, 10, 8)
    out = m(x)
    loss = out[0].mean() if isinstance(out, (list,tuple)) else out.mean()
    loss.backward()
    ok("Anomaly-Transformer forward + backward")
except Exception as e:
    fail("Anomaly-Transformer", e)

# ── 4. DCdetector ──────────────────────────────
print("\n--- DCdetector ---")
_clean()
try:
    sys.path.insert(0, str(ROOT / "external_baselines" / "DCdetector"))
    from model.DCdetector import DCdetector
    m = DCdetector(win_size=105, enc_in=4, c_out=4, channel=4, d_model=32, e_layers=1, patch_size=[3,5,7])
    x = torch.randn(2, 105, 4)
    series_list, prior_list = m(x)
    loss = torch.tensor(0., requires_grad=True)
    for t in series_list + prior_list:
        if isinstance(t, torch.Tensor):
            loss = loss + t.mean()
    loss.backward()
    ok("DCdetector forward + backward")
except Exception as e:
    fail("DCdetector", e)

# ── 5. GDN ─────────────────────────────────────
print("\n--- GDN ---")
_clean()
try:
    sys.path.insert(0, str(ROOT / "external_baselines" / "GDN"))
    from util.env import set_device
    set_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    from models.GDN import GDN
    node_num = 4
    edge_index = torch.tensor([[0,1,2,3],[1,2,3,0]], dtype=torch.long)
    m = GDN(edge_index_sets=[edge_index], node_num=node_num, dim=8, input_dim=10, out_layer_inter_dim=16, topk=2)
    x = torch.randn(2, node_num, 10)
    out = m(x, edge_index)
    loss = out.mean(); loss.backward()
    ok("GDN forward + backward")
except Exception as e:
    fail("GDN", e)

print(f"\n{'=' * 60}")
if errors:
    print(f"FAILED: {len(errors)} model(s): {errors}")
else:
    print("All 5 external baselines passed!")
print(f"{'=' * 60}")
