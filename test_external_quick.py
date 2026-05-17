"""
Quick validation for Kaggle: each external baseline does 1 forward + 1 backward
pass with tiny synthetic data. Runs each model in a separate subprocess to
avoid import conflicts.

Usage on Kaggle:
  import sys, subprocess
  result = subprocess.run([sys.executable, "/kaggle/working/test_external_quick.py"], capture_output=True, text=True)
  print(result.stdout)
  if result.returncode != 0: print(result.stderr)

Or directly:  python test_external_quick.py
"""
import sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = sys.executable

models = [
    ("GANF", """
import sys, torch; from pathlib import Path
sys.path.insert(0, str(Path(r'{root}') / 'external_baselines' / 'GANF'))
from models.GANF import GANF
B, K, L, D = 4, 5, 10, 1
x = torch.randn(B, K, L, D)
A = torch.eye(K).float()
m = GANF(n_blocks=2, input_size=1, hidden_size=8, n_hidden=1, model="MAF")
loss = m(x, A); loss.backward()
print("OK")
"""),
    ("TranAD", """
import sys, torch; from pathlib import Path
sys.path.insert(0, str(Path(r'{root}') / 'external_baselines' / 'TranAD'))
from src.models import TranAD
m = TranAD(4)
bs, nw = 2, 10
src = torch.randn(bs, nw, 4)
tgt = torch.randn(bs, nw, 4)
x1, x2 = m(src, tgt)
loss = (x1.mean() + x2.mean()) / 2; loss.backward()
print("OK")
"""),
    ("Anomaly-Transformer", """
import sys, torch; from pathlib import Path
sys.path.insert(0, str(Path(r'{root}') / 'external_baselines' / 'Anomaly-Transformer'))
from model.AnomalyTransformer import AnomalyTransformer
m = AnomalyTransformer(win_size=10, enc_in=8, c_out=8, d_model=64, e_layers=1)
x = torch.randn(2, 10, 8)
out = m(x)
loss = out[0].mean() if isinstance(out, (list,tuple)) else out.mean()
loss.backward()
print("OK")
"""),
    ("DCdetector", """
import sys, torch; from pathlib import Path
sys.path.insert(0, str(Path(r'{root}') / 'external_baselines' / 'DCdetector'))
from model.DCdetector import DCdetector
m = DCdetector(win_size=105, enc_in=4, c_out=4, channel=4, d_model=32, e_layers=1, patch_size=[3,5,7])
x = torch.randn(2, 105, 4)
series_list, prior_list = m(x)
loss = torch.tensor(0., requires_grad=True)
for t in series_list + prior_list:
    if isinstance(t, torch.Tensor):
        loss = loss + t.mean()
loss.backward()
print("OK")
"""),
    ("GDN", """
import sys, torch; from pathlib import Path
root = Path(r'{root}')
gdn_root = root / 'external_baselines' / 'GDN'
sys.path.insert(0, str(gdn_root))
from util.env import set_device
set_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
from models.GDN import GDN
node_num = 4
edge_index = torch.tensor([[0,1,2,3],[1,2,3,0]], dtype=torch.long)
m = GDN(edge_index_sets=[edge_index], node_num=node_num, dim=8, input_dim=10, out_layer_inter_dim=16, topk=2)
x = torch.randn(2, node_num, 10)
out = m(x, edge_index)
loss = out.mean(); loss.backward()
print("OK")
"""),
]

print("=" * 60)
print("Validating External Baselines (tiny data)")
print(f"Python: {sys.executable}")
print("=" * 60)

pass_count = 0
fail_count = 0

for name, code in models:
    code = code.format(root=str(ROOT))
    cmd = [str(VENV_PY), "-c", code]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode == 0:
        print(f"  [PASS] {name}")
        pass_count += 1
    else:
        stderr = result.stderr.strip().split("\n")[-3:]
        print(f"  [FAIL] {name}: {'; '.join(s for s in stderr if s)}")
        fail_count += 1

print(f"\n{'=' * 60}")
print(f"Result: {pass_count} passed, {fail_count} failed")
if fail_count == 0:
    print("All 5 external baselines passed!")
print(f"{'=' * 60}")
