from pathlib import Path
import os, shutil, subprocess, sys
import torch
def run(c): print("$", " ".join(map(str,c)),flush=True); subprocess.run(c,check=True)
assert torch.cuda.is_available(), "GPU must be enabled"
r=Path("/kaggle/working/EnhancedMTADGAT")
if r.exists(): shutil.rmtree(r)
run(["git","clone","--depth","1","--branch","main","https://github.com/wonkawonka/EnhancedMTADGAT.git",str(r)])
os.chdir(r); run([sys.executable,"-m","pip","install","-q","-r","requirements-kaggle-main.txt"])
s=list(Path("/kaggle/input").rglob("processed/lfp_discharge")); assert len(s)==1,s
os.environ["MTAD_GAT_CH_BATTERY_ROOT"]=str(s[0].parent.parent); os.environ["MTAD_GAT_RUNS_ROOT"]=str(r/"runs")
cases={"full":"","controls_only":"1,2","responses_only":"0,3,4,5,6","drop_voltage_extrema":"0,1,2,5,6","drop_temperature_extrema":"0,1,2,3,4"}
for seed in (3407,3408,3409):
 for name,dims in cases.items():
  c=[sys.executable,"-m","src.runners.run_unified_external_baselines","--method","mtad_gat","--dataset","CH_LFP_DISCHARGE","--seed",str(seed),"--lookback","32","--window_stride","4","--batch_size","64","--epochs","10","--learning_rate","0.001","--hidden_dim","64","--train_sample_limit","100000","--use_cuda","true","--require_cuda","--output_dir",str(r/"runs"/"research_05"/name/f"seed{seed}")]
  if dims: c += ["--feature_dims",dims]
  run(c)
print("Results:",shutil.make_archive(str(r/"research05_ch_schema_results"),"zip",root_dir=r/"runs"/"research_05"))
