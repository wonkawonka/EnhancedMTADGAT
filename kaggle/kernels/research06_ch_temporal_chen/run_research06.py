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
cases=[(f"lookback_{n}",n,4,1) for n in (8,16,32,64)]+[(f"stride_{n}",32,n,1) for n in (1,4,8)]+[(f"resample_{n}",32,4,n) for n in (1,2,4)]
for seed in (3407,3408,3409):
 for name,lookback,stride,resample in cases:
  run([sys.executable,"-m","src.runners.run_unified_external_baselines","--method","mtad_gat","--dataset","CH_LFP_DISCHARGE","--seed",str(seed),"--lookback",str(lookback),"--window_stride",str(stride),"--ch_resample_factor",str(resample),"--batch_size","64","--epochs","10","--learning_rate","0.001","--hidden_dim","64","--train_sample_limit","100000","--use_cuda","true","--require_cuda","--output_dir",str(r/"runs"/"research_06"/name/f"seed{seed}")])
print("Results:",shutil.make_archive(str(r/"research06_ch_temporal_results"),"zip",root_dir=r/"runs"/"research_06"))
