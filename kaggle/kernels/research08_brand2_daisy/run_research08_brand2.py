from pathlib import Path
import os, shutil, subprocess, sys
import torch
def run(c, env=None): print("$", " ".join(map(str,c)),flush=True); subprocess.run(c,check=True,env=env)
def find_root():
 for p in Path("/kaggle/input").rglob("battery_brand2"):
  if (p/"label").is_dir() and any((p/x).is_dir() for x in ("train","test","data")): return p.parent
 raise FileNotFoundError("battery_brand2")
assert torch.cuda.is_available(), "GPU must be enabled"
r=Path("/kaggle/working/EnhancedMTADGAT")
if r.exists(): shutil.rmtree(r)
run(["git","clone","--depth","1","--branch","main","https://github.com/wonkawonka/EnhancedMTADGAT.git",str(r)])
os.chdir(r); run([sys.executable,"-m","pip","install","-q","-r","requirements-kaggle-main.txt"])
env=os.environ.copy(); env["MTAD_GAT_TSINGHUA_EV_ROOT"]=str(find_root()); env["MTAD_GAT_RUNS_ROOT"]=str(r/"runs")
for seed in (3407,3408,3409):
 for fold in range(5):
  out=r/"runs"/"research_08_brand2"/f"fold{fold}_seed{seed}"; out.mkdir(parents=True,exist_ok=True); e=env.copy(); e["PLAN_OUTPUT_DIR"]=str(out)
  run([sys.executable,"-m","src.runners.train_nc_battery","--battery_brand","2","--battery_fold",str(fold),"--battery_fold_seed","0","--seed",str(seed),"--battery_split_protocol","strict_normal_validation","--battery_normalization","minmax","--battery_vehicle_top_ratio","0.05","--battery_vehicle_top_ratio_mode","fixed","--battery_windows_per_snippet","1","--lookback","127","--epochs","10","--bs","64","--init_lr","0.001","--dropout","0.3","--model_name","mtad_gat","--use_regime_condition","false","--use_cuda","true","--require_cuda","true","--deterministic","true","--num_workers","2","--predict_num_workers","0","--persistent_workers","true","--early_stopping_patience","0","--log_tensorboard","false"],e)
print("Results:",shutil.make_archive(str(r/"research08_brand2_results"),"zip",root_dir=r/"runs"/"research_08_brand2"))
