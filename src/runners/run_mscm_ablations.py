"""Subsequent MSCM ablations; requires the completed five-arm core first."""
import argparse
import fcntl
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from src.models.mscm_reconstruction import MSCMReconstructor
from src.runners.run_mscm_development import (UNITS,InputArrays,set_seed,initialize_banks,
    train_stage,infer_moments,evaluate_model)
from src.runners.run_local_residual_models import source_for
from src.runners.run_rcr import atomic_json,state_hash
from src.runners.run_rcr_memory_probe import digest
from src.runners.probe_scale_relation_evidence import read


ABLATIONS = {
    'pure_memory':dict(fusion='pure'),
    'direct_fusion':dict(fusion='direct'),
    'teacher_lambda0':dict(memory_weight=0.),
    'teacher_lambda0p1':dict(memory_weight=.1),
    'patch4':dict(patch=4),
    'patch16':dict(patch=16),
    'finetune_encoder':dict(finetune_encoder=True),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',default='runs/internal/mscm_mcr_development_20261025_v2')
    parser.add_argument('--output',default='runs/internal/mscm_model_one_ablations_20261025_v1')
    parser.add_argument('--units',nargs='+',default=UNITS)
    parser.add_argument('--ablations',nargs='+',choices=list(ABLATIONS),default=list(ABLATIONS))
    parser.add_argument('--epochs',type=int,default=5)
    args = parser.parse_args()
    core,root = Path(args.core),Path(args.output)
    if not (core/'model_one_analysis_complete.json').exists():
        raise RuntimeError('Complete and review model-one core evaluation before subsequent ablations')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; execute outside sandbox')
    settings = json.loads((core/'protocol_manifest.json').read_text())
    if settings.get('teacher_loss_reduction') != 'sum_scales mean_batch_patch sum_features squared_normalized_error':
        raise ValueError('Core teacher-loss protocol does not match the user specification')
    root.mkdir(parents=True,exist_ok=True)
    protocol = dict(task='MSCM-MODEL-ONE-ABLATIONS-20261025-v1',core=str(core),units=args.units,
        ablations={n:ABLATIONS[n] for n in args.ablations},epochs=args.epochs,
        teacher_loss_reduction=settings['teacher_loss_reduction'],normal_only=True,
        core_protocol_sha256=digest(core/'protocol_manifest.json'),test_selection=False,
        implementation_sha256=digest(__file__))
    mp = root/'protocol_manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != protocol: raise ValueError('Protocol changed; use new output')
    atomic_json(mp,protocol)
    torch.set_num_threads(4)
    with open('/tmp/enhancedmtadgat_mscm_gpu0.lock','a+') as lock,threadpool_limits(limits=4):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for unit in args.units:
            source,_ = source_for(unit)
            mapping = json.loads((source/'mapping.json').read_text())
            inputs = InputArrays(core/'inputs'/unit)
            teacher_state = torch.load(core/unit/'pretrain_multi/best.pt',map_location='cpu',weights_only=True)
            for name in args.ablations:
                output = root/unit/name;output.mkdir(parents=True,exist_ok=True)
                if (output/'complete.json').exists(): continue
                options = dict(seed=settings['seed'],batch=settings['batch'],patch=8,
                    gamma=.1,fusion='bounded',memory_weight=.01,patience=3,finetune_encoder=False)
                options.update(ABLATIONS[name])
                from types import SimpleNamespace
                config = SimpleNamespace(**options)
                set_seed(config.seed)
                teacher = MSCMReconstructor('A1',patch=config.patch).cuda().eval()
                teacher.load_state_dict(teacher_state)
                for p in teacher.parameters():p.requires_grad_(False)
                model = MSCMReconstructor('A4',patch=config.patch,gamma=config.gamma,fusion=config.fusion).cuda()
                missing,unexpected = model.load_state_dict(teacher_state,strict=False)
                assert not unexpected and all(k.startswith('memories.') for k in missing)
                atomic_json(output/'ablation.json',dict(name=name,options=options,teacher_hash=state_hash(teacher)))
                if not (output/'training/last.pt').exists() and not (output/'training/complete.json').exists():
                    initialize_banks(model,teacher,inputs,output,config)
                model = train_stage(model,teacher,inputs,output/'training','memory',args.epochs,config,config.memory_weight)
                infer_moments(model,inputs,source,output,mapping,config)
                evaluate_model(unit,name,inputs,source,output,mapping)
                del model,teacher;torch.cuda.empty_cache()
            del inputs,teacher_state;gc.collect();torch.cuda.empty_cache()
    frames = [read(root/u/n/'metrics.csv') for u in args.units for n in args.ablations]
    pd.concat(frames,ignore_index=True).to_csv(root/'all_metrics.csv',index=False)
    atomic_json(root/'complete.json',dict(units=args.units,ablations=args.ablations,test_selection=False))


if __name__ == '__main__':main()
