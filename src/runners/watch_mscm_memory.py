"""CPU observer of training checkpoints; never changes the training process."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from src.models.mscm_reconstruction import MSCMReconstructor
from src.runners.run_rcr import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='runs/internal/mscm_mcr_development_20261025_v2')
    parser.add_argument('--poll',type=float,default=1.)
    args = parser.parse_args()
    root = Path(args.root)
    settings = json.loads((root/'protocol_manifest.json').read_text())
    torch.set_num_threads(2)
    seen,arrays = {},{}
    for unit in settings['units']:
        for variant in ['A2','A4']:
            path = root/unit/variant/'training/observed_epoch_memory.json'
            if path.exists():
                records = json.loads(path.read_text())
                seen[unit,variant] = max((r['epoch'] for r in records),default=0)
    while True:
        for unit in settings['units']:
            input_path = root/'inputs'/unit/'train_x.npy'
            if not input_path.exists(): continue
            for variant in ['A2','A4']:
                training = root/unit/variant/'training'
                checkpoint = training/'last.pt'
                if not checkpoint.exists(): continue
                progress = training/'progress.json'
                if progress.exists() and json.loads(progress.read_text())['epoch'] <= seen.get((unit,variant),0): continue
                state = torch.load(checkpoint,map_location='cpu',weights_only=False)
                epoch = state['epoch']
                if epoch <= seen.get((unit,variant),0): continue
                if unit not in arrays:
                    values = np.load(input_path,mmap_mode='r')
                    selected = np.random.default_rng(settings['seed']).choice(len(values),min(512,len(values)),replace=False)
                    np.save(root/'inputs'/unit/'epoch_memory_probe_normal_indices.npy',selected)
                    arrays[unit] = torch.from_numpy(np.array(values[selected]))
                model = MSCMReconstructor(variant,patch=settings['patch'],gamma=settings['gamma'],fusion=settings['fusion']).eval()
                model.load_state_dict(state['model'])
                diagnostics = []
                with torch.inference_mode():
                    queries = model.patches(arrays[unit])
                    for s,(query,memory) in enumerate(zip(queries,model.memories)):
                        _,_,weights = memory(query)
                        soft = weights.mean((0,1)).numpy()
                        hard = torch.bincount(weights.argmax(-1).flatten(),minlength=8).numpy()
                        diagnostics.append(dict(scale=s,soft_occupancy=soft.tolist(),
                            hard_occupancy=(hard/hard.sum()).tolist(),
                            effective_prototypes=float(np.exp(-(soft*np.log(np.maximum(soft,1e-30))).sum()))))
                path = training/'observed_epoch_memory.json'
                records = json.loads(path.read_text()) if path.exists() else []
                records.append(dict(epoch=epoch,normal_training_windows=len(arrays[unit]),
                                    source='CPU observation of saved epoch checkpoint; fixed normal training sample',scales=diagnostics))
                atomic_json(path,records)
                seen[unit,variant] = epoch
                print('MEMORY OBSERVED',unit,variant,epoch,flush=True)
        if (root/'model_one_core_complete.json').exists(): break
        time.sleep(args.poll)
    atomic_json(root/'epoch_memory_observer_complete.json',dict(observed_epochs={u+'/'+v:e for (u,v),e in seen.items()},
        complete_epoch_coverage_not_guaranteed=True))


if __name__ == '__main__': main()
