"""Frozen-cache masked relation training and deterministic LOO scoring."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.rcr import MaskedResidualGraphNet, MaskedResidualMLP, EmpiricalCDF, RCRLite
from src.runners.run_rcr import atomic_json, atomic_checkpoint, state_hash
from src.runners.run_rcr_lite import SCORE_NAMES


class ResidualWindows(Dataset):
    def __init__(self, directory):
        self.z = np.load(directory/'standardized_residual.npy', mmap_mode='r')
        self.q = np.load(directory/'condition_standardized.npy', mmap_mode='r')
        self.record = json.loads((directory/'complete.json').read_text())
        assert len(self.z) == len(self.q) == self.record['windows']

    def __len__(self):
        return len(self.z)

    def __getitem__(self, ix):
        return torch.from_numpy(np.array(self.z[ix])), torch.from_numpy(np.array(self.q[ix]))


def points(z, q):
    b, length, d = z.shape
    return z.reshape(-1, d), q[:, None].expand(-1, length, -1).reshape(b*length, -1)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(architecture, variables, condition_mode, condition_dim):
    if architecture == 'dense_mlp':
        return MaskedResidualMLP(variables, condition_mode=condition_mode,
                                 condition_dim=condition_dim)
    settings = {
        'graph_only': (True, False),
        'transformer_only': (False, True),
        'full': (True, True),
    }
    if architecture not in settings:
        raise ValueError(architecture)
    graph, attention = settings[architecture]
    return MaskedResidualGraphNet(variables, condition_mode=condition_mode,
                                  condition_dim=condition_dim, feature_graph=graph,
                                  variable_attention=attention)


def train(args, output, datasets, device):
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    d, qdim = datasets['train'].z.shape[-1], datasets['train'].q.shape[-1]
    model = build_model(args.architecture, d, args.modulation, qdim).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(datasets['train'], batch_size=128, shuffle=True, generator=generator)
    validation = DataLoader(datasets['validation'], batch_size=128, shuffle=False)
    if (output/'training_complete.json').exists():
        model.load_state_dict(torch.load(output/'best.pt', map_location=device, weights_only=True))
        assert state_hash(model) == json.loads((output/'training_complete.json').read_text())['head_hash']
        return model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    start, best, stale, best_epoch, history = 1, float('inf'), 0, 0, []
    last = output/'last.pt'
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start, best, stale, best_epoch = state['epoch']+1, state['best'], state['stale'], state['best_epoch']
        history = state['history']
        torch.set_rng_state(state['rng']); generator.set_state(state['loader_rng'])
        if device == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        atomic_checkpoint(output/'initial.pt', model.state_dict())
        atomic_json(output/'initial.json', dict(hash=state_hash(model), parameters=sum(p.numel() for p in model.parameters())))
    for epoch in range(start, args.epochs+1):
        if stale >= 8:
            break
        model.train(); total, count = 0., 0; clock = time.monotonic()
        for batch, (z, q) in enumerate(train_loader, 1):
            z, q = points(z.to(device), q.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss = model.masked_loss(z, q)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite masked relation loss')
            loss.backward(); optimizer.step()
            total += float(loss.detach())*len(z); count += len(z)
            if batch % 1000 == 0:
                print(json.dumps(dict(stage='relation_batch', epoch=epoch, batch=batch,
                                      batches=len(train_loader), loss=total/count)), flush=True)
        model.eval(); val_total, val_count = 0., 0
        with torch.inference_mode():
            for z, q in validation:
                z, q = points(z.to(device), q.to(device))
                error = (z-model.leave_one_out(z, q)).square()
                val_total += float(error.sum()); val_count += error.numel()
        val = val_total/val_count
        if not np.isfinite(val):
            raise FloatingPointError('Nonfinite normal validation LOO loss')
        if val < best:
            best, best_epoch, stale = val, epoch, 0
            atomic_checkpoint(output/'best.pt', model.state_dict())
        else:
            stale += 1
        row = dict(epoch=epoch, train_loss=total/count, validation_loo_loss=val,
                   best_epoch=best_epoch, seconds=time.monotonic()-clock)
        history.append(row)
        pd.DataFrame(history).to_csv(output/'history.csv', index=False)
        atomic_checkpoint(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          epoch=epoch, best=best, stale=stale, best_epoch=best_epoch, history=history,
                          rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else [],
                          loader_rng=generator.get_state()))
        print(json.dumps(dict(stage='relation_epoch', **row)), flush=True)
    model.load_state_dict(torch.load(output/'best.pt', map_location=device, weights_only=True))
    atomic_json(output/'training_complete.json', dict(head_hash=state_hash(model), best_epoch=best_epoch,
                validation_loo_loss=best, epochs_executed=len(history), selection='normal_validation_loo'))
    return model.eval()


def infer(model, datasets, output, lite_root, device, save_errors):
    head_hash = state_hash(model)
    for partition, dataset in datasets.items():
        directory = output/partition
        directory.mkdir(exist_ok=True)
        done = directory/'loo_complete.json'
        if done.exists():
            assert json.loads(done.read_text())['head_hash'] == head_hash
            continue
        error = (np.lib.format.open_memmap(directory/'loo_squared_error.npy', mode='w+',
                                           dtype='float32', shape=dataset.z.shape)
                 if save_errors else None)
        score = np.lib.format.open_memmap(directory/'scores.npy', mode='w+', dtype='float64', shape=(len(dataset), 6))
        source = np.load(lite_root/partition/'scores.npy', mmap_mode='r')
        score[:, :2] = source[:, :2]  # Same magnitude/raw branch for Lite and Net.
        offset = 0
        with torch.inference_mode():
            for step, (z, q) in enumerate(DataLoader(dataset, batch_size=128, shuffle=False), 1):
                shape = z.shape
                z, q = points(z.to(device), q.to(device))
                errors = (z-model.leave_one_out(z, q)).square().reshape(shape).cpu().numpy()
                stop = offset+len(errors)
                if not np.isfinite(errors).all():
                    raise FloatingPointError('Nonfinite LOO prediction error')
                if error is not None:
                    error[offset:stop] = errors
                score[offset:stop, 2] = errors.mean((1, 2), dtype=np.float64)
                offset = stop
                if step % 1000 == 0:
                    print('RELATION INFERENCE', partition, offset, '/', len(dataset), flush=True)
        assert offset == len(dataset)
        if error is not None:
            error.flush()
        score.flush()
        atomic_json(done, dict(head_hash=head_hash, windows=len(dataset),
                              loo_squared_error_saved=bool(save_errors)))
    lite = RCRLite.load(lite_root)
    train_scores = np.load(output/'train/scores.npy', mmap_mode='r')
    relation_cdf = EmpiricalCDF().fit(train_scores[:, 2])
    np.savez(output/'score_calibration.npz', magnitude_reference=lite.mag_cdf.reference,
             relation_reference=relation_cdf.reference, alpha=.5)
    for partition, dataset in datasets.items():
        directory = output/partition
        score = np.load(directory/'scores.npy', mmap_mode='r+')
        score[:, 3] = lite.mag_cdf.transform(score[:, 1])
        score[:, 4] = relation_cdf.transform(score[:, 2])
        score[:, 5] = .5*(score[:, 3]+score[:, 4])
        score.flush()
        atomic_json(directory/'complete.json', dict(head_hash=head_hash, windows=len(dataset),
                    score_names=SCORE_NAMES, standardized_residual_source=str(lite_root/partition/'standardized_residual.npy'),
                    magnitude_saturation_fraction=float(np.mean(score[:, 3] == 1)),
                    relation_saturation_fraction=float(np.mean(score[:, 4] == 1))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--condition', choices=['continuous', 'none', 'cycle'], default='continuous')
    parser.add_argument('--modulation', choices=['mlp', 'linear', 'none'], default='mlp')
    parser.add_argument('--architecture', choices=['dense_mlp', 'graph_only',
                        'transformer_only', 'full'], default='full')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output-unit', help='Independent single-directory unit name, e.g. head_B1_seed3408')
    parser.add_argument('--no-save-errors', action='store_true', help='Save scores/mapping without the large per-coordinate LOO error array')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='CPU only permitted for smoke cache')
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(args.root)
    base_protocol = json.loads((root/'protocol.json').read_text())
    if args.device == 'cpu' and not base_protocol['smoke']:
        raise RuntimeError('Full-data training requires CUDA')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required; run outside sandbox')
    lite_root = root/('lite_'+args.condition)
    assert (lite_root/'complete.json').exists(), 'Complete Lite before relation training'
    output = (root/f'net_{args.condition}_{args.modulation}' if args.architecture == 'full'
              else root/f'net_ablation_{args.architecture}_{args.condition}_{args.modulation}')
    if args.output_unit:
        if Path(args.output_unit).name != args.output_unit or args.output_unit in {'.', '..'}:
            raise ValueError('--output-unit must be one directory name')
        output = root/args.output_unit
    output.mkdir(exist_ok=True)
    signatures = {str(p): file_hash(p) for p in [root/'backbone_best.pt', lite_root/'calibration.npz', lite_root/'lite.npz']}
    protocol = dict(condition=args.condition, modulation=args.modulation, epochs=args.epochs,
                    seed=args.seed, frozen_sources=signatures, mask_probability=.3, alpha=.5,
                    lr=.001, weight_decay=.0001, batch_windows=128, patience=8,
                    device=args.device, smoke=base_protocol['smoke'])
    if args.architecture != 'full':
        protocol['architecture'] = args.architecture
        protocol['loo_squared_error_saved'] = False
    if args.no_save_errors:
        protocol['loo_squared_error_saved'] = False
    if (output/'protocol.json').exists():
        assert json.loads((output/'protocol.json').read_text()) == protocol, 'Changed relation protocol/source'
    else:
        atomic_json(output/'protocol.json', protocol)
    datasets = {p: ResidualWindows(lite_root/p) for p in ['train', 'validation', 'test']}
    model = train(args, output, datasets, args.device)
    infer(model, datasets, output, lite_root, args.device,
          save_errors=args.architecture == 'full' and not args.no_save_errors)
    assert all(file_hash(p) == h for p, h in signatures.items()), 'Frozen source changed'
    atomic_json(output/'complete.json', dict(head_hash=state_hash(model), frozen_sources_unchanged=True,
                condition=args.condition, modulation=args.modulation,
                architecture=args.architecture, no_fault_training=True))
    print('RCR RELATION COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
