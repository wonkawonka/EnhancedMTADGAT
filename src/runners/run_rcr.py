"""Independent staged RCR experiments. Never selects using fault/test labels."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.ch_battery_utils import load_ch_battery_mixed_cycle_research_split
from src.models.rcr import RCRBackbone, condition_features
from src.runners.run_unified_external_baselines import WindowDataset


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def state_hash(model):
    result = hashlib.sha256()
    for key, value in model.state_dict().items():
        result.update(key.encode())
        result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def load_experiment(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    settings = dict(chemistry=args.chemistry, seed=args.seed, window=32, stride=4,
                    feature_graph=not args.no_feature_graph, temporal=not args.no_temporal,
                    epochs=args.epochs_backbone, patience=8, lr=.001, weight_decay=.0001,
                    batch_size=128, train_limit=args.train_limit, eval_limit=args.eval_limit,
                    loss='huber_delta_1', epoch_selection='normal_validation_only',
                    smoke=bool(args.train_limit or args.eval_limit), implementation='rcr_v1')
    manifest = root/'protocol.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != settings:
            raise RuntimeError('Output protocol differs; use a separate output directory')
    else:
        atomic_json(manifest, settings)
    split = load_ch_battery_mixed_cycle_research_split(chemistry=args.chemistry, seed=args.seed)
    columns = split['feature_columns']
    control = (columns.index('SUM_CURRENT'), columns.index('SOC'))
    data, ids, metadata = {}, {}, []
    for partition in ['train', 'validation', 'test']:
        ids[partition] = sorted(k for k, v in split[partition].items() if len(v) >= 32)
        rows = [dict(split[partition+'_metadata'][k], partition=partition) for k in ids[partition]]
        if partition != 'test':
            assert all(int(r['sample_label']) == 0 for r in rows)
        metadata.extend(rows)
        data[partition] = WindowDataset([split[partition][k] for k in ids[partition]], 32, 4,
                                       limit=args.train_limit if partition == 'train' else args.eval_limit,
                                       seed=args.seed)
    assert not (set(split['train_vins']) & set(split['validation_vins']))
    assert not ((set(split['train_vins']) | set(split['validation_vins'])) & set(split['test_normal_vins']))
    mapping = dict(feature_columns=columns, control_indices=control, sample_ids=ids,
                   train_vins=split['train_vins'], validation_vins=split['validation_vins'],
                   test_normal_vins=split['test_normal_vins'],
                   windows={p: len(d) for p, d in data.items()})
    # JSON normalizes tuples to lists; compare serialized values for safe resume.
    mapping = json.loads(json.dumps(mapping))
    if (root/'mapping.json').exists():
        assert json.loads((root/'mapping.json').read_text()) == mapping, 'Data split changed'
    else:
        atomic_json(root/'mapping.json', mapping)
        pd.DataFrame(metadata).to_csv(root/'metadata.csv', index=False)
    print(json.dumps(dict(stage='data_ready', windows=mapping['windows'],
                          samples={p: len(v) for p, v in ids.items()})), flush=True)
    return root, data, ids, control, len(columns)


def backbone_train(args, root, data, variables):
    best_path, last_path = root/'backbone_best.pt', root/'backbone_last.pt'
    model = RCRBackbone(variables, feature_graph=not args.no_feature_graph,
                        temporal=not args.no_temporal).cuda()
    if (root/'backbone_complete.json').exists():
        model.load_state_dict(torch.load(best_path, map_location='cuda', weights_only=True))
        assert state_hash(model) == json.loads((root/'backbone_complete.json').read_text())['checkpoint_hash']
        print('Verified completed backbone; no retraining', flush=True)
        return model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    generator = torch.Generator().manual_seed(args.seed)
    train = DataLoader(data['train'], batch_size=128, shuffle=True, generator=generator, pin_memory=True)
    validation = DataLoader(data['validation'], batch_size=128, shuffle=False, pin_memory=True)
    start, best, stale, best_epoch, history = 1, float('inf'), 0, 0, []
    if last_path.exists():
        # Trusted local checkpoint produced by this exact run; contains RNG state.
        saved = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start, best, stale, best_epoch = saved['epoch']+1, saved['best'], saved['stale'], saved['best_epoch']
        history = saved['history']
        torch.set_rng_state(saved['rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        generator.set_state(saved['loader_rng'])
    else:
        atomic_json(root/'backbone_initial.json', dict(hash=state_hash(model),
                    parameters=sum(p.numel() for p in model.parameters())))
    for epoch in range(start, args.epochs_backbone+1):
        if stale >= 8:
            break
        clock = time.monotonic()
        model.train(); total, count = 0., 0
        for step, (x, _, _) in enumerate(train, 1):
            x = x.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.huber_loss(model(x), x, delta=1.)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite backbone training loss')
            loss.backward(); optimizer.step()
            total += float(loss.detach())*len(x); count += len(x)
            if step % 1000 == 0:
                print(json.dumps(dict(stage='backbone_train', epoch=epoch, batch=step,
                                      batches=len(train), loss=total/count)), flush=True)
        model.eval(); val_total, val_count = 0., 0
        with torch.inference_mode():
            for x, _, _ in validation:
                x = x.cuda(non_blocking=True)
                value = torch.nn.functional.huber_loss(model(x), x, delta=1.)
                val_total += float(value)*len(x); val_count += len(x)
        val = val_total/val_count
        if not np.isfinite(val):
            raise FloatingPointError('Nonfinite backbone normal validation loss')
        if val < best:
            best, stale, best_epoch = val, 0, epoch
            atomic_checkpoint(best_path, model.state_dict())
        else:
            stale += 1
        row = dict(epoch=epoch, train_loss=total/count, validation_loss=val,
                   best_epoch=best_epoch, seconds=time.monotonic()-clock)
        history.append(row)
        pd.DataFrame(history).to_csv(root/'backbone_history.csv', index=False)
        atomic_checkpoint(last_path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          epoch=epoch, best=best, stale=stale, best_epoch=best_epoch, history=history,
                          rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                          loader_rng=generator.get_state()))
        print(json.dumps(dict(stage='backbone_epoch', **row)), flush=True)
    model.load_state_dict(torch.load(best_path, map_location='cuda', weights_only=True))
    atomic_json(root/'backbone_complete.json', dict(best_epoch=best_epoch, validation_loss=best,
                epochs_executed=len(history), checkpoint_hash=state_hash(model),
                selection='normal_validation_huber', no_test_selection=True))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def cache_residuals(model, data, ids, control, root):
    """Disk-backed full windows preserve signed per-variable residual and mapping."""
    expected_hash = state_hash(model)
    for partition, dataset in data.items():
        directory = root/'residuals'/partition
        directory.mkdir(parents=True, exist_ok=True)
        done = directory/'complete.json'
        if done.exists():
            record = json.loads(done.read_text())
            assert record['backbone_hash'] == expected_hash and record['windows'] == len(dataset)
            continue
        n, d = len(dataset), model.variables
        residual = np.lib.format.open_memmap(directory/'residual.npy', mode='w+', dtype='float32', shape=(n, dataset.window, d))
        q = np.lib.format.open_memmap(directory/'condition.npy', mode='w+', dtype='float32', shape=(n, 8))
        references = np.lib.format.open_memmap(directory/'references.npy', mode='w+', dtype='int64', shape=(n, 2))
        offset = 0
        with torch.inference_mode():
            for x, seq, end in DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True):
                x = x.cuda(non_blocking=True)
                stop = offset+len(x)
                residual[offset:stop] = (x-model(x)).cpu().numpy()
                q[offset:stop] = condition_features(x, *control).cpu().numpy()
                references[offset:stop] = np.column_stack((seq.numpy(), end.numpy()))
                offset = stop
        assert offset == n
        residual.flush(); q.flush(); references.flush()
        atomic_json(done, dict(backbone_hash=expected_hash, windows=n, sample_ids=ids[partition],
                              reference_columns=['sequence_id', 'window_end_index']))
        print(json.dumps(dict(stage='residual_cache_complete', partition=partition, windows=n)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chemistry', choices=['LFP', 'NCM'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--epochs-backbone', type=int, default=50)
    parser.add_argument('--train-limit', type=int, default=0, help='Smoke only; official run is full training data')
    parser.add_argument('--eval-limit', type=int, default=0, help='Smoke only; official run is full validation/test')
    parser.add_argument('--no-feature-graph', action='store_true')
    parser.add_argument('--no-temporal', action='store_true')
    parser.add_argument('--stage', choices=['backbone', 'residuals'], default='backbone')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; run outside sandbox')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    root, data, ids, control, variables = load_experiment(args)
    model = backbone_train(args, root, data, variables)
    if args.stage == 'residuals':
        cache_residuals(model, data, ids, control, root)
    print('RCR REQUESTED STAGE COMPLETE', args.stage, flush=True)


if __name__ == '__main__':
    main()
