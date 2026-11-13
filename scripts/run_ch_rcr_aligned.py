"""Full-window CH baselines with preregistered RCR-aligned training budgets.

Reuses existing model implementations without changing legacy or Brand1 runners.
Run with python -m scripts.run_ch_rcr_aligned; --preflight does not use CUDA.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.evaluate_rcr import metrics
from src.runners import run_unified_external_baselines as baseline
from src.runners.run_rcr import atomic_checkpoint, atomic_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = 'configs/external/ch_rcr_aligned_20261111.json'
PARTS = ('train', 'validation', 'test')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(plan):
    reference = json.loads(Path(plan['rcr_plan']).read_text())
    t = plan['training']
    for key, rcr_key in [('epochs', 'epochs'), ('patience', 'patience'),
                         ('batch_size', 'batch_size'), ('learning_rate', 'lr'),
                         ('weight_decay', 'weight_decay'), ('optimizer', 'optimizer')]:
        assert t[key] == reference['common_args'][rcr_key], key
    assert plan['seed'] == reference['data']['split_seed'] == 3407
    records = {}
    for unit in plan['datasets']:
        source = Path(plan['source_root']) / unit / 'full'
        cache = Path(plan['input_cache']) / unit
        mapping = json.loads((source / 'mapping.json').read_text())
        assert json.loads((cache / 'mapping.json').read_text()) == mapping
        assert json.loads((cache / 'complete.json').read_text())['source_mapping_sha256'] == digest(source / 'mapping.json')
        metadata = pd.read_csv(source / 'metadata.csv').set_index('sample_id')
        for part in PARTS:
            x = np.load(cache / f'{part}_x.npy', mmap_mode='r')
            refs = np.load(source / 'residuals' / part / 'references.npy', mmap_mode='r')
            assert x.shape == (mapping['windows'][part], t['window'], 7)
            assert refs.shape == (len(x), 2)
            assert np.all(np.diff(refs[:, 0]) >= 0)
            assert np.array_equal(np.unique(refs[:, 0]), np.arange(len(mapping['sample_ids'][part])))
            if part != 'test':
                assert metadata.loc[mapping['sample_ids'][part], 'sample_label'].eq(0).all()
        records[unit] = dict(windows=mapping['windows'], mapping_sha256=digest(source / 'mapping.json'),
                             source=str(source), cache=str(cache))
    assert len(plan['datasets']) * len(plan['methods']) == plan['expected_jobs'] == 10
    return records


class Arrays:
    def __init__(self, root):
        self.arrays = {p: np.load(root / f'{p}_x.npy', mmap_mode='r') for p in PARTS}
        self.resident = {}
        free, _ = torch.cuda.mem_get_info()
        needed = sum(self.arrays[p].nbytes for p in ('train', 'validation'))
        if needed < free * .4:
            for p in ('train', 'validation'):
                self.resident[p] = torch.from_numpy(np.array(self.arrays[p])).cuda()
        print('INPUT_READY', {p: list(x.shape) for p, x in self.arrays.items()},
              'gpu_cached', list(self.resident), flush=True)

    def get(self, part, index):
        if part in self.resident:
            if isinstance(index, np.ndarray):
                index = torch.from_numpy(index).cuda()
            return self.resident[part][index]
        return torch.from_numpy(np.array(self.arrays[part][index])).cuda()


def losses(method, model, x, association_weight=3.0):
    """Return native training objective and preregistered validation objective."""
    if method == 'mtad_gat':
        forecast, reconstruction = model(x)
        loss = .5 * F.mse_loss(forecast, x[:, -1]) + .5 * F.mse_loss(reconstruction, x)
        return loss, loss
    if method == 'tranad':
        first, second = model(x)
        loss = .5 * F.mse_loss(first, x[:, -1:]) + .5 * F.mse_loss(second, x[:, -1:])
        return loss, loss
    if method == 'omnianomaly':
        rec, mu, logvar = model(x)
        loss = F.mse_loss(rec, x) + .01 * (-.5 * (1 + logvar - mu.square() - logvar.exp()).mean())
        return loss, loss
    if method == 'anomaly_transformer':
        rec, series, prior, _ = model(x)
        series_loss, prior_loss = baseline._association_losses(series, prior, x.shape[1])
        rec_loss = F.mse_loss(rec, x)
        return rec_loss - association_weight * series_loss + association_weight * prior_loss, rec_loss
    if method == 'dcdetector':
        series, prior = model(x)
        series_loss, prior_loss = baseline._association_losses(series, prior, x.shape[1])
        return prior_loss - series_loss, series_loss
    raise ValueError(method)


def make_model(method, window, output):
    saved = sys.argv
    try:
        sys.argv = ['baseline', '--method', method, '--dataset', 'CH_LFP_MIXED_CYCLE',
                    '--output_dir', str(output), '--epochs', '0', '--lookback', str(window)]
        args = baseline.parse_args()
    finally:
        sys.argv = saved
    data = SimpleNamespace(train_sequences=[np.empty((window, 7), dtype='float32')], metadata={})
    model, auxiliary = baseline._fit_torch(data, args, [], window, torch.device('cuda'))
    assert auxiliary is None
    return model


def fit(plan, method, arrays, output):
    t = plan['training']; seed = plan['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = make_model(method, t['window'], output)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['learning_rate'], weight_decay=t['weight_decay'])
    best, stale, start, best_epoch, history = float('inf'), 0, 1, 0, []
    training = output / 'training'; training.mkdir(exist_ok=True)
    if (training / 'complete.json').exists():
        model.load_state_dict(torch.load(training / 'best.pt', weights_only=True))
        return model.eval()
    if (training / 'last.pt').exists():
        s = torch.load(training / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(s['model']); optimizer.load_state_dict(s['optimizer'])
        best, stale, start, best_epoch, history = s['best'], s['stale'], s['epoch'] + 1, s['best_epoch'], s['history']
        torch.set_rng_state(s['rng']); torch.cuda.set_rng_state_all(s['cuda_rng'])
        np.random.set_state(s['numpy_rng']); random.setstate(s['python_rng'])
    batch = t['batch_size']
    for epoch in range(start, t['epochs'] + 1):
        if stale >= t['patience']:
            break
        clock = last_log = time.monotonic(); model.train(); total = 0.; seen = 0
        n = len(arrays.arrays['train'])
        order = np.random.default_rng(seed + epoch).permutation(n)
        for lo in range(0, n, batch):
            x = arrays.get('train', order[lo:lo + batch])
            optimizer.zero_grad(set_to_none=True)
            loss, _ = losses(method, model, x)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss: {method}')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), t['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(x); seen += len(x)
            now = time.monotonic()
            if lo == 0 or seen == n or now - last_log >= 30:
                progress = dict(phase='training', epoch=epoch, windows=seen, total=n,
                                train_loss=total / seen, grad_norm=float(grad), seconds=now - clock,
                                cuda_peak_MiB=torch.cuda.max_memory_allocated() / 2**20)
                atomic_json(output / 'progress.json', progress)
                print('BATCH', method, json.dumps(progress), flush=True); last_log = now
        model.eval(); val_total = 0.; val_seen = 0
        with torch.inference_mode():
            for lo in range(0, len(arrays.arrays['validation']), batch):
                x = arrays.get('validation', slice(lo, lo + batch))
                _, val = losses(method, model, x)
                val_total += float(val) * len(x); val_seen += len(x)
                now = time.monotonic()
                if lo == 0 or now - last_log >= 30:
                    progress = dict(phase='validation', epoch=epoch, windows=val_seen,
                                    total=len(arrays.arrays['validation']), validation_loss=val_total / val_seen)
                    atomic_json(output / 'progress.json', progress)
                    print('VALIDATION', method, json.dumps(progress), flush=True); last_log = now
        current = val_total / val_seen
        if not np.isfinite(current):
            raise FloatingPointError('Nonfinite normal-validation objective')
        if current < best:
            best, best_epoch, stale = current, epoch, 0
            atomic_checkpoint(training / 'best.pt', model.state_dict())
        else:
            stale += 1
        history.append(dict(epoch=epoch, train_loss=total / seen, validation_loss=current,
                            best_epoch=best_epoch, stale=stale, seconds=time.monotonic() - clock))
        pd.DataFrame(history).to_csv(training / 'history.csv', index=False)
        atomic_checkpoint(training / 'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            best=best, stale=stale, epoch=epoch, best_epoch=best_epoch, history=history,
            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate()))
        print('EPOCH', method, json.dumps(history[-1]), flush=True)
    atomic_json(training / 'complete.json', dict(best_epoch=best_epoch, best_validation_loss=best,
        epochs_executed=history[-1]['epoch'], parameters=sum(p.numel() for p in model.parameters()),
        selection=plan['training']['validation_objectives'][method]))
    model.load_state_dict(torch.load(training / 'best.pt', weights_only=True))
    return model.eval()


def evaluate(plan, unit, method, model, arrays, output):
    source = Path(plan['source_root']) / unit / 'full'
    mapping = json.loads((source / 'mapping.json').read_text())
    metadata = pd.read_csv(source / 'metadata.csv').set_index('sample_id')
    frames = {}
    for part in ('validation', 'test'):
        n = len(arrays.arrays[part]); batch = plan['training']['batch_size']
        score = np.lib.format.open_memmap(output / f'{part}_window_scores.npy', mode='w+', dtype='float32', shape=(n,))
        last_log = time.monotonic()
        with torch.inference_mode():
            for lo in range(0, n, batch):
                x = arrays.get(part, slice(lo, lo + batch))
                values = baseline._batch_scores(method, model, None, x, plan['training']['window'])
                if not np.isfinite(values).all():
                    raise FloatingPointError(f'Nonfinite {part} anomaly scores')
                score[lo:lo + len(x)] = values
                now = time.monotonic()
                if lo == 0 or lo + len(x) == n or now - last_log >= 30:
                    progress = dict(phase=f'score_{part}', windows=lo + len(x), total=n)
                    atomic_json(output / 'progress.json', progress)
                    print('SCORE', method, json.dumps(progress), flush=True); last_log = now
        score.flush()
        refs = np.load(source / 'residuals' / part / 'references.npy', mmap_mode='r')
        edges = np.r_[0, np.flatnonzero(np.diff(refs[:, 0])) + 1, n]
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sid = mapping['sample_ids'][part][int(refs[lo, 0])]
            k = max(1, int(np.ceil(.05 * (hi - lo))))
            rows.append(dict(sample_id=sid, score=float(np.partition(score[lo:hi], -k)[-k:].mean()),
                             label=int(metadata.loc[sid, 'sample_label']), windows=int(hi - lo)))
        frames[part] = pd.DataFrame(rows)
        frames[part].to_csv(output / f'{part}_samples.csv', index=False)
    threshold = float(frames['validation'].score.quantile(.95))
    result = dict(dataset=unit, method=method, seed=plan['seed'], threshold=threshold,
                  **metrics(frames['test'].label, frames['test'].score, threshold))
    atomic_json(output / 'metrics.json', result)
    pd.DataFrame([result]).to_csv(output / 'metrics.csv', index=False)
    atomic_json(output / 'complete.json', dict(result=result,
        training=json.loads((output / 'training/complete.json').read_text()),
        mapping_sha256=digest(source / 'mapping.json'), test_based_selection=False))
    print('COMPLETE', json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default=DEFAULT_PLAN)
    parser.add_argument('--preflight', action='store_true')
    cli = parser.parse_args(); os.chdir(ROOT)
    plan = json.loads(Path(cli.plan).read_text()); records = preflight(plan)
    print('PREFLIGHT_OK', json.dumps(records), flush=True)
    if cli.preflight:
        return
    assert torch.cuda.is_available(), 'CUDA execution must run outside sandbox'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(plan['output']); root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'queue.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = [cli.plan, __file__, 'src/runners/run_unified_external_baselines.py',
               'src/models/unified_external_baselines.py', 'src/models/mtad_gat.py']
    sources += [str(p) for directory in ('external_baselines/Anomaly-Transformer/model',
                                       'external_baselines/DCdetector/model')
                for p in Path(directory).rglob('*.py')]
    fingerprints = {str(Path(p).resolve().relative_to(ROOT)): digest(p) for p in sources}
    manifest = dict(plan=plan, sources=fingerprints, data=records, gpu=torch.cuda.get_device_name(0))
    marker = root / 'protocol_manifest.json'
    if marker.exists():
        assert json.loads(marker.read_text()) == manifest, 'Plan/source changed; use a new output root'
    else:
        atomic_json(marker, manifest)
        for p in fingerprints:
            target = root / 'implementation' / p; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    completed = []; active = None
    try:
        for unit in plan['datasets']:
            arrays = Arrays(Path(plan['input_cache']) / unit)
            for method in plan['methods']:
                active = f'{unit}/{method}/seed{plan["seed"]}'
                output = root / 'jobs' / active; output.mkdir(parents=True, exist_ok=True)
                atomic_json(root / 'status.json', dict(status='running', active_job=active,
                    completed_jobs=len(completed), expected_jobs=10))
                if not (output / 'complete.json').exists():
                    atomic_json(output / 'config.json', dict(plan=plan, dataset=unit, method=method))
                    torch.cuda.reset_peak_memory_stats()
                    print('JOB_START', active, flush=True)
                    model = fit(plan, method, arrays, output)
                    evaluate(plan, unit, method, model, arrays, output)
                    del model; gc.collect(); torch.cuda.empty_cache()
                completed.append(json.loads((output / 'metrics.json').read_text()))
                pd.DataFrame(completed).to_csv(root / 'results.csv', index=False)
            del arrays; gc.collect(); torch.cuda.empty_cache()
        atomic_json(root / 'complete.json', dict(completed_jobs=len(completed)))
        atomic_json(root / 'status.json', dict(status='complete', completed_jobs=len(completed), expected_jobs=10))
    except BaseException:
        atomic_json(root / 'status.json', dict(status='failed', active_job=active,
            completed_jobs=len(completed), expected_jobs=10, error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
