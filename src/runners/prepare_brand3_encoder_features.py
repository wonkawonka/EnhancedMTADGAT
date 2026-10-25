"""Extract hidden H from original X using the completed frozen Brand3 models."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.external_protocol import load_external_protocol_data
from src.models.rcr import RCRBackbone, condition_features
from src.models.rcr_feature_memory import frozen_encoder_output
from src.runners.run_rcr import atomic_json, state_hash
from src.runners.run_rcr_memory_probe import digest
from src.runners.run_brand3_memory_probe import verify_source
from src.runners.run_unified_external_baselines import WindowDataset


def extract_fold(base, output, fold):
    source = base/'sources'/f'fold{fold}'/'full'
    protocol, mapping, meta, signatures = verify_source(source)
    backbone = json.loads((source/'backbone_complete.json').read_text())
    directory = output/f'fold{fold}'
    directory.mkdir(parents=True, exist_ok=True)
    settings = dict(dataset='Brand3', fold=fold, source=str(source.resolve()),
        backbone_hash=backbone['checkpoint_hash'], source_mapping_sha256=digest(source/'mapping.json'),
        layer='temporal_encoder_output_before_decoder', feature_pooling='mean_over_127_time_tokens',
        input='original training-fold MinMax scaled seven-channel X; first native 127 points',
        uses_residuals=False, uses_calibration=False, model_eval=True, requires_grad=False,
        batch_size=128, implementation_sha256={Path(p).name: digest(p) for p in [__file__,
            Path(__file__).parents[1]/'models/rcr_feature_memory.py']})
    manifest = directory/'protocol_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError('Feature-extraction protocol changed; use new output')
    atomic_json(manifest, settings)
    if (directory/'complete.json').exists():
        print('ENCODER H ALREADY COMPLETE', fold, flush=True)
        return
    data = load_external_protocol_data('Brand3', brand_fold=fold, brand_fold_seed=0, seed=3407,
        brand_split_protocol='strict_normal_validation', brand_normalization='minmax', brand_progress=True)
    if data.metadata['sample_ids'] != mapping['sample_ids'] or data.metadata['normalization'] != protocol['normalization']:
        raise ValueError('Original X order or training-only scaler changed')
    model = RCRBackbone(7).cuda().eval()
    model.load_state_dict(torch.load(source/'backbone_best.pt', map_location='cpu', weights_only=True))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if state_hash(model) != backbone['checkpoint_hash']:
        raise ValueError('Frozen Backbone changed')
    sequences = dict(train=data.train_sequences, validation=data.validation_sequences, test=data.test_sequences)
    audit = {}
    for partition, values in sequences.items():
        dataset = WindowDataset(values, 127, 127, seed=3407)
        references = np.load(source/'residuals'/partition/'references.npy', mmap_mode='r')
        residual = np.load(source/'residuals'/partition/'residual.npy', mmap_mode='r')
        condition = np.load(source/'residuals'/partition/'condition.npy', mmap_mode='r')
        h = np.lib.format.open_memmap(directory/f'{partition}_features.npy', mode='w+', dtype='float32',
                                      shape=(len(dataset), model.fusion.out_features))
        offset, max_residual_difference, max_condition_difference = 0, 0., 0.
        for x, seq, end in DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True):
            stop = offset+len(x)
            np.testing.assert_array_equal(references[offset:stop], np.column_stack((seq.numpy(), end.numpy())))
            x = x.cuda(non_blocking=True)
            hidden, reconstruction = frozen_encoder_output(model, x)
            replayed_residual = (x-reconstruction).cpu().numpy()
            replayed_condition = condition_features(x, 1, 2).cpu().numpy()
            np.testing.assert_allclose(replayed_residual, residual[offset:stop], rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(replayed_condition, condition[offset:stop], rtol=1e-5, atol=1e-7)
            max_residual_difference = max(max_residual_difference, float(np.abs(replayed_residual-residual[offset:stop]).max()))
            max_condition_difference = max(max_condition_difference, float(np.abs(replayed_condition-condition[offset:stop]).max()))
            h[offset:stop] = hidden.mean(1).cpu().numpy()
            if not np.isfinite(h[offset:stop]).all():
                raise FloatingPointError('Nonfinite Encoder features')
            offset = stop
        h.flush()
        if offset != len(mapping['sample_ids'][partition]):
            raise ValueError('Missing hidden features')
        audit[partition] = dict(windows=offset, hidden_dim=h.shape[1],
            max_residual_replay_difference=max_residual_difference,
            max_condition_replay_difference=max_condition_difference)
        print('ENCODER H PARTITION COMPLETE', fold, partition, offset, flush=True)
    if state_hash(model) != backbone['checkpoint_hash']:
        raise ValueError('Feature extraction modified the Backbone')
    atomic_json(directory/'complete.json', dict(backbone_hash=backbone['checkpoint_hash'],
        partitions=audit, backbone_unchanged=True, original_RCR_residuals_reproduced=True,
        memory_features_use_residuals=False))
    del model, data, sequences
    gc.collect(); torch.cuda.empty_cache()
    print('ENCODER H FOLD COMPLETE', fold, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='runs/internal/rcr_memory_brand3_20261025_v1')
    parser.add_argument('--output', default='runs/internal/rcr_encoder_memory_brand3_20261025_v1/features')
    args = parser.parse_args()
    base, output = Path(args.base).resolve(), Path(args.output).resolve()
    if not (base/'complete.json').exists():
        raise ValueError('Finish the requested Brand3 calibration experiment before hidden Memory')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA feature extraction must execute outside sandbox')
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        extract_fold(base, output, fold)
    atomic_json(output/'complete.json', dict(folds=list(range(5)), frozen_encoder=True, uses_residuals=False))


if __name__ == '__main__':
    main()
