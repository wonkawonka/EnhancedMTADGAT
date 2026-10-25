"""Shared, normal-only training and scoring for the complete MSCM/MCR audit."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_curve

from scripts.evaluate_rcr import metrics
from src.runners.run_rcr import atomic_json, atomic_checkpoint, state_hash
from src.runners.run_rcr_memory_probe import digest
from src.runners.run_rcr_calibration_sensitivity import aggregate
from src.runners.probe_scale_relation_evidence import read, rank_frame


PARTITIONS = ('train', 'validation', 'test')
UNITS = [f'Brand3_fold{i}' for i in range(5)] + ['NCM', 'LFP']


class IndexView:
    def __init__(self, array, indices):
        self.array, self.indices = array, np.asarray(indices)
        self.shape = (len(self.indices),) + array.shape[1:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.array[self.indices[index]]


class TrainingSubset:
    def __init__(self, original, indices):
        self.original, self.indices = original, np.asarray(indices)
        self.arrays = dict(original.arrays)
        self.arrays['train'] = IndexView(original.arrays['train'], self.indices)

    def get(self, partition, index):
        return self.original.get(partition, self.indices[index] if partition == 'train' else index)


def normal_split(source, output, seed=3407, calibration_fraction=.2):
    """Split original normal training VINs before ANY backbone optimization."""
    output.mkdir(parents=True, exist_ok=True)
    marker = output / 'normal_split.json'
    mapping = json.loads((source / 'mapping.json').read_text())
    metadata = read(source / 'metadata.csv')
    if 'car' in metadata:
        metadata['vin'] = metadata.car.astype(str)
    snippets = metadata.set_index('sample_id').loc[mapping['sample_ids']['train']]
    assert snippets.sample_label.eq(0).all()
    vins = np.sort(snippets.vin.astype(str).unique())
    if 'partition' in metadata:
        other=metadata.loc[metadata.partition.ne('train'),'vin'].astype(str)
        assert not set(vins)&set(other),'Training VIN overlaps validation/test'
        assert metadata.loc[metadata.partition.eq('validation'),'sample_label'].eq(0).all()
    assert len(vins) >= 3
    count = min(len(vins)-1, max(1, int(np.ceil(len(vins)*calibration_fraction))))
    selected = sorted(np.random.default_rng(seed).choice(vins, count, replace=False).tolist())
    ref = np.load(source / 'residuals/train/references.npy', mmap_mode='r')
    calibration = snippets.vin.astype(str).isin(selected).to_numpy()[ref[:, 0].astype(int)]
    fit_indices, cal_indices = np.flatnonzero(~calibration), np.flatnonzero(calibration)
    record = dict(seed=seed, calibration_fraction=calibration_fraction,
        reconstruction_vins=sorted(set(vins)-set(selected)), calibration_vins=selected,
        reconstruction_windows=len(fit_indices), calibration_windows=len(cal_indices),
        source_mapping_sha256=digest(source / 'mapping.json'),
        pretrained_weights_reused=False, calibration_used_for_checkpoint_selection=False)
    if marker.exists():
        assert json.loads(marker.read_text()) == record
        np.testing.assert_array_equal(np.load(output/'fit_window_indices.npy'), fit_indices)
        np.testing.assert_array_equal(np.load(output/'calibration_window_indices.npy'), cal_indices)
    else:
        np.save(output/'fit_window_indices.npy', fit_indices)
        np.save(output/'calibration_window_indices.npy', cal_indices)
        atomic_json(marker, record)
    return fit_indices, cal_indices, mapping


def configure_reconstruction(model, stage, finetune=False):
    if stage != 'backbone':
        model.configure_stage(stage, finetune)
        return
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.backbone.parameters():
        p.requires_grad_(True)


def validation_loss(model, arrays, batch, relation=False):
    model.eval()
    total, count = 0., 0
    with torch.inference_mode():
        for lo in range(0, len(arrays.arrays['validation']), batch):
            item = arrays.get('validation', slice(lo, lo+batch))
            if relation:
                z, condition = item
                loss = F.huber_loss(model(z, condition), z, reduction='sum')
                elements = z.numel()
            else:
                loss = F.huber_loss(model(item), item, reduction='sum')
                elements = item.numel()
            total += float(loss)
            count += elements
    return total/count


def train_network(model, arrays, output, config, stage, teacher=None, relation=False,
                  memory_weight=.01, finetune=False):
    output.mkdir(parents=True, exist_ok=True)
    if not relation:
        configure_reconstruction(model, stage, finetune)
    if (output/'complete.json').exists():
        model.load_state_dict(torch.load(output/'best.pt', map_location=next(model.parameters()).device, weights_only=True))
        assert state_hash(model) == json.loads((output/'complete.json').read_text())['model_hash']
        return model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3,
        threshold=1e-4, threshold_mode='rel', min_lr=1e-6)
    cuda = next(model.parameters()).is_cuda
    eval_batch = min(config.batch,getattr(config,'microbatch',config.batch)) if relation else config.batch
    best = validation_loss(model, arrays, eval_batch, relation)
    anchor, best_epoch, stale, start, history, limit = best, 0, 0, 1, [], config.epochs
    if (output/'last.pt').exists():
        saved = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        best, anchor, best_epoch, stale = (saved[k] for k in ['best','anchor','best_epoch','stale'])
        start, history, limit = saved['epoch']+1, saved['history'], saved['limit']
        torch.set_rng_state(saved['rng'])
        if cuda: torch.cuda.set_rng_state_all(saved['cuda_rng'])
    else:
        atomic_checkpoint(output/'best.pt', model.state_dict())
        atomic_json(output/'initial.json', dict(parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in params), normal_validation_huber=best,
            stage=stage, relation_full_loo=relation))
    n = len(arrays.arrays['train'])
    stop = 'budget_limit'
    for epoch in range(start, config.max_epochs+1):
        if stale >= config.patience:
            stop = 'normal_validation_relative_patience'; break
        if epoch > limit:
            stop = 'normal_validation_stable_at_initial_limit'; break
        clock = time.monotonic(); model.train()
        if not relation and stage == 'memory': model.backbone.temporal.eval()
        order = np.random.default_rng(config.seed+epoch).permutation(n)
        total, memory_total, count = 0., 0., 0
        soft, hard, patch_counts = [], [], []
        for step, lo in enumerate(range(0, n, config.batch), 1):
            item = arrays.get('train', order[lo:lo+config.batch])
            optimizer.zero_grad(set_to_none=True)
            if relation:
                z, condition = item
                rec = z.new_zeros(())
                micro = min(config.batch,getattr(config,'microbatch',config.batch))
                for offset in range(0,len(z),micro):
                    target=z[offset:offset+micro]
                    partial=F.huber_loss(model(target,condition[offset:offset+micro]),target)
                    if not torch.isfinite(partial):raise FloatingPointError(f'{output}: nonfinite LOO loss')
                    weighted=partial*len(target)/len(z)
                    weighted.backward()
                    rec += weighted.detach()
                memory_loss = rec.new_zeros(())
                size = len(z)
            else:
                targets = None
                if teacher is not None and model.use_memory and memory_weight:
                    with torch.no_grad(): targets = teacher.patches(item)
                reconstruction, memory_loss, _, weights = model(item, details=True, teacher_patches=targets)
                rec = F.huber_loss(reconstruction, item); size = len(item)
                for s, w in enumerate(weights):
                    if w is None: continue
                    if len(soft) <= s:
                        soft.append(torch.zeros(w.shape[-1], device=w.device, dtype=torch.float64))
                        hard.append(torch.zeros_like(soft[-1])); patch_counts.append(0)
                    soft[s] += w.detach().double().sum((0,1))
                    hard[s] += torch.bincount(w.detach().argmax(-1).flatten(), minlength=w.shape[-1])
                    patch_counts[s] += w.shape[0]*w.shape[1]
            loss = rec + memory_weight*memory_loss
            if not torch.isfinite(loss): raise FloatingPointError(f'{output}: nonfinite loss')
            if not relation:loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.); optimizer.step()
            total += float(rec.detach())*size; memory_total += float(memory_loss.detach())*size; count += size
            if step % 2000 == 0:
                print(json.dumps(dict(event='FULL_BATCH',path=str(output),epoch=epoch,batch=step,
                    batches=int(np.ceil(n/config.batch)),loss=total/count)),flush=True)
        val = validation_loss(model, arrays, eval_batch, relation)
        if not np.isfinite(val): raise FloatingPointError('Nonfinite normal validation loss')
        if val < best:
            best, best_epoch = val, epoch
            atomic_checkpoint(output/'best.pt', model.state_dict())
        if val < anchor-max(1e-12, abs(anchor)*1e-4): anchor, stale = val, 0
        else: stale += 1
        scheduler.step(val)
        row = dict(epoch=epoch, reconstruction_huber=total/count, teacher_memory_loss=memory_total/count,
            normal_validation_huber=val,best_epoch=best_epoch,lr=optimizer.param_groups[0]['lr'],
            seconds=time.monotonic()-clock,full_training_windows=count,all_target_variables=relation)
        history.append(row); pd.DataFrame(history).to_csv(output/'history.csv',index=False)
        if soft:
            atomic_json(output/f'memory_epoch_{epoch:03d}.json', dict(epoch=epoch,normal_fit_only=True,
                soft_occupancy=[(v/patch_counts[s]).cpu().tolist() for s,v in enumerate(soft)],
                hard_occupancy=[(v/patch_counts[s]).cpu().tolist() for s,v in enumerate(hard)]))
        if epoch == config.epochs and config.max_epochs > config.epochs:
            recent = [h['normal_validation_huber'] for h in history[-config.patience:]]
            improvement = (recent[0]-min(recent))/max(abs(recent[0]),1e-12)
            if best_epoch >= epoch-config.patience and improvement > .001 and stale < config.patience:
                limit = config.max_epochs
        atomic_checkpoint(output/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),epoch=epoch,best=best,anchor=anchor,best_epoch=best_epoch,
            stale=stale,history=history,limit=limit,rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if cuda else []))
        print(json.dumps(dict(event='FULL_EPOCH',path=str(output),**row)),flush=True)
    model.load_state_dict(torch.load(output/'best.pt', map_location=next(model.parameters()).device,weights_only=True))
    atomic_json(output/'complete.json',dict(model_hash=state_hash(model),best_epoch=best_epoch,
        epochs_executed=len(history),normal_validation_huber=best,stop_reason=stop,
        initial_limit=config.epochs,actual_limit=limit,hard_limit=config.max_epochs,
        at_final_budget_boundary=best_epoch==limit,normal_fit_windows=n,relation_full_loo=relation,
        teacher_hash=state_hash(teacher) if teacher is not None else None,memory_weight=memory_weight,
        checkpoint_selection='normal validation Huber only',calibration_vins_seen=False))
    return model.eval()


def ranked_frames(source, mapping, window_values, names):
    metadata = read(source/'metadata.csv').set_index('sample_id')
    brand = mapping.get('dataset') == 'Brand3'
    if brand: metadata['vin'] = metadata.car.astype(str)
    frames, snippets = {}, {}
    for partition in PARTITIONS:
        refs = np.load(source/'residuals'/partition/'references.npy',mmap_mode='r')
        snippets[partition] = metadata.loc[mapping['sample_ids'][partition]].reset_index().merge(
            aggregate(window_values[partition],refs,mapping['sample_ids'][partition],names),
            on='sample_id',validate='one_to_one')
        frames[partition] = rank_frame(snippets[partition],names,brand)
    return frames, snippets


def evaluate_frames(unit, output, frames, names, snippets=None):
    rows, thresholds = [], {}
    for partition, frame in frames.items(): frame.to_csv(output/f'{partition}_scores.csv',index=False)
    if snippets is not None:
        for partition,frame in snippets.items():frame.to_csv(output/f'{partition}_snippet_scores.csv',index=False)
    assert frames['validation'].sample_label.eq(0).all()
    test = frames['test']
    for name in names:
        thresholds[name] = {}
        fpr,tpr,_ = roc_curve(test.sample_label,test[name])
        for q in [.95,.99]:
            threshold = float(frames['validation'][name].quantile(q)); thresholds[name][str(q)] = threshold
            rows.append(dict(unit=unit,method=name,q=q,threshold=threshold,
                validation_fpr=float((frames['validation'][name]>=threshold).mean()),
                recall_at_test_fpr05=float(tpr[fpr<=.05].max(initial=0.)),
                recall_at_test_fpr01=float(tpr[fpr<=.01].max(initial=0.)),
                **metrics(test.sample_label,test[name],threshold)))
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
    atomic_json(output/'thresholds.json',thresholds)
    return rows
