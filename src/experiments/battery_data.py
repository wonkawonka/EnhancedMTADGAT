"""Frozen, portable input bundles with explicit boundaries and lazy windows."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from src.experiment_preflight import sha256

PARTS = ('train', 'validation', 'test')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def starts(n, span, stride):
    check(n >= span and stride > 0, 'Invalid snippet length/window')
    loc = np.arange(0, n-span+1, stride, dtype=np.int64)
    return np.append(loc, n-span) if loc[-1] != n-span else loc


def window_starts(lengths, span, stride):
    offset = 0
    rows = []
    counts = []
    for n in lengths:
        local = starts(int(n), span, stride)
        rows.append(local+offset)
        counts.append(len(local))
        offset += int(n)
    return np.concatenate(rows), np.asarray(counts, dtype=np.int64)


def finish_bundle(dest, dataset, features, mapping, source):
    dest = Path(dest)
    entities = {p:set(map(str, mapping[p]['entities'])) for p in PARTS}
    check(all(entities[a].isdisjoint(entities[b]) for a,b in
              [('train','validation'),('train','test'),('validation','test')]), 'Entity leakage')
    counts, points = {}, {}
    for part in PARTS:
        x = np.load(dest/f'{part}_raw.npy', mmap_mode='r')
        ns = np.load(dest/f'{part}_lengths.npy')
        labels = np.asarray(mapping[part]['labels'], dtype=np.int8)
        check(ns.sum() == len(x) and ns.min() >= 105 and x.shape[1] == 7, 'Input shapes/boundaries')
        check(len(labels) == len(ns) == len(mapping[part]['sample_ids']), 'Label/ID alignment')
        check(len(set(mapping[part]['sample_ids'])) == len(ns), 'Duplicate sample IDs')
        check(part == 'test' or np.all(labels == 0), 'Fault in training/validation')
        for lo in range(0, len(x), 1000000):
            check(np.isfinite(x[lo:lo+1000000]).all(), 'Invalid prepared values')
        np.save(dest/f'{part}_labels.npy', labels)
        counts[part], points[part] = len(ns), int(ns.sum())
    train = np.load(dest/'train_raw.npy', mmap_mode='r')
    minimum, maximum = train.min(0).astype(np.float64), train.max(0).astype(np.float64)
    np.savez(dest/'scaler.npz', minimum=minimum, maximum=maximum, features=features)
    write_json(dest/'mapping.json', mapping)
    manifest = dict(schema_version=1,status='passed',dataset=dataset,features=features,
                    counts=counts,points=points,entity_isolation=True,source=source,
                    minimum_length=105,scaler_fit='each retained normal training real point once',
                    original_bad_value_audit='zero invalid in retained source data; audit preserved as source evidence',
                    files={p.name:sha256(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name != 'bundle_manifest.json'})
    write_json(dest/'bundle_manifest.json', manifest)
    print(json.dumps({'prepared':dataset,'counts':counts,'points':points}), flush=True)


def prepare_all(output):
    """CPU preparation only. Existing verified clean packs reused, NCM test parsed once."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    check(not (out/'bundle_manifest.json').exists(), 'Existing completed bundle; do not overwrite')
    audit = json.loads(Path('runs/external/original_missing_audit_20261112_v1/summary.json').read_text())
    for ds in ('LFP','NCM'):
        dest = out/ds
        if (dest/'bundle_manifest.json').exists():
            continue
        dest.mkdir(exist_ok=True)
        source = Path('runs/internal/rcr_v1')/ds/'full'
        oldmap = json.loads((source/'mapping.json').read_text())
        meta = pd.read_csv(source/'metadata.csv',dtype={'vin':str}).set_index('sample_id')
        cohort = pd.read_csv('runs/external/reference_epoch_timing_20261111_v1/inputs/'+ds+'_cohort_lengths.csv')
        mapping = {}
        for part in PARTS:
            table = cohort[(cohort.partition == part) & cohort.kept]
            ids = table.sample_id.tolist()
            ns = table.length.to_numpy(dtype=np.int64)
            mapping[part] = dict(sample_ids=ids,entities=meta.loc[ids,'vin'].astype(str).tolist(),labels=table.label.astype(int).tolist())
            if ds == 'LFP':
                prior = Path('runs/external/lfp_stride_pair_20261112_v1/inputs')
                prior_mapping = json.loads((prior/'mapping.json').read_text())
                check(mapping[part] == prior_mapping[part], 'LFP immutable cache mapping changed')
                file = prior/f'{part}_raw.npy'
                evidence = json.loads((prior/'preparation.json').read_text())
                expected = evidence['files'][str(file)]
                check(sha256(file) == expected, 'LFP cache fingerprint')
                if not (dest/file.name).exists():
                    os.link(file,dest/file.name)
            elif part in ('train','validation'):
                prior = Path('runs/external/reference_epoch_timing_20261111_v1/inputs')
                check(json.loads((prior/f'NCM_{part}_ids.json').read_text()) == ids, 'NCM cached IDs')
                check(np.array_equal(np.load(prior/f'NCM_{part}_lengths.npy'), ns), 'NCM cached lengths')
                if not (dest/f'{part}_raw.npy').exists():
                    os.link(prior/f'NCM_{part}_raw.npy', dest/f'{part}_raw.npy')
            else:
                raw = np.lib.format.open_memmap(dest/f'{part}_raw.npy',mode='w+',dtype=np.float32,shape=(int(ns.sum()),7))
                offset=0
                for i,(sid,n) in enumerate(zip(ids,ns)):
                    frame=pd.read_csv(meta.loc[sid,'file_path'],usecols=oldmap['feature_columns']).reindex(columns=oldmap['feature_columns'])
                    x=frame.apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float32)
                    check(x.shape==(n,7) and np.isfinite(x).all(), 'Previously audited NCM source changed; re-audit required')
                    raw[offset:offset+n]=x;offset+=n
                    if (i+1)%1000==0:print(json.dumps({'prepare':ds,'partition':part,'snippets':i+1}),flush=True)
                raw.flush();del raw
            np.save(dest/f'{part}_lengths.npy',ns)
        finish_bundle(dest,ds,oldmap['feature_columns'],mapping,{'mapping_sha256':sha256(source/'mapping.json'),'cohort_sha256':sha256('runs/external/reference_epoch_timing_20261111_v1/inputs/'+ds+'_cohort_lengths.csv'),'invalid_audit_sha256':sha256('runs/external/original_missing_audit_20261112_v1/summary.json')})
    dest=out/'Brand1'
    if not (dest/'bundle_manifest.json').exists():
        dest.mkdir(exist_ok=True)
        split=Path('runs/external/brand1_split_70_10_20_seed0_20261112')
        m=json.loads((split/'mapping.json').read_text())
        split_manifest=json.loads((split/'split_manifest.json').read_text())
        packed=Path('kaggle/brand1_packed_inputs_20261109')
        ids=json.loads((packed/'brand1_sample_ids.json').read_text())
        old=np.load(packed/'brand1_raw.npy',mmap_mode='r')
        check(old.shape==(len(ids),128,7),'Brand1 original pack shape')
        mapping={}
        for part in PARTS:
            rows=np.load(split/f'{part}_rows.npy')
            check([ids[i] for i in rows]==m['sample_ids'][part],'Brand1 row/ID alignment')
            x=np.lib.format.open_memmap(dest/f'{part}_raw.npy',mode='w+',dtype=np.float32,shape=(len(rows)*128,7))
            for lo in range(0,len(rows),4096):
                chunk=old[rows[lo:lo+4096]]
                check(np.isfinite(chunk).all(),'Brand1 packed input invalid')
                x[lo*128:(lo+len(chunk))*128]=chunk.reshape(-1,7)
            x.flush();del x
            np.save(dest/f'{part}_lengths.npy',np.full(len(rows),128,dtype=np.int64))
            mapping[part]=dict(sample_ids=m['sample_ids'][part],entities=list(map(str,m['cars'][part])),labels=m['labels'][part])
        finish_bundle(dest,'Brand1',m['feature_columns'],mapping,{'split_id':split_manifest['split_id'],'split_manifest_sha256':sha256(split/'split_manifest.json'),'original_pack_sha256':sha256(packed/'brand1_raw.npy'),'invalid_audit_sha256':sha256('runs/external/original_missing_audit_20261112_v1/summary.json')})
    write_json(out/'bundle_manifest.json',{'schema_version':1,'status':'passed','datasets':{ds:{'manifest_sha256':sha256(out/ds/'bundle_manifest.json'),'path':ds} for ds in ('LFP','NCM','Brand1')}})


class Bundle:
    def __init__(self, root, dataset, expected, verify=True):
        self.root=Path(root)/dataset
        self.manifest=json.loads((self.root/'bundle_manifest.json').read_text())
        check(self.manifest['status']=='passed' and self.manifest['dataset']==dataset,'Wrong bundle')
        if verify:
            for name,h in self.manifest['files'].items():check(sha256(self.root/name)==h,'Bundle hash mismatch: '+name)
        self.mapping=json.loads((self.root/'mapping.json').read_text())
        self.raw={};self.lengths={};self.offsets={}
        self.scale=np.load(self.root/'scaler.npz')
        self.minimum=self.scale['minimum']
        width=self.scale['maximum']-self.minimum
        self.width=np.where(width<1e-12,1.,width)
        for part in PARTS:
            self.raw[part]=np.load(self.root/f'{part}_raw.npy',mmap_mode='r')
            self.lengths[part]=np.load(self.root/f'{part}_lengths.npy')
            self.offsets[part]=np.r_[0,np.cumsum(self.lengths[part])]
            count=expected['train'] if part=='train' else expected['validation'] if part=='validation' else expected['test_normal']+expected['test_fault']
            check(len(self.lengths[part])==count,'Unexpected full cohort count '+part)
            check(len(self.raw[part])==expected[part+'_points'],'Unexpected point count '+part)
        check(self.raw['train'].shape[1]==7,'Feature count')
        sets={p:set(self.mapping[p]['entities']) for p in PARTS}
        check(all(sets[a].isdisjoint(sets[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]),'Entity leakage')
        check(np.array_equal(self.raw['train'].min(0),self.minimum) and np.array_equal(self.raw['train'].max(0),self.scale['maximum']), 'Scaler not fitted on this normal train partition')

    def batch(self, part, positions, span, dtype=np.float32):
        ids=np.asarray(positions)[:,None]+np.arange(span)[None,:]
        x=self.raw[part][ids].astype(np.float64)
        return ((x-self.minimum)/self.width).astype(dtype)

    def point_batch(self, part, lo, hi):
        return ((self.raw[part][lo:hi].astype(np.float64)-self.minimum)/self.width).astype(np.float32)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',type=Path,required=True)
    prepare_all(p.parse_args().prepare)
