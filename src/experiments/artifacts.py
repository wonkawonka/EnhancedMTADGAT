"""Explicit efficiency accounting and small, verifiable result archives."""
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import threading
import time
import zipfile
from pathlib import Path
import numpy as np
import psutil
import torch
from src.experiment_preflight import sha256
from .battery_data import write_json


class Efficiency:
    def __init__(self, output, device):
        self.output=Path(output);self.device=device
        self.started=time.monotonic();self.phases=[];self.rss_peak=0;self.stack=[]
        old=self.output/'efficiency.json'
        previous=json.loads(old.read_text()) if old.exists() else None
        if previous:self.phases.extend(previous.get('phases',[]))
        self.stop_event=threading.Event()
        self.process=psutil.Process()
        def sample():
            while not self.stop_event.is_set():
                self.rss_peak=max(self.rss_peak,self.process.memory_info().rss)
                self.stop_event.wait(.5)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start()
        cuda=device.type=='cuda'
        cpu=platform.processor()
        cpuinfo=Path('/proc/cpuinfo')
        if cpuinfo.exists():
            cpu=next((line.split(':',1)[1].strip() for line in cpuinfo.read_text().splitlines() if line.startswith('model name')),cpu)
        self.record=dict(environment=dict(cpu=cpu,available_cpus=len(os.sched_getaffinity(0)),
                         torch=torch.__version__,python=platform.python_version(),numpy=np.__version__,
                         package_versions={name:importlib.metadata.version(name) for name in ('scikit-learn','pandas','psutil','einops','joblib','threadpoolctl')},
                         cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),torch_threads=torch.get_num_threads(),
                         GPU=torch.cuda.get_device_name(device) if cuda else None,
                         GPU_total_memory_bytes=torch.cuda.get_device_properties(device).total_memory if cuda else None,
                         RAM_total_bytes=psutil.virtual_memory().total),phases=self.phases,
                         RSS_sampling_interval_seconds=.5,device_utilization='N/A: no device sampler enabled',
                         previous_process_wall_seconds=(previous.get('previous_process_wall_seconds',0)+previous.get('wall_seconds_this_process',0)) if previous else 0)

    @contextlib.contextmanager
    def phase(self, name, **detail):
        cuda=self.device.type=='cuda'
        if cuda:
            torch.cuda.synchronize(self.device)
            if self.stack:
                self.stack[-1]['allocated']=max(self.stack[-1]['allocated'],torch.cuda.max_memory_allocated(self.device))
                self.stack[-1]['reserved']=max(self.stack[-1]['reserved'],torch.cuda.max_memory_reserved(self.device))
            torch.cuda.reset_peak_memory_stats(self.device)
        frame=dict(allocated=0,reserved=0,child_seconds=0.)
        self.stack.append(frame)
        started=time.monotonic()
        try:yield detail
        finally:
            if cuda:torch.cuda.synchronize(self.device)
            elapsed=time.monotonic()-started
            self.stack.pop()
            row=dict(name=name,seconds=elapsed,exclusive_seconds=elapsed-frame['child_seconds'],**detail)
            row['peak_cuda_allocated_bytes']=max(frame['allocated'],torch.cuda.max_memory_allocated(self.device)) if cuda else None
            row['peak_cuda_reserved_bytes']=max(frame['reserved'],torch.cuda.max_memory_reserved(self.device)) if cuda else None
            if self.stack:
                parent=self.stack[-1];parent['child_seconds']+=elapsed
                if cuda:
                    parent['allocated']=max(parent['allocated'],row['peak_cuda_allocated_bytes'])
                    parent['reserved']=max(parent['reserved'],row['peak_cuda_reserved_bytes'])
            self.phases.append(row);self.flush()

    def flush(self):
        self.record.update(wall_seconds_this_process=time.monotonic()-self.started,process_RSS_sampled_peak_bytes=self.rss_peak)
        for row in self.phases:
            seconds=row.get('exclusive_seconds',row['seconds'])
            if seconds>0:
                for key in ('windows','points','snippets'):
                    if key in row:row[key+'_per_second']=row[key]/seconds
        write_json(self.output/'efficiency.json',self.record)

    def close(self):
        self.stop_event.set();self.thread.join();self.flush()


def _archive(files, path, base):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    members={}
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
        for f in sorted(files):
            name=str(f.relative_to(base))
            z.write(f,name)
            members[name]=dict(bytes=f.stat().st_size,sha256=sha256(f))
        z.writestr('archive_manifest.json',json.dumps({'members':members},ensure_ascii=False))
    with zipfile.ZipFile(temporary) as z:
        error=z.testzip()
        if error:raise ValueError('Archive CRC failure: '+error)
        for name,info in members.items():
            h=hashlib.sha256()
            with z.open(name) as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=info['sha256']:raise ValueError('Archive hash mismatch: '+name)
    temporary.replace(path)
    return dict(file=path.name,bytes=path.stat().st_size,sha256=sha256(path),members=len(members))


def package_cell(cell, exports, identity, include_groups=None):
    """Only known generated files; never include inputs, repository or credentials."""
    started=time.monotonic()
    cell=Path(cell);exports=Path(exports)
    groups={'results':[],'weights':[],'native_scores':[],'recovery':[]}
    for f in cell.iterdir():
        if not f.is_file():continue
        if f.name in ('latest.pt','interrupt.pt'):groups['recovery'].append(f)
        elif f.name in ('selected.pt','forest.joblib','calibration.npz','center.npy'):groups['weights'].append(f)
        elif f.name.endswith('_native.npy') or f.name.endswith('_native_mapping.npz'):
            groups['native_scores'].append(f)
        elif f.suffix in ('.json','.jsonl','.csv','.txt','.npz'):
            groups['results'].append(f)
    previous=exports/f'{identity}_archive_index.json'
    retained=[]
    if include_groups is not None and previous.exists():
        retained=[row for row in json.loads(previous.read_text())['archives'] if row['group'] not in include_groups]
    index=dict(identity=identity,archives=retained,compression='lossless deflate, original dtype',dependencies=[])
    # Per-cell groups are usually small. Split at file boundaries without discarding evidence.
    for group,files in groups.items():
        if include_groups is not None and group not in include_groups:continue
        if not files:continue
        shards=[[]];size=0;limit=(256 if group=='results' else 512)*2**20
        for f in sorted(files):
            if size+f.stat().st_size>limit and shards[-1]:shards.append([]);size=0
            shards[-1].append(f);size+=f.stat().st_size
        for i,shard in enumerate(shards):
            result=_archive(shard,exports/f'{identity}_{group}_{i+1:02d}.zip',cell)
            index['archives'].append(dict(group=group,**result))
    index['packaging_seconds']=time.monotonic()-started
    index['archive_bytes']=sum(row['bytes'] for row in index['archives'])
    write_json(exports/f'{identity}_archive_index.json',index)
    return index


def summarize_output(root):
    """Flat CSV indexes for this output only; incomplete jobs remain explicit."""
    import csv
    root=Path(root);rows=[];eff=[]
    for path in sorted(root.glob('*/*/seed*/metrics.json')):
        row=json.loads(path.read_text())
        row['complete']=(path.parent/'complete.json').exists()
        rows.append(row)
        perf=path.parent/'efficiency.json'
        if perf.exists():
            p=json.loads(perf.read_text());summary={k:row.get(k) for k in ('dataset','experiment','seed')}
            summary.update(GPU=p['environment'].get('GPU'),parameters=p.get('parameters_total'),
                           wall_seconds=p.get('wall_seconds_this_process',0)+p.get('previous_process_wall_seconds',0),
                           RSS_peak_bytes=p.get('process_RSS_sampled_peak_bytes'))
            for phase in p['phases']:
                name=phase['name']
                summary[name+'_seconds']=summary.get(name+'_seconds',0)+phase.get('exclusive_seconds',phase['seconds'])
                for key in ('peak_cuda_allocated_bytes','peak_cuda_reserved_bytes'):
                    if phase.get(key) is not None:summary[key]=max(summary.get(key,0),phase[key])
            eff.append(summary)
    for name,data in [('metrics_summary.csv',rows),('efficiency_summary.csv',eff)]:
        if not data:continue
        keys=sorted(set().union(*(row.keys() for row in data)))
        with (root/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(data)
