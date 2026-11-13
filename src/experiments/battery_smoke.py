"""Bounded numerical checks of the exact formal loss/scoring/gradient paths."""
import gc
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.ensemble import IsolationForest
from . import reference_models as ref
from .battery_data import write_json,check,starts
from src.models.omni_anomaly_port import TF1Adam,clip_gradients_per_parameter


def run_smoke(output,device,dataset='LFP',experiments=None):
    # All methods, including the full declared batches; no formal fitting or test labels.
    from src.runners.run_battery_formal import source_manifest,seed_all,evaluate,top_mean,object_scores
    from .artifacts import package_cell
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    plan_path=ref.ROOT/'configs/external/battery_kaggle_formal_20261112_v1.json'
    plan=json.loads(plan_path.read_text());record=dict(status='running',device=str(device),torch=torch.__version__,dataset=dataset,methods={},experiments={})
    record['files']=source_manifest(plan)['files']
    seed_all(3407)
    jobs=[j for j in plan['cells'] if j['dataset']==dataset and j['method']!='iforest'
          and (experiments is None or j['experiment'] in experiments)]
    for job in jobs:
        method=job['method']
        cfg=job['config'];arch=dict(plan['models'][method],window=cfg['window'])
        arch.update(job.get('architecture_overrides',{}))
        model,compat=ref.build(method,device,arch)
        dtype=torch.float64 if method=='tranad' else torch.float32
        # Numerical/reference comparisons use three independent windows.
        x=torch.linspace(.05,.95,3*cfg['span']*7,device=device,dtype=dtype).reshape(3,cfg['span'],7)
        model.eval();kl=ref.kl_for(method) if method in ('at','dc') else None
        center=torch.linspace(.1,.3,32,device=device) if method=='svdd' else None
        epoch=1
        actual=ref.terms(method,model,x,epoch,kl,center)
        if method=='tranad':
            seq=x.transpose(0,1);first,second=model(seq,seq[-1:])
            for n in (1,2,5):
                expected=F.mse_loss(first,seq[-1:])/n+(1-1/n)*F.mse_loss(second,seq[-1:])
                torch.testing.assert_close(ref.terms(method,model,x,n)[0],expected)
            decoder_grad=torch.autograd.grad(actual[0],tuple(model.transformer_decoder2.parameters()),retain_graph=True)
            check(all(torch.count_nonzero(g)==0 for g in decoder_grad),'TranAD epoch1 decoder2 gradient')
            late=ref.terms(method,model,x,2)[0]
            g=torch.autograd.grad(late,tuple(model.transformer_decoder2.parameters()))
            check(any(torch.count_nonzero(t)>0 for t in g),'TranAD epoch2 decoder2 never trained')
        if method=='rcr':
            rng=np.random.default_rng(3407);q=rng.normal(size=(128,8));r=rng.normal(size=(128,cfg['window'],7))*.1
            moments=np.stack([r.mean(1),r.var(1)],axis=1)
            cal=ref.fit_scale(q,moments,.05);dense=ref.LinearResidualCalibration(.05).fit(q,r)
            for key in ('mean_coef','scale_coef','scale_low','scale_high'):
                np.testing.assert_allclose(getattr(cal,key),getattr(dense,key),rtol=1e-8,atol=1e-10)
        with torch.no_grad():
            if method=='omni':
                generator=torch.Generator(device=device).manual_seed(3407)
                torch_score=ref.scores(method,model,x,generator=generator)
                generator.manual_seed(3407);z=model(x,noise=ref.noise(model,x,generator))
                expected=-z['logpx_dim'][:,-1].sum(-1)
            else:
                torch_score=ref.scores(method,model,x,kl,center=center)
                expected=[]
                for one in x.split(1):
                    if method=='tranad':
                        seq=one.transpose(0,1);_,second=model(seq,seq[-1:]);v=F.mse_loss(second,seq[-1:])
                    elif method=='mtad':
                        forecast,_=model(one[:,:100]);_,rec=model(one[:,1:]);v=((forecast-one[:,100]).abs()+(rec[:,-1]-one[:,100]).abs()).mean()
                    elif method=='rcr':v=(one-model(one)).double().square().mean()
                    elif method=='svdd':v=(model.encoder(one)-center).square().sum()
                    else:
                        if method=='at':rec,series,prior,_=model(one)
                        else:series,prior=model(one)
                        a,b=0.,0.
                        for s,p in zip(series,prior):
                            p=p/p.sum(-1,keepdim=True);a=a+50*kl(s,p.detach());b=b+50*kl(p,s.detach())
                        v=torch.softmax(-a-b,dim=-1)
                        if method=='at':v=v*(rec-one).square().mean(-1)
                        v=v[0]
                    expected.append(v)
                expected=torch.stack(expected)
            torch.testing.assert_close(torch_score,expected,atol=3e-6,rtol=5e-5)
        # First update at the exact declared physical batch catches OOM/API/gradient failures.
        model.train();batch=x[:1].expand(cfg['batch'],-1,-1).contiguous()
        model.zero_grad(set_to_none=True)
        vals=ref.terms(method,model,batch,2 if method=='tranad' else 1,kl,center)
        for j,term in enumerate(vals):term.backward(retain_graph=j<len(vals)-1)
        grads=[t.grad for t in model.parameters() if t.grad is not None]
        check(grads and all(torch.isfinite(g).all() for g in grads),'Nonfinite smoke gradient')
        opt=TF1Adam(model.parameters(),lr=cfg['lr']) if method=='omni' else (torch.optim.AdamW if cfg['optimizer']=='AdamW' else torch.optim.Adam)(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        if method=='omni':clip_gradients_per_parameter(model,10.)
        opt.step()
        check(all(torch.isfinite(t).all() for t in model.parameters()),'Nonfinite updated parameters')
        if method=='svdd':
            model.zero_grad(set_to_none=True);ref.terms(method,model,batch,1,stage='pretrain')[0].backward()
        record['methods'][method]=dict(status='passed',physical_batch=cfg['batch'],native_score_reference='passed',gradient='passed',compatibility=compat)
        record['experiments'][job['experiment']]=dict(status='passed',config=cfg,architecture_overrides=job.get('architecture_overrides',{}))
        write_json(out/'smoke.json',record)
        del model,opt,x,batch,grads,vals,actual,torch_score,expected
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    rng=np.random.default_rng(3407);x=rng.normal(size=(512,7)).astype(np.float32)
    forest=IsolationForest(n_estimators=200,max_samples=256,random_state=3407).fit(x)
    scores=-forest.score_samples(x)
    check(np.isfinite(scores).all(),'IF finite score')
    for span in (10,32,100,101,105,128):
        for stride in (1,4,5,100,128):
            a=starts(131,span,stride);check(a[0]==0 and a[-1]+span==131,'Tail alignment')
    values=np.array([.1,.3,.2,.9]);mapping=dict(labels=[0,0,1,1],entities=['a','a','b','b'],sample_ids=['1','2','3','4'])
    obj,labs,ids=object_scores(values,mapping,'Brand1')
    np.testing.assert_allclose(obj,[.3,.9]);check(ids==['a','b'],'Entity mapping')
    metrics=evaluate(obj,np.array([.1,.2]),labs);check(metrics['AP']==1.,'Metric integration')
    # Real exporter, including CRC and member hashes, on a tiny generated result.
    fixture=out/'archive_fixture';fixture.mkdir(exist_ok=True)
    write_json(fixture/'metrics.json',metrics);np.savez_compressed(fixture/'test_scores.npz',scores=values)
    package_cell(fixture,out/'archive_smoke','fixture')
    for job in plan['cells']:
        if job['dataset']==dataset and job['method']=='iforest' and (experiments is None or job['experiment'] in experiments):
            record['experiments'][job['experiment']]=dict(status='passed',config=job['config'],architecture_overrides={})
    check(record['files']==source_manifest(plan)['files'],'Sources changed during numerical checks')
    record.update(status='passed',IF='passed',boundaries='passed',aggregation='passed',archives='passed')
    write_json(out/'smoke.json',record)
    return record
