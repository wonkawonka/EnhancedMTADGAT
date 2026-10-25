"""Strict residual-coordinate LOO sequence predictors conditioned on normal context.

The default graph prior uses current/SOC to predict normal Memory occupancy.
No response-dependent test adaptation or target-valued attention queries.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.rcr import MaskedResidualGraphNet


class NormalContextMemoryPrior(nn.Module):
    """Fixed normal-only log-occupancy regression, excluding target controls.

    Predicting current never reads current statistics; predicting SOC never
    reads SOC statistics. Other response targets may use all eight controls.
    The Memory state is projected from normal training states, rather than
    recomputed from potentially anomalous response variables at test time.
    """
    def __init__(self, mean, std, low, high, coefficients, slots=8):
        super().__init__()
        for name,value in [('mean',mean),('std',std),('low',low),('high',high),('coefficients',coefficients)]:
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
        self.slots=slots
        self.variables=coefficients.shape[0]
        keep=np.ones((self.variables,8),dtype=np.float32)
        keep[1,:4]=0;keep[2,4:]=0
        self.register_buffer('keep',torch.from_numpy(keep))

    @classmethod
    def fit(cls,q,memory_state,ridge=.001,variables=7):
        q=np.asarray(q,dtype=np.float64);state=np.asarray(memory_state,dtype=np.float64)
        if q.shape!=(len(state),8) or state.shape[1]%8:
            raise ValueError('Expected aligned normal controls and scale-wise eight-slot occupancy')
        mean,std=q.mean(0),np.maximum(q.std(0),1e-8)
        standardized=(q-mean)/std
        low,high=np.quantile(standardized,[.01,.99],axis=0)
        standardized=np.clip(standardized,low,high)
        target=np.log(np.maximum(state,1e-6))
        coefficients=np.zeros((variables,9,state.shape[1]))
        for target_index,columns in [('response',np.arange(8)),('current',np.arange(4,8)),('soc',np.arange(4))]:
            x=np.column_stack((standardized[:,columns],np.ones(len(q))))
            penalty=np.eye(x.shape[1])*ridge;penalty[-1,-1]=0
            beta=np.linalg.solve(x.T@x/len(x)+penalty,x.T@target/len(x))
            targets=[j for j in range(variables) if j not in [1,2]] if target_index=='response' else [1 if target_index=='current' else 2]
            for j in targets:
                coefficients[j,columns]=beta[:-1];coefficients[j,-1]=beta[-1]
        return cls(mean,std,low,high,coefficients)

    def forward(self,q):
        standardized=torch.maximum(torch.minimum((q-self.mean)/self.std,self.high),self.low)
        masked=standardized[:,None,:]*self.keep[None]
        design=torch.cat((masked,torch.ones_like(masked[...,:1])),-1)
        logits=torch.einsum('bni,nic->bnc',design,self.coefficients)
        weights=logits.reshape(len(q),self.variables,-1,self.slots).softmax(-1).flatten(-2)
        return torch.cat((masked,weights),-1)

    def save(self,path):
        np.savez(path,**{name:value.detach().cpu().numpy() for name,value in self.state_dict().items() if name!='keep'})

    @classmethod
    def load(cls,path):
        with np.load(path) as values:
            return cls(**{name:values[name].copy() for name in values.files})


class VariableTCN(nn.Module):
    def __init__(self,hidden=32):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(1,hidden,3,padding=1),nn.GELU(),
            nn.Conv1d(hidden,hidden,3,padding=1),nn.GELU(),nn.Dropout(.1))
        self.project=nn.Linear(hidden*4,hidden)
        self.norm=nn.LayerNorm(hidden)

    def forward(self,z):
        b,t,d=z.shape
        features=self.conv(z.transpose(1,2).reshape(b*d,1,t))
        # Four temporal bins retain coarse temporal order in each variable token.
        features=F.adaptive_avg_pool1d(features,4).flatten(1)
        return self.norm(self.project(features)).reshape(b,d,-1)


class MCRSequencePredictor(nn.Module):
    def __init__(self,mode,length,condition_dim=32,variables=7,hidden=32,heads=2,templates=4,rank=4):
        super().__init__()
        if mode not in ['gat','memory_gat','memory_mlp','q_only_gat']:raise ValueError(mode)
        self.mode,self.variables,self.heads,self.hidden=mode,variables,heads,hidden
        self.encoder=VariableTCN(hidden)
        self.identity=nn.Parameter(torch.randn(variables,hidden)*.02)
        if mode!='memory_mlp':
            self.query=nn.Linear(hidden,hidden,bias=False)
            self.key=nn.Linear(hidden,hidden,bias=False)
            self.value=nn.Linear(hidden,hidden,bias=False)
            self.decoder=nn.Sequential(nn.Linear(hidden*3,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,length))
            self.attention_dropout=nn.Dropout(.1)
        if mode!='gat':
            self.context=nn.Sequential(nn.Linear(condition_dim,32),nn.GELU(),nn.Linear(32,hidden))
        if mode in ['memory_gat','q_only_gat']:
            self.mix=nn.Linear(hidden,templates)
            self.template_u=nn.Parameter(torch.randn(templates,variables,rank)*.02)
            self.template_v=nn.Parameter(torch.randn(templates,variables,rank)*.02)
        if mode=='q_only_gat':
            self.register_buffer('memory_constant',torch.full((condition_dim-8,),1/8))
        if mode=='memory_mlp':
            self.predictor=nn.Sequential(nn.Linear(variables*hidden+hidden*2,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,length))

    def forward(self,z,condition,targets=None,details=False):
        b,t,d=z.shape
        tokens=self.encoder(z)+self.identity[None]
        if self.mode=='q_only_gat':
            condition=torch.cat((condition[...,:8],self.memory_constant.expand(b,d,-1)),-1)
        context=self.context(condition) if self.mode!='gat' else tokens.new_zeros((b,d,self.hidden))
        identity=self.identity[None].expand(b,-1,-1)
        mask=torch.eye(d,device=z.device,dtype=torch.bool)
        attention,bias=None,None
        if self.mode=='memory_mlp':
            others=tokens[:,None].expand(b,d,d,self.hidden).masked_fill(mask[None,:,:,None],0.)
            pred=self.predictor(torch.cat((others.flatten(-2),identity,context),-1))
        else:
            h=self.hidden//self.heads
            query=self.query(self.identity).reshape(d,self.heads,h).permute(1,0,2)
            key=self.key(tokens).reshape(b,d,self.heads,h).transpose(1,2)
            value=self.value(tokens).reshape(b,d,self.heads,h).transpose(1,2)
            logits=torch.einsum('hid,bhjd->bhij',query,key)/(h**.5)
            if self.mode in ['memory_gat','q_only_gat']:
                templates=torch.einsum('knr,kmr->knm',self.template_u,self.template_v)
                bias=torch.einsum('bik,kij->bij',self.mix(context).softmax(-1),templates)
                logits=logits+bias[:,None]
            attention=logits.masked_fill(mask[None,None],-torch.inf).softmax(-1)
            message=(self.attention_dropout(attention)@value).transpose(1,2).reshape(b,d,self.hidden)
            pred=self.decoder(torch.cat((message,identity,context),-1))
        pred=pred.transpose(1,2)
        if targets is not None:
            pred=pred.gather(-1,targets[:,None,None].expand(-1,t,1)).squeeze(-1)
        return (pred,attention,bias) if details else pred


class OriginalRCRLOOPredictor(nn.Module):
    """Existing RCR graph/Transformer relation head, refitted on the same new z."""
    def __init__(self,variables=7,condition_dim=32):
        super().__init__()
        self.variables=variables
        self.net=MaskedResidualGraphNet(variables,condition_dim=condition_dim,condition_mode='mlp')

    def forward(self,z,condition,targets=None):
        b,t,d=z.shape
        if targets is not None:
            mask=F.one_hot(targets,d).bool()[:,None].expand(b,t,d).reshape(b*t,d)
            c=condition[torch.arange(b,device=z.device),targets][:,None].expand(b,t,-1).reshape(b*t,-1)
            prediction=self.net(z.reshape(b*t,d),c,mask).reshape(b,t,d)
            return prediction.gather(-1,targets[:,None,None].expand(b,t,1)).squeeze(-1)
        mask=torch.eye(d,device=z.device,dtype=torch.bool)[None,:,None,:].expand(b,d,t,d).reshape(-1,d)
        values=z[:,None].expand(b,d,t,d).reshape(-1,d)
        c=condition[:,:,None].expand(b,d,t,-1).reshape(b*d*t,-1)
        prediction=self.net(values,c,mask).reshape(b,d,t,d)
        selected=torch.arange(d,device=z.device)[None,:,None,None].expand(b,d,t,1)
        return prediction.gather(-1,selected).squeeze(-1).transpose(1,2)
