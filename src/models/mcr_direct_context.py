"""Actual frozen Memory retrieval after masking each raw target variable."""
import torch
from torch import nn

from src.models.rcr import condition_features


class DirectMaskedMemoryContext(nn.Module):
    def __init__(self, reconstructor, neutral, mean, std, low, high):
        super().__init__()
        self.reconstructor = reconstructor.eval()
        for p in self.reconstructor.parameters():p.requires_grad_(False)
        for name,value in [('neutral',neutral),('mean',mean),('std',std),('low',low),('high',high)]:
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))

    def forward(self,x,raw_q=None):
        q=condition_features(x,1,2) if raw_q is None else raw_q
        standardized=torch.maximum(torch.minimum((q-self.mean)/self.std,self.high),self.low)
        contexts=[]
        for j in range(x.shape[-1]):
            masked=x.clone();masked[:,:,j]=self.neutral[j]
            weights=[]
            for query,memory in zip(self.reconstructor.patches(masked),self.reconstructor.memories):
                _,_,attention=memory(query)
                weights.append(attention.mean(1))
            controls=standardized.clone()
            if j==1:controls[:,:4]=0
            if j==2:controls[:,4:]=0
            contexts.append(torch.cat((controls,torch.cat(weights,-1)),-1))
        return torch.stack(contexts,1)

    @torch.no_grad()
    def audit(self,x):
        baseline=self(x)
        errors=[]
        for j in range(x.shape[-1]):
            changed=x.clone();changed[:,:,j]+=10
            errors.append(float((self(changed)[:,j]-baseline[:,j]).abs().max()))
        assert max(errors)<1e-6,'Raw target leaked through direct Memory condition'
        return dict(raw_target_context_max_error=max(errors),per_target_errors=errors,
            neutral_source='normal reconstruction training VINs only',
            memory_frozen=all(not p.requires_grad for p in self.reconstructor.parameters()))
