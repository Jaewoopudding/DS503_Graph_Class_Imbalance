import random

import torch.nn as nn
import torch
from torchmetrics.functional import pairwise_cosine_similarity
from torch.nn.functional import triplet_margin_loss


class ConstrativeLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, x):
        sim = pairwise_cosine_similarity(x)
        sim = (sim.sum(-1) / self.temperature).mean()
        return sim
    
    
class ConstrativeLosswithPositiveSample(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.sample = None
        
    def forward(self, cnode_embedding, samples_embedding):
        sim_cnode = pairwise_cosine_similarity(cnode_embedding)
        sim_cnode = (sim_cnode.sum(-1) / self.temperature).mean()
        
        sim_sample = pairwise_cosine_similarity(cnode_embedding, samples_embedding).diag().mean()
        sim = sim_cnode - sim_sample
        return sim
    
    def sample_class_node(self, num_of_class, data):
        '''
        클래스 갯수와 data를 받아서
        0번부터 70번 라벨을 갖는 인덱스를 리턴한다. 
        '''
        for i in range(num_of_class):
            sample_source = (data.y[data.train_mask] == i).nonzero()
            idx = sample_source[random.randrange(0, sample_source.shape[0])]
            try: 
                result = torch.cat([result, idx], axis=0)
            except:
                result = torch.tensor(idx)  
        return result
    

class Triplet_on_closest_emb(nn.Module):
    def __init__(self, ):
        super().__init__()
    def forward(self, x):
        sim = pairwise_cosine_similarity(x)
        ind  = torch.argmax(sim.flatten()).item()
        row = ind // sim.shape[0]
        col = ind % sim.shape[0]
        loss = triplet_margin_loss(x[row],x[row], x[col])
        return loss