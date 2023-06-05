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