import torch.nn as nn
import torch
from torchmetrics.functional import pairwise_cosine_similarity

class ConstrativeLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, x):
        sim = pairwise_cosine_similarity(x)
        sim = (sim.sum(-1) / self.temperature).mean()
        return sim