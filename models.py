import random
import numpy as np

import torch
import torch.nn as nn

import torch_geometric.nn as gnn 

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, number_of_classes, num_of_hidden_layers, device, heads=8):
        super().__init__()
        torch.manual_seed(42)
        self.num_of_hidden_layers = num_of_hidden_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.number_of_classes = number_of_classes
        self.device = device
        self.heads = heads
        self.model = self.build_model().to(device)

    def forward(self, x, edge_index):
        return self.model(x, edge_index)
    
    def build_model(self):
        layers=[]
        for i in range(self.num_of_hidden_layers):
            #layers.append((nn.Dropout(0.5), 'x -> x'))
            layers.append((gnn.GATConv(in_channels=self.in_channels if i==0 else self.hidden_channels,
                                  out_channels=self.hidden_channels,
                                  head=self.heads),'x, edge_index -> x'))
            layers.append(nn.ELU(inplace=True))
        layers.append(nn.Linear(in_features=self.hidden_channels, out_features=self.number_of_classes))
        return gnn.Sequential('x, edge_index', layers)
    
    def get_n_params(self):
        pp=0
        for p in list(self.model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp