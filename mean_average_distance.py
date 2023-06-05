import torch
import torch.nn as nn
import torch_geometric
from torchmetrics.functional import pairwise_cosine_similarity


class MAD(nn.Module):
    '''
    Calculate <Mean Average Distance> of given node features.
    if global flag is False, adjacency matrix is needed when forward call
    '''
    def __init__(self, device, global_flag:bool):
        super().__init__()
        self.device=device
        self.global_flag=global_flag
    
    def forward(self, node_features, target:torch.tensor=None):
        D = (1 - pairwise_cosine_similarity(node_features)).to(self.device)
        if self.global_flag:
            M_tgt = torch.ones(node_features.shape[0], node_features.shape[0], device = self.device)
        else:
            M_tgt = target
        M_tgt.fill_diagonal_(0)
        
        D_tgt = torch.mul(D, M_tgt).sum(axis=-1) / M_tgt.sum(axis=-1)
        mad = torch.nansum(D_tgt) / (M_tgt.sum(axis=-1)/(M_tgt.sum(axis=-1)+1e-7)).sum()
        return mad
    
class MADGap(nn.Module):
    '''
    Calculate <Mean Average Distance Gap> of given node features.
    embedding of the nodes and the edge index is needed for the MADGap Calculation
    '''
    def __init__(self, device, neighbor_order, remote_order):
        super().__init__()
        self.device = device
        self.neighbor_order = neighbor_order
        self.remote_order = remote_order
        self.MAD = MAD(device=device, global_flag = False)
        
    def forward(self, embedding, edge_index):
        adj_mat = torch_geometric.utils.to_dense_adj(edge_index).to(self.device)
        neighbor_adj = torch.linalg.matrix_power(adj_mat, self.neighbor_order).to(torch.bool)[0]
        neighbor_mad = self.MAD(embedding, neighbor_adj)
        remote_adj = torch.linalg.matrix_power(adj_mat, self.remote_order).to(torch.bool)[0]
        remote_mad = self.MAD(embedding, remote_adj)
        return remote_mad - neighbor_mad
        