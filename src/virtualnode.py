import copy

import numpy as np
import torch
import torch_geometric
from torch_geometric.data.datapipes import functional_transform

@functional_transform('virtual_class_node')
class VirtualClassNode(torch_geometric.transforms.BaseTransform):
    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data :
        data = copy.deepcopy(data)
        num_nodes, (row, col) = data.num_nodes, data.edge_index
        edge_type = data.get('edge_type', torch.zeros_like(row))
        
        y = copy.deepcopy(data.y)
        y[(~data.train_mask).nonzero(as_tuple=True)[0]] = -1

        clss, cnts = y.unique(return_counts=True)
        clss = clss[1:]
        cnts = cnts[1:]
        for cls, cnt in zip(clss, cnts):
            
            node_idx_of_same_class = (y==cls).nonzero(as_tuple=True)[0]
            full = row.new_full((cnt, ), num_nodes)
            row = torch.cat([row, node_idx_of_same_class, full], dim=0)
            col = torch.cat([col, full, node_idx_of_same_class], dim=0)

            edge_index = torch.stack([row, col], dim=0)
            new_type = edge_type.new_full((cnt, ), int(edge_type.max()) + 1)
            edge_type = torch.cat([edge_type, new_type, new_type + 1], dim=0)
            num_nodes += 1

        for key, _ in data.items():
            if key == 'x':
                temp_x = torch.zeros((70, 8710)).to(data.x.device)
                data.x = torch.cat([data.x,temp_x], dim=0)
            elif key == 'y':
                temp_y = torch.arange(70).long().to(data.x.device)
                data.y = torch.cat([data.y, temp_y], dim=0)
            elif key == 'train_mask':
                temp_mask = torch.ones((70)).to(torch.bool).to(data.x.device)
                data.train_mask = torch.cat([data.train_mask, temp_mask]) 
            elif key == 'valid_mask':
                temp_mask = torch.ones((70)).to(torch.bool).to(data.x.device)
                data.valid_mask = torch.cat([data.valid_mask, temp_mask])
            elif key == 'test_mask':
                temp_mask = torch.ones((70)).to(torch.bool).to(data.x.device)
                data.test_mask = torch.cat([data.test_mask, temp_mask])
                    
        data.edge_index = edge_index
        data.edge_type = edge_type

        if 'num_nodes' in data:
            data.num_nodes = num_nodes

        return data
    

class UnidirectionalVirtualClassNode(torch_geometric.transforms.BaseTransform):
    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data :
        data = copy.deepcopy(data)
        num_nodes, (row, col) = data.num_nodes, data.edge_index
        edge_type = data.get('edge_type', torch.zeros_like(row))
        
        y = copy.deepcopy(data.y)
        y[(~data.train_mask).nonzero(as_tuple=True)[0]] = -1

        clss, cnts = y.unique(return_counts=True)
        clss = clss[1:]
        cnts = cnts[1:]
        for cls, cnt in zip(clss, cnts):
            
            node_idx_of_same_class = (y==cls).nonzero(as_tuple=True)[0]
            full = row.new_full((cnt, ), num_nodes)
            row = torch.cat([row, node_idx_of_same_class], dim=0)
            col = torch.cat([col, full], dim=0)

            edge_index = torch.stack([row, col], dim=0)
            new_type = edge_type.new_full((cnt, ), int(edge_type.max()) + 1)
            edge_type = torch.cat([edge_type, new_type, new_type + 1], dim=0)
            num_nodes += 1

        for key, _ in data.items():
            if key == 'x':
                temp_x = torch.zeros((70, 8710)).to(data.x.device)
                data.x = torch.cat([data.x,temp_x], dim=0)
            elif key == 'y':
                temp_y = torch.arange(70).long().to(data.x.device)
                data.y = torch.cat([data.y, temp_y], dim=0)
            elif key == 'train_mask':
                temp_mask = torch.ones((70)).to(torch.bool).to(data.x.device)
                data.train_mask = torch.cat([data.train_mask, temp_mask]) 
            elif key == 'valid_mask':
                temp_mask = torch.ones((70)).to(torch.bool).to(data.x.device)
                data.valid_mask = torch.cat([data.valid_mask, temp_mask])
            elif key == 'test_mask':
                temp_mask = torch.ones((70)).to(torch.bool).to(data.x.device)
                data.test_mask = torch.cat([data.test_mask, temp_mask])
                    
        data.edge_index = edge_index
        data.edge_type = edge_type

        if 'num_nodes' in data:
            data.num_nodes = num_nodes

        return data


class VirtualClassNode_init(torch_geometric.transforms.BaseTransform):
    def __init__(self, init_x="class_mean"):
        super().__init__()
        self.init_x = init_x

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data :
        data = copy.deepcopy(data)
        num_nodes, (row, col) = data.num_nodes, data.edge_index
        edge_type = data.get('edge_type', torch.zeros_like(row))
        
        y = copy.deepcopy(data.y)
        y[(~data.train_mask).nonzero(as_tuple=True)[0]] = -1

        clss, cnts = y.unique(return_counts=True)
        clss = clss[1:]
        cnts = cnts[1:]
        for cls, cnt in zip(clss, cnts):
            
            node_idx_of_same_class = (y==cls).nonzero(as_tuple=True)[0]
            full = row.new_full((cnt, ), num_nodes)
            row = torch.cat([row, node_idx_of_same_class, full], dim=0)
            col = torch.cat([col, full, node_idx_of_same_class], dim=0)

            edge_index = torch.stack([row, col], dim=0)
            new_type = edge_type.new_full((cnt, ), int(edge_type.max()) + 1)
            edge_type = torch.cat([edge_type, new_type, new_type + 1], dim=0)
            num_nodes += 1

        for key, _ in data.items():
            if key == 'x':
                if self.init_x == "zero":
                    vnode_x = torch.zeros((70, 8710)).to(data.x.device)
                elif self.init_x == "class_mean":
                    vnode_x = []
                    for cls, cnt in zip(clss, cnts):
                        temp_x = torch.zeros((1, 8710)).to(data.x.device)
                        temp_x = temp_x + data.x[node_idx_of_same_class].sum(dim=0).unsqueeze(0)
                        temp_x = temp_x / cnt
                        vnode_x.append(temp_x)
                    vnode_x = torch.cat(vnode_x, dim=0).to(data.x.device)
                elif self.init_x == "randomly_selected":
                    vnode_x = []
                    for cls, cnt in zip(clss, cnts):
                        cand_x = torch.unbind(data.x[node_idx_of_same_class], dim=0)
                        choice = torch.randint(0, len(cand_x), (1,)).item()
                        temp_x = cand_x[choice].unsqueeze(0)
                        vnode_x.append(temp_x)
                    vnode_x = torch.cat(vnode_x, dim=0).to(data.x.device)
                data.x = torch.cat([data.x,vnode_x], dim=0)
            elif key == 'y':
                temp_y = torch.arange(70).long().to(data.x.device)
                data.y = torch.cat([data.y, temp_y], dim=0)
            elif key == 'train_mask':
                temp_mask = torch.ones((70)).to(torch.bool).to(data.x.device)
                data.train_mask = torch.cat([data.train_mask, temp_mask]) 
            elif key == 'valid_mask':
                temp_mask = torch.ones((70)).to(torch.bool).to(data.x.device)
                data.valid_mask = torch.cat([data.valid_mask, temp_mask])
            elif key == 'test_mask':
                temp_mask = torch.ones((70)).to(torch.bool).to(data.x.device)
                data.test_mask = torch.cat([data.test_mask, temp_mask])
                
                    
        data.edge_index = edge_index
        data.edge_type = edge_type

        if 'num_nodes' in data:
            data.num_nodes = num_nodes

        return data

class UnidirectionalVirtualClassNode_init(torch_geometric.transforms.BaseTransform):
    def __init__(self, init_x="class_mean"):
        super().__init__()
        self.init_x = init_x

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data :

        data = copy.deepcopy(data)
        num_nodes, (row, col) = data.num_nodes, data.edge_index
        edge_type = data.get('edge_type', torch.zeros_like(row))
        
        y = copy.deepcopy(data.y)
        y[(~data.train_mask).nonzero(as_tuple=True)[0]] = -1

        clss, cnts = y.unique(return_counts=True)
        clss = clss[1:]
        cnts = cnts[1:]
        for cls, cnt in zip(clss, cnts):
            
            node_idx_of_same_class = (y==cls).nonzero(as_tuple=True)[0]
            full = row.new_full((cnt, ), num_nodes)
            row = torch.cat([row, node_idx_of_same_class], dim=0)
            col = torch.cat([col, full], dim=0)

            edge_index = torch.stack([row, col], dim=0)
            new_type = edge_type.new_full((cnt, ), int(edge_type.max()) + 1)
            edge_type = torch.cat([edge_type, new_type], dim=0)
            num_nodes += 1

        for key, _ in data.items():
            if key == 'x':
                if self.init_x == "zero":
                    vnode_x = torch.zeros((70, 8710)).to(data.x.device)
                elif self.init_x == "class_mean":
                    vnode_x = []
                    for cls, cnt in zip(clss, cnts):
                        temp_x = torch.zeros((1, 8710)).to(data.x.device)
                        temp_x = temp_x + data.x[node_idx_of_same_class].sum(dim=0).unsqueeze(0)
                        temp_x = temp_x / cnt
                        vnode_x.append(temp_x)
                    vnode_x = torch.cat(vnode_x, dim=0).to(data.x.device)
                elif self.init_x == "randomly_selected":
                    vnode_x = []
                    for cls, cnt in zip(clss, cnts):
                        cand_x = torch.unbind(data.x[node_idx_of_same_class], dim=0)
                        choice = torch.randint(0, len(cand_x), (1,)).item()
                        temp_x = cand_x[choice].unsqueeze(0)
                        vnode_x.append(temp_x)
                    vnode_x = torch.cat(vnode_x, dim=0).to(data.x.device)
                    data.x = torch.cat([data.x,vnode_x], dim=0)
            elif key == 'y':
                temp_y = torch.arange(70).long().to(data.x.device)
                data.y = torch.cat([data.y, temp_y], dim=0)
            elif key == 'train_mask':
                temp_mask = torch.ones((70)).to(torch.bool).to(data.x.device)
                data.train_mask = torch.cat([data.train_mask, temp_mask]) 
            elif key == 'valid_mask':
                temp_mask = torch.ones((70)).to(torch.bool).to(data.x.device)
                data.valid_mask = torch.cat([data.valid_mask, temp_mask])
            elif key == 'test_mask':
                temp_mask = torch.ones((70)).to(torch.bool).to(data.x.device)
                data.test_mask = torch.cat([data.test_mask, temp_mask])
                    
        data.edge_index = edge_index
        data.edge_type = edge_type

        if 'num_nodes' in data:
            data.num_nodes = num_nodes

        return data