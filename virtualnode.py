import copy

import torch
import torch_geometric


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
                print(data.y)
                data.y = torch.cat([data.y, temp_y], dim=0)
            elif key == 'train_mask':
                temp_mask = torch.ones((70)).to(torch.bool).to(data.x.device)
                data.train_mask = torch.cat([data.train_mask, temp_mask])
                    
        data.edge_index = edge_index
        data.edge_type = edge_type

        if 'num_nodes' in data:
            data.num_nodes = num_nodes

        return data