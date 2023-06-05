from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import torch_geometric

from loss import ConstrativeLoss

def train_model(model, data:torch_geometric.data.data.Data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out, embedding = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    pred = out.argmax(dim=-1)
    correct = pred[data.train_mask] == data.y[data.train_mask]
    acc = int(correct.sum()) / int(data.train_mask.sum())
    return loss.detach().cpu().numpy(), acc

def train_constrative_model(model, data:torch_geometric.data.data.Data, optimizer, criterion, constrative_coef=2e-3, temperature=0.07):
    CL = ConstrativeLoss(temperature=temperature)
    model.train()
    optimizer.zero_grad()
    out, embedding = model(data.x, data.edge_index)
    constrative_loss = constrative_coef * CL(embedding[-70:])
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) + constrative_loss
    loss.backward()
    optimizer.step()
    pred = out.argmax(dim=-1)
    correct = pred[data.train_mask] == data.y[data.train_mask]
    acc = int(correct.sum()) / int(data.train_mask.sum())
    return loss.detach().cpu().numpy(), acc

def test_model(model, data:torch_geometric.data.data.Data):
    model.eval()
    out, embedding = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    report = classification_report(data.y[~data.train_mask.cpu()].cpu(), pred[~data.train_mask.cpu()].cpu(), output_dict=True)
    return report

def valid_model(model, data:torch_geometric.data.data.Data, criterion, constrative_flag=False ,constrative_coef=2e-3, temperature=0.07):
    CL = ConstrativeLoss(temperature=temperature)
    model.eval()
    out, embedding = model(data.x, data.edge_index)

    loss = criterion(out[data.valid_mask], data.y[data.valid_mask])
    if constrative_flag:
        constrative_loss = constrative_coef * CL(embedding[-70:])
        loss = loss + constrative_loss
    pred = out.argmax(dim=-1)
    correct = pred[data.valid_mask] == data.y[data.valid_mask]
    acc = int(correct.sum()) / int(data.train_mask.sum())
    return loss.detach().cpu().numpy(), acc

def save_result(report):
    pass