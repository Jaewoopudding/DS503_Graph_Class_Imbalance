from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import torch_geometric

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

def test_model(model, data:torch_geometric.data.data.Data):
    model.eval()
    out, embedding = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    report = classification_report(data.y[~data.train_mask.cpu()].cpu(), pred[~data.train_mask.cpu()].cpu(), output_dict=True)
    return report

def save_result(report):
    pass