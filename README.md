# DS503_Graph_Class_Imbalance
## Models

Message Passing Neural Networks 

- GraphSAGE (WL Hamilton et. al., Inductive Representation Learning on Large Graphs., NIPS2017)

- Graph Attention Network (P Veličković et. al. Graph Attention Networks., ICLR2018)

- GIN (Keyulu Xu et. al. How Powerful are Graph Neural Networks?., CVPR2019)


## User Guide

Enviroment

```
conda create -n ds503 python=3.9
python3 -m pip install -r requirements.txt
```


Detailed Implementation of each model can be found at *models.py*

Experiment has been done with [experiments_optuna.ipynb](https://github.com/Jaewoopudding/DS503_Graph_Class_Imbalance/blob/main/experiments_optuna.ipynb)


Experiments_optuna.ipynb contains end to end model train and inference

```
sage = GraphSAGE(in_channels=dataset.num_features, hidden_channels=256, number_of_classes=dataset.num_classes, num_of_hidden_layers=4, device=device)
gat = GAT(in_channels=dataset.num_features, hidden_channels=476, number_of_classes=dataset.num_classes, num_of_hidden_layers=4, device=device, heads=1)
gin = GIN(in_channels=dataset.num_features, hidden_channels=415, number_of_classes=dataset.num_classes, num_of_hidden_layers=4, device=device)
```


