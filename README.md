# DS503_Graph_Class_Imbalance
## Models

Message Passing Neural Networks 

- GraphSAGE (WL Hamilton et. al., Inductive Representation Learning on Large Graphs., NIPS2017)

- Graph Attention Network (P Veličković et. al. Graph Attention Networks., ICLR2018)

- GIN (Keyulu Xu et. al. How Powerful are Graph Neural Networks?., CVPR2019)

Detailed Implementation of each model can be found at *models.py*

Experiment has been done with gps.ipynb and message_passing_models.ipynb

```
sage = GraphSAGE(in_channels=dataset.num_features, hidden_channels=256, number_of_classes=dataset.num_classes, num_of_hidden_layers=4, device=device)
gat = GAT(in_channels=dataset.num_features, hidden_channels=476, number_of_classes=dataset.num_classes, num_of_hidden_layers=4, device=device, heads=1)
gin = GIN(in_channels=dataset.num_features, hidden_channels=415, number_of_classes=dataset.num_classes, num_of_hidden_layers=4, device=device)
```


## Graph Transformers
> 1. [Do Transformers Really Perform Bad for Graph Representation?](https://arxiv.org/abs/2106.05234) NeurIPS 2021. [CODE](https://github.com/microsoft/Graphormer)
> 2. [Pure Transformers are Powerful Graph Learners.](https://arxiv.org/abs/2207.02505) NeurIPS 2022 [CODE](https://github.com/jw9730/tokengt)
> 3. [Structure-Aware Transformer for Graph Representation Learning.](https://arxiv.org/abs/2202.03036) ICML 2022 [CODE](https://github.com/BorgwardtLab/SAT)
> 4. [NodeFormer: A Graph Transformer for Node-Level Prediction.](https://openreview.net/pdf?id=sMezXGG5So) NIPS2022 [CODE](https://github.com/qitianwu/NodeFormer)
