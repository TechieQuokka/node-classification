import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=16):
        super(GCN, self).__init__()
        # GCN 논문(Kipf & Welling)의 2-layer 구조
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # x: 노드 특징 행렬, edge_index: 그래프 연결 구조(COO format)
        
        # 1st Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 2nd Layer
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
