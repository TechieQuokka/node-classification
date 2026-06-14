import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from model import GCN
from visualize import visualize_embeddings

def train():
    # 1. 데이터셋 로드 (Cora)
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    # 2. 모델 초기화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(
        num_node_features=dataset.num_node_features,
        num_classes=dataset.num_classes
    ).to(device)
    data = data.to(device)
    
    # [시각화 추가] 학습 전 초기 임베딩 상태 확인
    model.eval()
    out = model(data.x, data.edge_index)
    visualize_embeddings(out, color=data.y, filename="embeddings_before.png")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # 3. 학습 루프
    model.train()
    for epoch in range(201):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')

    # 4. 평가 및 결과 시각화
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')
    
    visualize_embeddings(out, color=data.y, filename="embeddings_after.png")

if __name__ == "__main__":
    train()

