import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

def visualize_embeddings(h, color, filename):
    # h: 노드 임베딩 (Tensor)
    # color: 노드 레이블 (Tensor)
    # filename: 저장할 파일 이름
    
    # t-SNE를 사용하여 고차원 임베딩을 2차원으로 차원 축소
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    # 레이블(color)에 따라 점의 색상을 다르게 표시
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color.cpu(), cmap="Set2")
    plt.title(f"Node Embeddings ({filename})")
    
    # 이미지 파일로 저장
    plt.savefig(filename)
    plt.close() # 메모리 해제를 위해 닫아줌
    print(f"시각화 결과가 {filename}에 저장되었습니다.")
