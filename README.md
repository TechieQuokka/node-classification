# GNN Node Classification: GCN on Cora

This repository provides a clean and modular implementation of **Graph Convolutional Networks (GCN)** for the node classification task using the **Cora** dataset. This project is designed for both academic research and self-learning purposes.

## Overview
The goal is to predict the category of scientific papers (nodes) in a citation network based on:
1.  **Node Features**: Sparse bag-of-words representation of papers.
2.  **Graph Structure**: Citation links between papers.

## Training Process & Results
The model was trained for **200 epochs** using the Adam optimizer. Below is the recorded training process showing the loss reduction and final accuracy.

### Execution Log
```text
학습 전 초기 임베딩 시각화 중...
시각화 결과가 embeddings_before.png에 저장되었습니다.
Epoch 000, Loss: 1.9455
Epoch 020, Loss: 0.2433
Epoch 040, Loss: 0.0522
Epoch 060, Loss: 0.0401
Epoch 080, Loss: 0.0356
Epoch 100, Loss: 0.0396
Epoch 120, Loss: 0.0426
Epoch 140, Loss: 0.0455
Epoch 160, Loss: 0.0296
Epoch 180, Loss: 0.0304
Epoch 200, Loss: 0.0317
Accuracy: 0.8030
시각화 결과가 embeddings_after.png에 저장되었습니다.
```

### Final Performance
*   **Test Accuracy**: **80.30%**
*   **Analysis**: The model converged rapidly around Epoch 60. The final accuracy of 80.3% aligns with the benchmarks of the original GCN paper (81.5%), proving the effectiveness of the Graph Convolutional operator on semi-supervised tasks.

## Visualization (t-SNE)
The node embeddings are projected into 2D space using t-SNE to observe how the model learns to cluster different classes.

### 1. Before Training (`embeddings_before.png`)
Initially, node embeddings are randomly distributed because the weights are initialized randomly.
![Embeddings Before](embeddings_before.png)

### 2. After Training (`embeddings_after.png`)
After 200 epochs, nodes belonging to the same category are physically clustered together, demonstrating that the GCN successfully captured both semantic features and structural connectivity.
![Embeddings After](embeddings_after.png)

## Key Theoretical Concepts
*   **Message Passing**: Aggregating information from 1-hop neighborhood nodes to update the target node representation.
*   **Semi-supervised Learning**: Training the model using only a small fraction of labeled nodes (140 nodes for Cora).
*   **Spectral Graph Convolution**: Utilizing the Graph Laplacian to perform convolutions in the spectral domain.

## Project Structure
*   `model.py`: Definition of the 2-layer GCN architecture.
*   `train.py`: Main script for data loading, training loop, and evaluation.
*   `visualize.py`: Utility to project high-dimensional embeddings into 2D space using **t-SNE**.
*   `requirements.txt`: List of dependencies.

## Getting Started
### Installation
```bash
pip install -r requirements.txt
```

### Running the Project
```bash
python train.py
```
