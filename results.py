best_shd_shd = []
best_shd_tpr = []
best_tpr_shd = []
best_tpr_tpr = []
test = False
GCN = True
GNN = True
tau_A = 0.001
import numpy as np

if GCN:
    if tau_A == 0:
        best_shd_shd = [28, 36, 31]
        best_shd_tpr = [0.3, 0.3, 0.15]

        best_tpr_shd = [35, 36, 31]
        best_tpr_tpr = [0.35, 0.3, 0.15]

        # --- 輸出平均值 ---
        print("\n---GCN tau_A == 0 平均結果 ---")
        print(f"best_shd_shd mean: {np.mean(best_shd_shd):.2f}")
        print(f"best_shd_tpr mean: {np.mean(best_shd_tpr):.2f}")
        print(f"best_tpr_shd mean: {np.mean(best_tpr_shd):.2f}")
        print(f"best_tpr_tpr mean: {np.mean(best_tpr_tpr):.2f}")
        # --------------------

    elif tau_A == 0.001:
        best_shd_shd = [26, 24, 20]
        best_shd_tpr = [0.3, 0.1, 0.0]

        best_tpr_shd = [36, 24, 36]
        best_tpr_tpr = [0.7,0.1,0.4]

        # --- 輸出平均值 ---
        print("\n---GCN tau_A == 0.001 基準平均結果 ---")
        print(f"best_shd_shd mean: {np.mean(best_shd_shd):.2f}")
        print(f"best_shd_tpr mean: {np.mean(best_shd_tpr):.2f}")
        print(f"best_tpr_shd mean: {np.mean(best_tpr_shd):.2f}")
        print(f"best_tpr_tpr mean: {np.mean(best_tpr_tpr):.2f}")
        # --------------------

        if test:
            best_shd_shd = [12, 17]
            best_shd_tpr = [0.5, 0.1]

            best_tpr_shd = [36, 33]
            best_tpr_tpr = [0.7, 0.5]

            # --- 輸出平均值 ---
            print("\n---GCN tau_A == 0.001 (test) 平均結果 ---")
            print(f"best_shd_shd mean: {np.mean(best_shd_shd):.2f}")
            print(f"best_shd_tpr mean: {np.mean(best_shd_tpr):.2f}")
            print(f"best_tpr_shd mean: {np.mean(best_tpr_shd):.2f}")
            print(f"best_tpr_tpr mean: {np.mean(best_tpr_tpr):.2f}")
            # --------------------
if GNN:
    if tau_A == 0:
        best_shd_shd = [20,20,20]
        best_shd_tpr = [0.0,0.05,0.05]

        best_tpr_shd = [21,20,22]
        best_tpr_tpr = [0.05,0.05,0.05]

        # --- 輸出平均值 ---
        print("\n--- GNN tau_A == 0 平均結果 ---")
        print(f"best_shd_shd mean: {np.mean(best_shd_shd):.2f}")
        print(f"best_shd_tpr mean: {np.mean(best_shd_tpr):.2f}")
        print(f"best_tpr_shd mean: {np.mean(best_tpr_shd):.2f}")
        print(f"best_tpr_tpr mean: {np.mean(best_tpr_tpr):.2f}")
        # --------------------

tpr_list = [0.0, 0.05, 0.2, 0.1, 0.2]
shd_list = [20, 20, 26, 31, 23]
print(f"tpr = {sum(tpr_list)/5}")
print(f"shd = {sum(shd_list)/5}")

import networkx as nx
import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.linalg as slin
import scipy.sparse as sp
import networkx as nx
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import os
import matplotlib.pyplot as plt
_DIR = os.path.dirname(os.path.abspath(__file__))
_REAL_WORLD_DIR = os.path.join(_DIR, "data/")
df = pd.read_csv(os.path.join(_REAL_WORLD_DIR, "sachs_cd3cd28.csv"))
import networkx as nx
import matplotlib.pyplot as plt

# label 對應表
label_mapping = {
    0: "Raf",
    1: "Mek",
    2: "Plcg",
    3: "PIP2",
    4: "PIP3",
    5: "Erk",
    6: "Akt",
    7: "PKA",
    8: "PKC",
    9: "P38",
    10: "Jnk",
}

# ground truth graph
graph = {
    7: [10, 9, 6, 5, 1, 0, 4],
    8: [10, 9, 1, 0],
    4: [6, 3, 2, 7],
    1: [5],
    0: [1],
    3: [8],
    2: [8, 3],
}

# 建立有向圖
G = nx.DiGraph(graph)
G.add_nodes_from(range(11))

# 節點名稱
labels = {node: label_mapping[node] for node in G.nodes()}

# circular layout
pos = nx.circular_layout(G)

# 畫圖
plt.figure(figsize=(8, 6))

# 畫節點（白底黑框）
nx.draw_networkx_nodes(
    G, pos,
    node_color='white',
    edgecolors='black',
    node_size=900,
    linewidths=1.5
)

# 畫邊
nx.draw_networkx_edges(
    G, pos,
    edge_color='gray',
    arrows=True,
    arrowsize=20
)

# 畫節點標籤
nx.draw_networkx_labels(
    G, pos,
    labels=labels,
    font_size=10,
    font_color='black'
)

plt.title("Ground Truth DAG (Circular Layout)", fontsize=13)
plt.axis('off')
plt.show()
