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