# best_shd_shd = []
# best_shd_tpr = []
# best_tpr_shd = []
# best_tpr_tpr = []
# test = False
# GCN = True
# GNN = True
# tau_A = 0.001
# import numpy as np
#
# if GCN:
#     if tau_A == 0:
#         best_shd_shd = [28, 36, 31]
#         best_shd_tpr = [0.3, 0.3, 0.15]
#
#         best_tpr_shd = [35, 36, 31]
#         best_tpr_tpr = [0.35, 0.3, 0.15]
#
#         # --- 輸出平均值 ---
#         print("\n---GCN tau_A == 0 平均結果 ---")
#         print(f"best_shd_shd mean: {np.mean(best_shd_shd):.2f}")
#         print(f"best_shd_tpr mean: {np.mean(best_shd_tpr):.2f}")
#         print(f"best_tpr_shd mean: {np.mean(best_tpr_shd):.2f}")
#         print(f"best_tpr_tpr mean: {np.mean(best_tpr_tpr):.2f}")
#         # --------------------
#
#     elif tau_A == 0.001:
#         best_shd_shd = [26, 24, 20]
#         best_shd_tpr = [0.3, 0.1, 0.0]
#
#         best_tpr_shd = [36, 24, 36]
#         best_tpr_tpr = [0.7,0.1,0.4]
#
#         # --- 輸出平均值 ---
#         print("\n---GCN tau_A == 0.001 基準平均結果 ---")
#         print(f"best_shd_shd mean: {np.mean(best_shd_shd):.2f}")
#         print(f"best_shd_tpr mean: {np.mean(best_shd_tpr):.2f}")
#         print(f"best_tpr_shd mean: {np.mean(best_tpr_shd):.2f}")
#         print(f"best_tpr_tpr mean: {np.mean(best_tpr_tpr):.2f}")
#         # --------------------
#
#         if test:
#             best_shd_shd = [12, 17]
#             best_shd_tpr = [0.5, 0.1]
#
#             best_tpr_shd = [36, 33]
#             best_tpr_tpr = [0.7, 0.5]
#
#             # --- 輸出平均值 ---
#             print("\n---GCN tau_A == 0.001 (test) 平均結果 ---")
#             print(f"best_shd_shd mean: {np.mean(best_shd_shd):.2f}")
#             print(f"best_shd_tpr mean: {np.mean(best_shd_tpr):.2f}")
#             print(f"best_tpr_shd mean: {np.mean(best_tpr_shd):.2f}")
#             print(f"best_tpr_tpr mean: {np.mean(best_tpr_tpr):.2f}")
#             # --------------------
# if GNN:
#     if tau_A == 0:
#         best_shd_shd = [20,20,20]
#         best_shd_tpr = [0.0,0.05,0.05]
#
#         best_tpr_shd = [21,20,22]
#         best_tpr_tpr = [0.05,0.05,0.05]
#
#         # --- 輸出平均值 ---
#         print("\n--- GNN tau_A == 0 平均結果 ---")
#         print(f"best_shd_shd mean: {np.mean(best_shd_shd):.2f}")
#         print(f"best_shd_tpr mean: {np.mean(best_shd_tpr):.2f}")
#         print(f"best_tpr_shd mean: {np.mean(best_tpr_shd):.2f}")
#         print(f"best_tpr_tpr mean: {np.mean(best_tpr_tpr):.2f}")
#         # --------------------
#
# tpr_list = [0.0, 0.05, 0.2, 0.1, 0.2]
# shd_list = [20, 20, 26, 31, 23]
# print(f"tpr = {sum(tpr_list)/5}")
# print(f"shd = {sum(shd_list)/5}")
#
# 檔案: results.py
#
import matplotlib.pyplot as plt


def save_diagnostic_plot(
        log_epoch,
        log_A_magnitude,
        log_A_grad_magnitude,
        log_loss_recon,
        log_loss_dag_h_A,
        log_loss_sparse,
        filename="diagnostic_plot.png"
):
    """
    接收訓練日誌列表，並將它們繪製成診斷圖表並儲存。

    參數:
    log_epoch (list): Epoch 或 AL 步驟的列表。
    log_A_magnitude (list): A 矩陣平均大小的日誌。
    log_A_grad_magnitude (list): A 梯度平均大小的日誌。
    log_loss_recon (list): 重建損失的日誌。
    log_loss_dag_h_A (list): h_A (DAG 損失) 的日誌。
    log_loss_sparse (list): 稀疏損失的日誌。
    filename (str): 要儲存的圖片檔案名稱。
    """

    print(f"訓練結束，正在產生診斷圖表並儲存至 {filename}...")

    # 創建一個 3x1 的圖
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    # --- 圖 1: A 矩陣的現象 (大小 vs 梯度) ---
    ax1.set_title("A Matrix Diagnosis (Phenomenon vs. Cause)")
    ax1.plot(log_epoch, log_A_magnitude, 'b-o', label='A Magnitude (avg_A_mag)')
    ax1.plot(log_epoch, log_A_grad_magnitude, 'r--x', label='A Gradient (avg_A_grad_mag)')
    ax1.set_ylabel("Magnitude (Log Scale)")
    # [重要] 使用對數尺度才能看清微小的梯度
    # 我們加一個 try-except 以防止值為 0 或負數時出錯
    try:
        ax1.set_yscale('log')
    except ValueError:
        print("Warning: 無法設定 Y 軸為 log scale (可能因為值為 0)。")
        pass
    ax1.legend()
    ax1.grid(True)

    # --- 圖 2: 拔河比賽的「拉力」 (L_Recon vs h_A) ---
    ax2.set_title("Pulling Forces (Losses pulling A away from 0)")
    ax2.plot(log_epoch, log_loss_recon, 'g-o', label='L_Recon (Reconstruction)')
    ax2.plot(log_epoch, log_loss_dag_h_A, 'k--x', label='h_A (DAG Constraint)')
    ax2.set_ylabel("Loss Value")
    ax2.legend()
    ax2.grid(True)

    # --- 圖 3: 拔河比賽的「推力」 (L_Sparse) ---
    ax3.set_title("Pushing Force (Loss pushing A toward 0)")
    ax3.plot(log_epoch, log_loss_sparse, 'm-o', label='L_Sparse (L1 + L2-trace)')
    ax3.set_xlabel("Augmented Lagrangian Step (Epoch)")
    ax3.set_ylabel("Loss Value")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(filename)  # 儲存圖表
    print(f"診斷圖表已成功儲存！")
    plt.show() # 如果你想在腳本運行時直接顯示圖表