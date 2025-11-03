import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1️⃣ 載入資料
data = pd.read_csv("data\sachs_cd3cd28.csv")  # 替換成你的檔名

# 2️⃣ 選擇要觀察的變數
col = "PIP3"
values = data[col].dropna()

# 3️⃣ 計算平均值和標準差
mean = np.mean(values)
std = np.std(values)
skew_pd = values.skew()

print(f"{col} → 平均值: {mean:.4f}, 標準差: {std:.4f}")
print(f"{col} → 偏態係數: {skew_pd:.4f}")

# 4️⃣ 畫直方圖 + 常態分布曲線
plt.figure(figsize=(8,5))
plt.hist(values, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)

# 疊加常態分布
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, std)
plt.plot(x, p, 'r', linewidth=2, label='Normal fit')

plt.title(f'Distribution of {col}')
plt.xlabel(col)
plt.ylabel('Density')
plt.legend()
plt.show()

# 5️⃣ 畫箱型圖
plt.figure(figsize=(8,2))
plt.boxplot(values, vert=False)
plt.title(f'Boxplot of {col}')
plt.xlabel(col)
plt.show()

# 6️⃣ 可選：log 轉換後的直方圖
values_log = np.log1p(values)  # log(x+1) 避免 log(0)
mean_log = np.mean(values_log)
std_log = np.std(values_log)

plt.figure(figsize=(8,5))
plt.hist(values_log, bins=30, density=True, color='lightgreen', edgecolor='black', alpha=0.7)

# 疊加常態分布
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_log, std_log)
plt.plot(x, p, 'r', linewidth=2, label='Normal fit (log-transformed)')
plt.title(f'Distribution of log1p({col})')
plt.xlabel(f'log1p({col})')
plt.ylabel('Density')
plt.legend()
plt.show()
