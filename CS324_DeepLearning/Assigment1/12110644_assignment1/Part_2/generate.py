import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# 使用 make_moons 方法生成数据集
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 保存生成的数据集
np.savez('moons_dataset.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

# 可视化生成的数据集
# plt.figure(figsize=(8, 6))
# plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
# plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')
# plt.title('Generated Moons Dataset')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.legend()
# plt.show()