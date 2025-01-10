import numpy as np
import matplotlib.pyplot as plt
import datetime

def save_fig(name):
    # 从文件中加载数据集
    data = np.load('dataset.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    # 可视化训练集
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', label='Distribution 1 (Train)')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='Distribution 2 (Train)')

    # 可视化测试集
    plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='cyan', label='Distribution 1 (Test)', marker='x')
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='magenta', label='Distribution 2 (Test)', marker='x')

    plt.title('Visualization of Generated Points')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.savefig(f'distribution_{name}.png')
    # plt.show()