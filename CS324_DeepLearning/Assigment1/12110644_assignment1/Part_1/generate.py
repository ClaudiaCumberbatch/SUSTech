import numpy as np
from sklearn.model_selection import train_test_split
from watch import save_fig

def generate_gaussian(mean1 = [0, 0], cov1 = [[1, 0], [0, 1]]):

    mean2 = [5, 5]
    cov2 = [[1, 0], [0, 1]] 

    points1 = np.random.multivariate_normal(mean1, cov1, 100)
    points2 = np.random.multivariate_normal(mean2, cov2, 100)

    points = np.vstack((points1, points2))
    labels = np.array([-1]*100 + [1]*100) 

    X_train, X_test, y_train, y_test = train_test_split(points, labels, test_size=0.2, stratify=labels)

    np.savez('dataset.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    
    save_fig(cov1)
