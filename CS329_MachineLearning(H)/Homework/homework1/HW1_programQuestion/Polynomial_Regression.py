import numpy as np
import itertools
import functools

class PolynomialFeature(object):
    def __init__(self, degree=2):
        assert isinstance(degree, int)
        self.degree = degree

    def transform(self, x):
        if x.ndim == 1:
            x = x[:, None]
        x_t = x.transpose()
        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda x, y: x * y, items))
        return np.asarray(features).transpose()
    
class Regression(object):
    pass
    
class LinearRegression(Regression):
    def fit(self, X:np.ndarray, t:np.ndarray):
        self.w = np.linalg.pinv(X) @ t
        self.var = np.mean(np.square(X @ self.w - t))

    def predict(self, X:np.ndarray, return_std:bool=False):
        y = X @ self.w
        if return_std:
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y, y_std
        return y

if __name__=="__main__":
    # 从标准输入读
    N_M = input().split()
    N = (int)(N_M[0])
    M = (int)(N_M[1])
    X_train = []
    Y_train = []
    X_test = []
    for n in range(N):
        x_y = input().split()
        X_train.append((float)(x_y[0]))
        Y_train.append((float)(x_y[1]))
    for m in range(M):
        x = input().split()
        X_test.append((float)(x[0]))
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    poly_feature_generator = PolynomialFeature(3)
    poly_feature_train = poly_feature_generator.transform(X_train)
    poly_feature_test = poly_feature_generator.transform(X_test)
    lin_rgr = LinearRegression()
    lin_rgr.fit(poly_feature_train, Y_train)
    Y_pred = lin_rgr.predict(poly_feature_test)
    for y in Y_pred:
        print(y)