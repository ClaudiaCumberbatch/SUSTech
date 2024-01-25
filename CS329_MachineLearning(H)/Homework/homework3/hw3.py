import numpy as np

class LogisticRegression():
    def __init__(self, n_features, n_classes, max_epoch, lr) -> None:
        self.w = np.zeros((n_features + 1, n_classes))
        self.max_epoch = max_epoch
        self.lr = lr

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    def softmax(self, z):
        # z = z - z.max()
        return np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)
    
    def accuracy(self, y_pred, y_true):
        return np.sum(y_pred == y_true) / len(y_true)

    def predict(self, X):
        y_pred = np.dot(X, self.w)
        y_pred = self.softmax(y_pred)
        # y_pred_argmax = np.argmax(y_pred, axis=1)
        # y_pred_one_hot = np.eye(y_pred.shape[1])[y_pred_argmax]
        return y_pred

    def fit(self, X, y):
        num_samples, _ = X.shape
        X = np.concatenate([X, np.ones((num_samples, 1))], axis=1)

        for e in range(self.max_epoch):
            y_pred = self.predict(X)
            error = y_pred - y
            grad = np.dot(X.T, error) / num_samples
            self.w -= self.lr * grad
            
            if e % 100 == 0:
                y_pred_class = np.argmax(y_pred, axis=1)
                y_true_class = np.argmax(y, axis=1)
                acc = self.accuracy(y_pred_class, y_true_class)
                # print(f"Epoch {e+1}: Accuracy = {acc:.3f}")

def input_data():
    line = input().split()
    # number of training samples
    N = int(line[0])

    # number of data dimension
    D = int(line[1])

    # number of targets
    C = int(line[2])

    # number of max epoch
    E = int(line[3])

    # learning rate
    L = float(line[4])

    X_train = []
    y_train = []
    for i in range(N):
        line = input().split()
        X_train.append([float(x) for x in line])
    for i in range(N):
        line = input().split()
        y_train.append([int(x) for x in line])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train, D, C, E, L


if __name__ == '__main__':
    X_train, y_train, D, C, E, L = input_data()

    lrc = LogisticRegression(D, C, E, L)
    lrc.fit(X_train, y_train)
    w_reshaped = lrc.w.reshape(-1)
    for weight in w_reshaped:
        print("%.3f" % weight)