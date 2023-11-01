import numpy as np
import math
import operator


def distanceFunc(vec1, vec2):
    distance = np.linalg.norm(vec1-vec2) 
    return distance


def computeDistancesNeighbors(K, X_train, y_train, sample):
    distance_and_y = []
    for i in range(len(X_train)):
        x = X_train[i]
        distance_and_y.append((distanceFunc(x, sample), y_train[i]))
    distance_and_y.sort(key=operator.itemgetter(0))
    neighbors = [x[1] for x in distance_and_y[:K]]
    
    return neighbors


def Majority(neighbors):
    predicted_value = 0
    vote = np.sum(neighbors)
    if vote > len(neighbors)/2:
        predicted_value = 1
    
    return predicted_value


def KNN(K, X_train, y_train, X_val):
    predictions = []
    for i in range(len(X_val)):
        neighbors = computeDistancesNeighbors(K, X_train, y_train, X_val[i])
        predictions.append(Majority(neighbors))

    return predictions


if __name__ == '__main__':
    X_train = []
    y_train = []
    N1 = int(input())
    for i in range(N1):
        data = input().split()
        y_train.append(int(data[0]))
        X_train.append([float(x) for x in data[1:]])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    N2 = int(input())
    y_pred = []
    for i in range(N2):
        data = input().split()
        Q = int(data[0]) # classifier
        x = [float(x) for x in data[1:]]
        x = np.array(x)
        neighbors = computeDistancesNeighbors(Q, X_train, y_train, x)
        y_pred.append(Majority(neighbors))

    for y in y_pred:
        print(y)