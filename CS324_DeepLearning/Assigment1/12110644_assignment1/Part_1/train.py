from perceptron import Perceptron
from generate import generate_gaussian
import numpy as np
import argparse

def run():
    # get dataset from dataset.npz
    dataset = np.load('dataset.npz')
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']

    # train the perceptron
    perceptron = Perceptron(n_inputs=2, max_epochs=15, learning_rate=0.1)
    perceptron.train(X_train, y_train)

    # test the perceptron
    correct = 0
    for i in range(len(X_test)):
        if perceptron.forward(X_test[i]) == y_test[i]:
            correct += 1
    accuracy = correct / len(X_test)
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='mean', help='vary mean or cov')

    FLAGS = parser.parse_known_args()[0]

    means_list = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8]]
    cov_list = [[[1, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 3]], [[4, 0], [0, 4]], [[5, 0], [0, 5]], [[6, 0], [0, 6]], [[7, 0], [0, 7]], [[8, 0], [0, 8]]]

    if FLAGS.type == 'mean':
        for mean1 in [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8]]:
            generate_gaussian(mean1=mean1)
            accuracy = run()
            print(f"mean1: {mean1}, Accuracy: {accuracy}")

    else:
        for cov1 in [[[1, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 3]], [[4, 0], [0, 4]], [[5, 0], [0, 5]], [[6, 0], [0, 6]], [[7, 0], [0, 7]], [[8, 0], [0, 8]]]:
            generate_gaussian(cov1=cov1)
            accuracy = run()
            print(f"cov1: {cov1}, Accuracy: {accuracy}")