from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
from pytorch_mlp import MLP

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """

    accuracy = (np.argmax(predictions, 1) == np.argmax(targets, 1)).sum() / float(targets.shape[0])
    return accuracy

def train(dataset):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should evaluate the model on the whole test set each eval_freq iterations.
    """
    # Load data
    dataset = np.load(dataset)
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']

    # Convert y_train and y_test to one-hot encoding if needed
    y_train = np.eye(len(np.unique(y_train)))[y_train]
    y_test = np.eye(len(np.unique(y_test)))[y_test]
    
    n_inputs = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    # Initialize the model that we are going to use
    model = MLP(n_inputs, [int(i) for i in FLAGS.dnn_hidden_units.split(',')], n_classes)
    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    # Training loop
    for step in range(FLAGS.max_steps):
        # Convert data to PyTorch tensors
        inputs = torch.from_numpy(X_train).float()
        targets = torch.from_numpy(np.argmax(y_train, axis=1)).long()
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Compute the loss
        loss = F.cross_entropy(outputs, targets)
        # Backward pass
        loss.backward()
        # Optimize
        optimizer.step()
        # Print some statistics
        if step % FLAGS.eval_freq == 0:
            # print('Step: %d, Loss: %f' % (step, loss.item()))
            # Compute the accuracy on the test set
            test_inputs = torch.from_numpy(X_test).float()
            test_outputs = model(test_inputs)
            test_outputs = test_outputs.detach().numpy()
            test_accuracy = accuracy(test_outputs, y_test)
            test_loss = F.cross_entropy(torch.from_numpy(test_outputs), torch.from_numpy(np.argmax(y_test, axis=1)).long()).item()
            train_accuracy = accuracy(outputs.detach().numpy(), y_train)
            print(f"Step: {step}, Train Loss: {loss.item()}, Test Loss: {test_loss}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

            # print('Test accuracy: %f' % test_accuracy)

def main(dataset):
    """
    Main function
    """
    train(dataset)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='moons_dataset.npz',
                        help='Dataset to use')
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS.dataset)