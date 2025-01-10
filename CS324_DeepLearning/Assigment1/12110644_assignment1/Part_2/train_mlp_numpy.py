import argparse
import numpy as np
from mlp_numpy import MLP  
from modules import *

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 15 # adjust if you use batch or not, 19
EVAL_FREQ_DEFAULT = 10
BATCH_SIZE_DEFAULT = 1  # Default to stochastic gradient descent, 10

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.
    
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding
    
    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    # TODO: Implement the accuracy calculation
    # Hint: Use np.argmax to find predicted classes, and compare with the true classes in targets
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1))


def train(dataset, dnn_hidden_units, learning_rate, max_steps, eval_freq, batch_size):
    """
    Performs training and evaluation of MLP model.
    
    Args:
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set
        NOTE: Add necessary arguments such as the data, your model...
    """
    # TODO: Load your data here
    dataset = np.load(dataset)
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']

    # Convert y_train and y_test to one-hot encoding if needed
    y_train = np.eye(len(np.unique(y_train)))[y_train]
    y_test = np.eye(len(np.unique(y_test)))[y_test]
    
    # TODO: Initialize your MLP model and loss function (CrossEntropy) here
    n_inputs = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    n_hidden = list(map(int, dnn_hidden_units.split(',')))
    model = MLP(n_inputs, n_hidden, n_classes)
    loss_fn = CrossEntropy()

    num_samples = X_train.shape[0]
    
    for step in range(max_steps+1):
        # Stochastic Gradient Descent
        if batch_size != 0:
            # Shuffle the data at the beginning of each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                # Forward pass
                predictions = model.forward(X_batch)
                
                # Compute loss
                loss = loss_fn.forward(predictions, y_batch)
                
                # Backward pass (compute gradients)
                dout = loss_fn.backward(predictions, y_batch)
                model.backward(dout)
                
                # Update weights
                for layer in model.layers:
                    if isinstance(layer, Linear):
                        layer.weights -= learning_rate * layer.dweights
                        layer.biases -= learning_rate * layer.dbiases

                it = (step * num_samples + start_idx) // batch_size
                
                if it % eval_freq == 0:
                    test_predictions = model.forward(X_test)
                    test_loss = loss_fn.forward(test_predictions, y_test)
                    test_accuracy = accuracy(test_predictions, y_test)

                    train_predictions = model.forward(X_train)
                    train_loss = loss_fn.forward(train_predictions, y_train)
                    train_accuracy = accuracy(train_predictions, y_train)
                    
                    # test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(y_test, axis=1))
                    print(f"Step: {it}, Train Loss: {train_loss}, Test Loss: {test_loss}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        
        else:
            # TODO: Implement the training loop
            # 1. Forward pass
            # 2. Compute loss
            # 3. Backward pass (compute gradients)
            # 4. Update weights

            # 前向传播
            predictions = model.forward(X_train)
                
            # 计算损失
            loss = loss_fn.forward(predictions, y_train)
                
            # 反向传播（计算梯度）
            dout = loss_fn.backward(predictions, y_train)
            model.backward(dout)
                
            # 更新权重
            for layer in model.layers:
                if isinstance(layer, Linear):
                    layer.weights -= learning_rate * layer.dweights
                    layer.biases -= learning_rate * layer.dbiases
        
            if step % eval_freq == 0 or step == max_steps - 1:
                # TODO: Evaluate the model on the test set
                # 1. Forward pass on the test set
                # 2. Compute loss and accuracy
                test_predictions = model.forward(X_test)
                test_loss = loss_fn.forward(test_predictions, y_test)
                test_accuracy = accuracy(test_predictions, y_test)

                train_predictions = model.forward(X_train)
                train_loss = loss_fn.forward(train_predictions, y_train)
                train_accuracy = accuracy(train_predictions, y_train)
                
                # test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(y_test, axis=1))
                print(f"Step: {step}, Train Loss: {train_loss}, Test Loss: {test_loss}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

    print("Training complete!")

def main():
    """
    Main function.
    """
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='moons_dataset.npz',
                        help='Dataset to use')
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Size of the batch for gradient descent (1 for stochastic gradient descent)')
    
    FLAGS = parser.parse_known_args()[0]
    
    train(FLAGS.dataset, FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq, FLAGS.batch_size)

if __name__ == '__main__':
    main()
