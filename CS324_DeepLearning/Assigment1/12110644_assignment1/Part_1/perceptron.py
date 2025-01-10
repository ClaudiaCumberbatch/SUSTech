import numpy as np

class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=10, learning_rate=0.1):
        """
        Initializes the perceptron object.
        - n_inputs: Number of inputs.
        - max_epochs: Maximum number of training cycles.
        - learning_rate: Magnitude of weight changes at each training cycle.
        - weights: Initialize weights (including bias).
        """
        self.n_inputs = n_inputs  # Fill in: Initialize number of inputs
        self.max_epochs = max_epochs  # Fill in: Initialize maximum number of epochs
        self.learning_rate = learning_rate  # Fill in: Initialize learning rate
        # Fill in: Initialize weights with zeros 
        self.weights = np.zeros(n_inputs + 1)  # +1 for bias
        
    def forward(self, input_vec):
        """
        Predicts label from input.
        Args:
            input_vec (np.ndarray): Input array of training data, input vec must be all samples
        Returns:
            int: Predicted label (1 or -1) or Predicted lables.
        """
        return np.sign(np.dot(input_vec, self.weights[1:]) + self.weights[0])
        
    def train(self, training_inputs, labels):
        """
        Trains the perceptron.
        Args:
            training_inputs (list of np.ndarray): List of numpy arrays of training points.
            labels (np.ndarray): Array of expected output values for the corresponding point in training_inputs.
        """
        # we need max_epochs to train our model
        for _ in range(self.max_epochs): 
            """
                What we should do in one epoch ? 
                you are required to write code for 
                1.do forward pass
                2.calculate the error
                3.compute parameters' gradient 
                4.Using gradient descent method to update parameters(not Stochastic gradient descent!,
                please follow the algorithm procedure in "perceptron_tutorial.pdf".)
            """
            predictions = self.forward(training_inputs)  # Get predictions for all samples
            errors = labels - predictions  # Compute error for each sample

            for i in range(len(training_inputs)):  # Update weights for each sample based on error
                if errors[i] != 0:  # Update only if there is an error
                    self.weights[1:] += self.learning_rate * errors[i] * training_inputs[i]
                    self.weights[0] += self.learning_rate * errors[i]  # Update bias
