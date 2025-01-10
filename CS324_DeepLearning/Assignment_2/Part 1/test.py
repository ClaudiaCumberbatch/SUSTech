from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_mlp import MLP

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '512,256,128'
LEARNING_RATE_DEFAULT = 1e-3
MAX_EPOCHS_DEFAULT = 10
EVAL_FREQ_DEFAULT = 1

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 1D int array of size [number_of_data_samples] with ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    accuracy = (np.argmax(predictions, 1) == targets).sum() / float(targets.shape[0])
    return accuracy

def train():
    """
    Performs training and evaluation of MLP model.
    NOTE: You should evaluate the model on the whole test set each eval_freq iterations.
    """
    # Load CIFAR-10 data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)

    n_inputs = 3 * 32 * 32  # CIFAR-10 图像的输入维度
    n_classes = 10  # CIFAR-10 有 10 个类别

    # Initialize the model that we are going to use
    model = MLP(n_inputs, [int(i) for i in FLAGS.dnn_hidden_units.split(',')], n_classes)
    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    # Training loop
    for epoch in range(FLAGS.max_steps):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入数据
            inputs, labels = data
            inputs = inputs.view(-1, 3 * 32 * 32)  # 展平输入图像

            # 将梯度缓存清零
            optimizer.zero_grad()

            # 前向传播 + 反向传播 + 优化
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            # 打印统计信息
            running_loss += loss.item()
            if i % 200 == 199:    # 每200个小批量打印一次
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

        # 在测试集上评估模型
        if epoch % FLAGS.eval_freq == 0:
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images = images.view(-1, 3 * 32 * 32)  # 展平输入图像
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

def main():
    """
    Main function
    """
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()