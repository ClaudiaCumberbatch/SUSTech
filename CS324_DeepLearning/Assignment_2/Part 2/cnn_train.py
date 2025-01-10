from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from cnn_model import CNN
import torch
import torchvision
import torchvision.transforms as transforms

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 500
EVAL_FREQ_DEFAULT = 1
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = '/home/zhousc/deep_learning/Assignment_2/Part 1/data'
PATIENCE_DEFAULT = 10  # 早停的耐心值

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
    accuracy = (np.argmax(predictions, 1) == np.argmax(targets, 1)).mean()
    return accuracy

def train():
    """
    Performs training and evaluation of CNN model.
    NOTE: You should evaluate the model on the whole test set each eval_freq iterations.
    """
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据增强和标准化
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加载 CIFAR-10 数据集
    trainset = torchvision.datasets.CIFAR10(root=FLAGS.data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=FLAGS.data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=2)

    # 定义网络
    net = CNN(3, 10).to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS.learning_rate)

    # 早停参数
    patience = FLAGS.patience
    best_loss = np.inf
    epochs_no_improve = 0

    # 训练网络
    for epoch in range(FLAGS.max_steps):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, data in enumerate(trainloader, 0):
            # 获取输入
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向+后向+优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计训练损失和准确率
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if i % 200 == 199:  # 每 200 mini-batches 打印一次
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

        # 每 eval_freq 评估一次模型
        if epoch % FLAGS.eval_freq == FLAGS.eval_freq - 1:
            net.eval()
            correct_test = 0
            total_test = 0
            test_loss = 0.0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()

            train_accuracy = 100 * correct_train / total_train
            test_accuracy = 100 * correct_test / total_test
            test_loss /= len(testloader)

            print(f'Epoch {epoch + 1}, Train Loss: {running_loss / len(trainloader):.3f}, Train Accuracy: {train_accuracy:.2f}%')
            print(f'Epoch {epoch + 1}, Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.2f}%')

            # 检查是否早停
            if test_loss < best_loss:
                best_loss = test_loss
                epochs_no_improve = 0
                # 保存模型
                torch.save(net.state_dict(), 'best_model.pth')
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print('Early stopping!')
                break

    print('Finished Training')

def main():
    """
    Main function
    """
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--patience', type=int, default=PATIENCE_DEFAULT,
                        help='Patience for early stopping')
    FLAGS, unparsed = parser.parse_known_args()

    main()