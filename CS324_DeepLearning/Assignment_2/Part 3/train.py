from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import RMSprop
from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from utils import AverageMeter, accuracy

def train(model, data_loader, optimizer, criterion, device, config):
    model.train()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        # 前向传播
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optimizer.step()

        # 统计损失和准确率
        losses.update(loss.item(), batch_inputs.size(0))
        acc = accuracy(outputs, batch_targets)
        accuracies.update(acc, batch_inputs.size(0))

        # if step % 10 == 0:
        #     print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg

@torch.no_grad()
def evaluate(model, data_loader, criterion, device, config):
    model.eval()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        # 前向传播
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)

        # 统计损失和准确率
        losses.update(loss.item(), batch_inputs.size(0))
        acc = accuracy(outputs, batch_targets)
        accuracies.update(acc, batch_inputs.size(0))

        # if step % 10 == 0:
        #     print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg

def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化模型
    model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes).to(device)

    # 初始化数据集和数据加载器
    dataset = PalindromeDataset(config.input_length, config.data_size)
    train_size = int(config.portion_train * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # 设置损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = RMSprop(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.max_epoch):
        # 训练模型
        train_loss, train_acc = train(model, train_dloader, optimizer, criterion, device, config)

        # 评估模型
        val_loss, val_acc = evaluate(model, val_dloader, criterion, device, config)

        print(f'Epoch [{epoch + 1}/{config.max_epoch}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}')

    # 保存模型
    torch.save(model.state_dict(), 'vanilla_rnn.pth')
    print('Finished Training')

if __name__ == "__main__":
    # 解析训练配置
    parser = argparse.ArgumentParser()

    # 模型参数
    parser.add_argument('--input_length', type=int, default=19, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_epoch', type=int, default=1000, help='Number of epochs to run for')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--data_size', type=int, default=1000000, help='Size of the total dataset')
    parser.add_argument('--portion_train', type=float, default=0.8, help='Portion of the total dataset used for training')

    config = parser.parse_args()
    # 训练模型
    main(config)