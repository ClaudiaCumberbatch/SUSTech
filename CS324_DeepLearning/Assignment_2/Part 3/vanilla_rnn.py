from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class VanillaRNN(nn.Module):

    def __init__(self, input_length, input_dim, hidden_dim, output_dim):
        super(VanillaRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_length = input_length

        # 初始化权重
        self.Wxh = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.Whh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.Why = nn.Parameter(torch.randn(hidden_dim, output_dim))
        self.bh = nn.Parameter(torch.zeros(hidden_dim))
        self.by = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        for t in range(self.input_length):
            xt = x[:, t, :]
            h = torch.tanh(xt @ self.Wxh + h @ self.Whh + self.bh)

        out = h @ self.Why + self.by
        return out