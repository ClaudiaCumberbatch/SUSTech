
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):
    def __init__(self, seq_length, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        # LSTM weights
        self.W_i = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_i = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.Tensor(hidden_dim))
        
        self.W_f = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_f = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.Tensor(hidden_dim))
        
        self.W_c = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_c = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_c = nn.Parameter(torch.Tensor(hidden_dim))
        
        self.W_o = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_o = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.Tensor(hidden_dim))
        
        # Output layer weights
        self.W_out = nn.Parameter(torch.Tensor(hidden_dim, output_dim))
        self.b_out = nn.Parameter(torch.Tensor(output_dim))
        
        # Add dropout
        self.dropout = nn.Dropout(0.5)
        
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'W_' in name or 'U_' in name:
                nn.init.xavier_uniform_(param)
            elif 'b_f' in name:  # Initialize forget gate bias to 1
                nn.init.constant_(param, 1.0)
            elif 'b_' in name:  # Initialize other biases to 0
                nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        batch_size = x.size(0)
        h_t, c_t = (torch.zeros(batch_size, self.hidden_dim).to(x.device),
                    torch.zeros(batch_size, self.hidden_dim).to(x.device))
        
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            
            i_t = torch.sigmoid(x_t @ self.W_i + h_t @ self.U_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_f + h_t @ self.U_f + self.b_f)
            g_t = torch.tanh(x_t @ self.W_c + h_t @ self.U_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.W_o + h_t @ self.U_o + self.b_o)
            
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            
            # Apply dropout to hidden state
            h_t = self.dropout(h_t)
        
        y = h_t @ self.W_out + self.b_out
        return y
