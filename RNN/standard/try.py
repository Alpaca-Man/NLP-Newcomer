import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

seq_len = 60
batch_size = 64
emb_size = 128
num_layers = 1
hidden_size = 256

x = torch.rand(seq_len, batch_size, emb_size)# 输入
state = torch.rand(num_layers, batch_size, hidden_size)# 就是 h0 如果双向的话第一维度要乘上2
RNN = nn.RNN(emb_size, hidden_size, num_layers, nonlinearity='relu' )# 事实上激活函数设置为 Relu 会比较好

y, new_state = RNN(x, state)# 也可以不输入 state

# y: [seq_len, batch_size, hidden_size]
print('y: {}'.format(y.shape))
# new_state: [num_layers * num_directions, batch_size, hidden_size]
print('new_state: {}'.format(new_state.shape))
