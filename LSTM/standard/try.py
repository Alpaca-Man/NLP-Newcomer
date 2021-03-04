import torch
import torch.nn as nn

seq_len = 60
batch_size = 64
emb_size = 128
num_layers = 2
hidden_size = 256


x = torch.rand(seq_len, batch_size, emb_size)# 输入
hState = torch.rand(num_layers * 2, batch_size, hidden_size)# 就是 h0 如果双向的话第一维度要乘上2
cState = torch.rand(num_layers * 2, batch_size, hidden_size)# 就是 c0 如果双向的话第一维度要乘上2

LSTM = nn.LSTM(emb_size, hidden_size, num_layers, bidirectional=True)

y, (hidden, cell) = LSTM(x, (hState, cState))# 也可以不输入 (hState, cState)

# y: [seq_len, batch_size, hidden_size]
print('y: {}'.format(y.shape))
# hidden: [num_layers * num_directions, batch_size, hidden_size]
print('hidden: {}'.format(hidden.shape))
# cell: [num_layers * num_directions, batch_size, hidden_size]
print('cell: {}'.format(cell.shape))