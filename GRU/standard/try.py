import torch
import torch.nn as nn

seq_len = 60
batch_size = 64
emb_size = 128
num_layers = 1
hidden_size = 256

'''
	nn.GRU(input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False)
	{
		input_size: emb_size
		hidden_size: 输出的最后维度
		num_layers: 层数
		bias: 偏置
		batch_first: 输入的矩阵第一维是否是 batch,若为 True 则返回的 Y 第一维也是 batch
		dropout: 失活率
		bidirectional: 是否双向,若为 True 则输入输出 H 的维度为 num_layers * 2
	}
	input
	{
		X: [seq_len, batch_size, emb_size]
		H: [num_layers * num_directions, batch_size, hidden_size] or None
	}
	output:
	{
		Y: [seq_len, batch_size, hidden_size]
		H: [num_layers * num_directions, batch_size, hidden_size]
	}
'''
x = torch.rand(seq_len, batch_size, emb_size)# 输入
h = torch.rand(num_layers, batch_size, hidden_size)# 就是 h0 如果双向的话第一维度要乘上2

GRU = nn.GRU(emb_size, hidden_size, num_layers)

y, hidden = GRU(x, h)# 也可以不输入 h

# y: [seq_len, batch_size, hidden_size]
print('y: {}'.format(y.shape))
# hidden: [num_layers * num_directions, batch_size, hidden_size]
print('hidden: {}'.format(hidden.shape))
