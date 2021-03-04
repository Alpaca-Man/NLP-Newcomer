import torch
import torch.nn as nn

'''
@doc: LSTM 模型细节
@author: Alpaca-Man
@date: 2021/2/16
'''

# LSTM 默认参数
num_layers = 1 # 层数
dropout = 0.00 # 失活率
bidirectional = False # LSTM 是否双向

hState = None # h0
cState = None # c0

class LSTM(nn.Module):
	def __init__(self,vocab_size, emb_size,hidden_size, num_classes,num_layers=num_layers, dropout=dropout, bidirectional=bidirectional):
		'''
		@doc: LSTM 模型细节
        @author: Alpaca-Man
        @date: 2021/2/16
        @param: {
                    vocab_size: 单词个数
                    emb_size: 词嵌入维度
                    hidden_size: 隐藏层维度
                    num_classes: 标签种类数量
                    num_layers: LSTM 层数
                    dropout: 失活率
                    bidirectional: LSTM 是否双向
                }
        @return: { }
		'''
		super(LSTM, self).__init__()
		self.emb = nn.Embedding(vocab_size, emb_size)
		self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
		# x => [batch_size, num_classes]
		self.fc = nn.Linear(hidden_size,num_classes)

	def forward(self, x, hState=hState, cState=cState):
		'''
		@doc: 前向传播函数
        @author: Alpaca-Man
        @date: 2021/2/16
        @param: {
                    x: 输入 [batch_size, seq_len]
                    hState: 隐状态初始值(即每一层的 h0) [num_layers, batch_size, hidden_size] or None
                    cState: 隐状态初始值(即每一层的 c0) [num_layers, batch_size, hidden_size] or None
                }
        @return: {
        		pred: 预测值 [batch_size, num_classes]
        		}
		'''
		batch_size = x.shape[0]
		x = self.emb(x)# x: [batch_size, seq_len]
		x = x.transpose(0, 1)# 维度转换 x: [seq_len, batch_size, emb_size]
		# out: [seq_len, batch_size, hidden_size]
		# hidden: [num_directions(=1) * num_layers, batch_size, hidden_size]
		# cell: [num_directions(=1) * num_layers, batch_size, hidden_size]
		if (hState, cState) == (None, None):
			out, (hidden, cell) = self.lstm(x, None)
		else:
			out, (hidden, cell) = self.lstm(x, (hState, cState))
		out = out[-1]# out: [batch_size, num_directions(=1) * hidden_size]
		pred = self.fc(out)# pred: [batch_size, num_classes]
		return pred