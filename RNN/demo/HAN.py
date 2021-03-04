import torch
import torch.nn as nn
from torch.autograd import Variable

'''
@doc: HAN
@author: Alpaca-Man
@date: 2021/2/16
'''

# BiGRU 默认参数
embSize = 200  # 词向量维度
hiddenSize = 50  # 单向 GRU 隐藏层维度
numLayers = 1  # 双向 GRU 层数
bias = True  # 偏置
batchFirst = True  # 输入的第一维度是否是 batch
dropout = 0  # 失活率
bidirectional = True  # BiGRU
contextSize = 100  # 单词/句子向量维度


class Attention(nn.Module):
	def __init__(self, contextSize=contextSize):
		'''
		@doc: 用于计算 attention 的类， 返回 [batchSize, seqLen or 2 * hiddenSize, 1]
		@author: Alpaca-Man
		@date: 2021/2/26
		@param: {
		        	contextSize:  word/sentence vector 维度 default 100
		        }
		@return: { }
		'''

		super(Attention, self).__init__()
		# self.vector = nn.Parameter(torch.rand(contextSize, 1),requires_grad=True)
		self.vector = nn.Linear(contextSize, 1)  # 网上比较推荐上面这种
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, u):
		# u: [batchSize, seqLen or 2 * hiddenSize, contextSize]
		# temp = torch.matmul(u, self.vector) # temp: [batchSize, seqLen or 2 * hiddenSize, 1]
		temp = self.vector(u)  # temp: [batchSize, seqLen or 2 * hiddenSize, 1]
		attention = self.softmax(temp)  # attention: [batchSize, seqLen or 2 * hiddenSize, 1]
		return attention


class HAN(nn.Module):
	def __init__(self, vocabSize, numClasses, embSize=embSize, hiddenSize=hiddenSize, numLayers=numLayers, bias=bias,
				 batchFirst=batchFirst, dropout=dropout, bidirectional=bidirectional, contextSize=contextSize):
		'''
		@doc: HAN
		@author: Alpaca-Man
		@date: 2021/2/26
		@param: {
		            vocabSize: 单词表长度
		            numClasses: 标签种类数量
		            embSize: 词向量维度 default 200
		            hiddenSize: 单向 GRU 隐藏层维度 default 50
		            numLayers: 双向 GRU 层数 default 1
		            bias: 偏置 default True
		            batchFirst: 输入序列的第一维度是不是批次 default True
		            dropout: 失活率 default 0
		            bidirectional: 双向 GRU default True
		            contextSize: word/sentence vector 维度 default 100
		        }
		@return: { }
		'''

		super(HAN, self).__init__()
		# word/sentence vector
		self.contextSize = contextSize

		# Word Encoder
		self.emb = nn.Embedding(vocabSize, embSize)
		self.wordBiGRU = nn.GRU(embSize, hiddenSize, numLayers, bias, batchFirst, dropout, bidirectional)
		self.wordLinear = nn.Linear(2 * hiddenSize, contextSize)
		self.wordattention = Attention()

		# Sentence Encoder
		self.sentenceBiGRU = nn.GRU(1, hiddenSize, numLayers, bias, batchFirst, dropout, bidirectional)
		self.sentenceLinear = nn.Linear(2 * hiddenSize, contextSize)
		self.sentenceattention = Attention()

		# Feed Forward
		self.fc = nn.Linear(2 * hiddenSize, numClasses)

	def forward(self, x):
		'''
		@doc: HAN
		@author: Alpaca-Man
		@date: 2021/2/26
		@param: {
		            x: [batchSize, seqLen]
		        }
		@return: {
					output: [batchSize. 2 * hiddenSize]
				}
		'''

		# x: [batchSize, seqLen] -> x: [batchSize, seqLen, embSize]
		x = self.emb(x)
		# x: [batchSize, seqLen, embSize] -> wordH: [batchSize, seqLen, 2 * hiddenSize]
		wordH, _ = self.wordBiGRU(x)

		# wordH: [batchSize, seqLen, 2 * hiddenSize] -> wordU: [batchSize, seqLen, contextSize]
		wordU = torch.tanh(self.wordLinear(wordH))

		wordAttention = self.wordattention(wordU)  # wordAttention: [batchSize, seqLen, 1]

		# wordH: [batchSize, seqLen, 2 * hiddenSize] wordAttention: [batchSize, seqLen, 1]
		s = torch.matmul(wordH.transpose(1, 2), wordAttention).squeeze(-1)  # s: [batchSize, 2 * hiddenSize]

		# s: [batchSize, 2 * hiddenSize, 1] -> sentenceH: [batchSize, 2 * hiddenSize, 2 * hiddenSize]
		sentenceH, _ = self.sentenceBiGRU(s.unsqueeze(-1))
		# sentenceH: [batchSize, 2 * hiddenSize, 2 * hiddenSize] -> sentenceU: [batchSize, 2 * hiddenSize, contextSize]
		sentenceU = torch.tanh(self.sentenceLinear(sentenceH))

		sentenceAttention = self.sentenceattention(sentenceU)  # sentenceAttention: [batchSize, 2 * hiddenSize, 1]

		# sentenceH: [batchSize, 2 * hiddenSize, 2 * hiddenSize] sentenceAttention: [batchSize, 2 * hiddenSize, 1]
		v = torch.matmul(sentenceH.transpose(1, 2), sentenceAttention).squeeze(-1)  # v: [batchSize, 2 * hiddenSize]

		# v: [batchSize, numClasses]
		output = self.fc(v)
		return output