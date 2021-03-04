import math
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data

'''
@doc: 处理运算工具包
@author: Alpaca-Man
@date: 2021/2/22
'''

dtype = torch.LongTensor # 数据类型
device = torch.device("cuda" if not torch.cuda.is_available() else "cpu") # 运算设备

dropout = 0.1 # 失活率
bias = False # 偏置
numHeads = 8 # 头数
tgtLen = 50 # 生成句子长度


class MyDataSet(Data.Dataset):
	def __init__(self, enInputs, deInputs, deOutputs):
		'''
		@doc: 该类负责继承 Data.Dataset 用于迭代处理数据
		@author: Alpaca-Man
		@date: 2021/2/22
		@param: {
					enInputs: Encoder 端的输入
					deInputs: Decoder 端的输入
					deOutputs: Decoder 端的输出
		        }
		@return: { }
		'''

		super(MyDataSet, self).__init__()
		self.enInputs = enInputs
		self.deInputs = deInputs
		self.deOutputs = deOutputs

	def __len__(self):
		return self.enInputs.shape[0]

	def __getitem__(self, idx):
		return self.enInputs[idx], self.deInputs[idx], self.deOutputs[idx]


class Position(nn.Module):
	def __init__(self, vocabSize, embSize, dropout=dropout):
		'''
		@doc: 该类负责生成输出序列的位置编码信息，并与原输入(已通过词嵌入层)相加后返回
		@author: Alpaca-Man
		@date: 2021/2/22
		@param: {
					vocabSize: 单词表长度 (=enVocabSize or deVocabSize)
					embSize: Position Encoding 出的维度(=embSize)
		        }
		@return: { }
		'''

		super(Position, self).__init__()
		self.dropout = nn.Dropout(p=dropout) # Position Encoding 处的位置信息随机丢失率
		self.vocabSize = vocabSize # Encoding 端和 Decoding 端的单词表长度不同
		self.embSize = embSize # 词向量维度


	def forward(self, inputs):
		'''
		@doc: Position Encoding Calculation
		@author: Alpaca-Man
		@date: 2021/2/22
		@param: {
					inputs: Encoder or Decoder 端的输入 [batchSize, len(=seqLen or tgtLen), embSize]
		        }
		@return: {
					outputs: 原输入加上位置编码处理得到的结果 [batchSize, len, embSize]
		    	}
		'''

		# inputs: [batchSize, len, embSize] -> inputs: [len, batchSize, embSize]
		inputs = inputs.transpose(0,1)

		pos = torch.zeros(self.vocabSize, self.embSize)
		position = torch.arange(0, self.vocabSize , dtype=torch.float).unsqueeze(1) # position: [vocabSize(= enVocabSize or deVocabSize), 1]
		div = torch.exp(torch.arange(0, self.embSize, 2).float() * (-math.log(10000.0) / self.embSize))

		pos[:, 0::2] = torch.sin(position * div)
		pos[:, 1::2] = torch.sin(position * div)

		# pos: [vocabSize, embSize] -> pos: [vocabSize, 1, embSize]
		pos = pos.unsqueeze(0).transpose(0, 1)
		inputs += pos[:inputs.size(0), :]

		# inputs: [len, batchSize, embSize] -> outputs: [batchSize, len, embSize]
		outputs = self.dropout(inputs).transpose(0, 1)

		return outputs


class MultiHeadAttention(nn.Module):
	def __init__(self,embSize, qSize, kSize, vSize, numHeads=numHeads, bias=bias):
		'''
		@doc: 该类主要根据输入生成对应的多头注意力，输出维度为[batch_size, len(=seqLen or tgtLen), embSize]
		@author: Alpaca-Man
		@date: 2021/2/22
		@param: {
					embSize: 词向量维度
					qSize: Q 的维度
					kSize: K 的维度(需要与 Q 的维度相等)
					vSize: V 的维度
					numHeads: 头数 default 8
					bias: 偏置 default False
			    }
		@return: { }
		'''

		super(MultiHeadAttention, self).__init__()
		self.qSize = qSize # Q 的维度
		self.kSize = kSize # K 的维度(需要与 Q 的维度相等)
		self.vSize = vSize # V 的维度
		self.numHeads = numHeads # 头数

		self.W_Q = nn.Linear(embSize, qSize * numHeads, bias=bias)
		self.W_K = nn.Linear(embSize, kSize * numHeads, bias=bias)
		self.W_V = nn.Linear(embSize, vSize * numHeads, bias=bias)
		self.fc = nn.Linear(vSize * numHeads, embSize, bias=bias)

	def forward(self, qInputs, kvInputs, mask):
		'''
		@doc: MultiHeadAttention Calculation
		@author: Alpaca-Man
		@date: 2021/2/22
		@param: {
					qInputs: Encoder 端输入 or Decoder 端输入 [batchSize, len(=seqLen or tgtLen), embSize]
					kvInputs: Encoder 端输入 or Decoder 端输入 [batchSize, len(=seqLen or tgtLen), embSize]
					mask: Encoder 端输入的 Padding Mask [batchSize, numHeads, seqLen, seqLen]
						 	or Decoder 端输入的 Padding Mask and Tokens Mask [batchSize, numHeads, tgtLen, tgtLen]
			    }
		@return: {
					context: 加了 Attention 的 inputs [batch_size, len, embSize]
				}
		'''
		batchSize = qInputs.size(0)
		# inputs: [batch_size, len(=seqLen or tgtLen), embSize] -> Q, K, V: [batch_size, len(=seqLen or tgtLen), size(=qSize or kSize or vSize) * numHeads]
		# Q, K, V: [batch_size, len(=seqLen or tgtLen), size(=qSize or kSize or vSize) * numHeads] -> Q, K, V: [batch_size, numHeads, len(=seqLen or tgtLen), size(=qSize or kSize or vSize)]
		Q = self.W_Q(qInputs).view(batchSize, -1, self.numHeads, self.qSize).transpose(1, 2)
		K = self.W_K(kvInputs).view(batchSize, -1, self.numHeads, self.kSize).transpose(1, 2)
		V = self.W_V(kvInputs).view(batchSize, -1, self.numHeads, self.vSize).transpose(1, 2)

		# 下面这一步即 Scaled Dot-Product Attention 处理
		scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.qSize) # scores: [batchSize, numHeads, len, len]
		# scores 对应 mask 为 true 的部分赋予无穷小, 这样 Softmax 得到的结果趋近于0无梯度
		scores.masked_fill_(mask, -1e9)

		# attention 在最后一个维度 Softmax
		attention = nn.Softmax(dim=-1)(scores) # attention: [batch_size, numHeads, len, len]
		context = torch.matmul(attention, V) # context: [batch_size, numHeads, len,vSize]

		# context: [batch_size, numHeads, len,vSize] -> context: [batch_size, len, vSize * numHeads]
		context = context.transpose(1, 2).reshape(batchSize, -1, self.vSize * self.numHeads)
		context = self.fc(context) # context: [batch_size, len, embSize]

		return context


class AddNorm(nn.Module):
	def __init__(self):
		'''
		@doc: 该类用于对输入进行 Add&Norm 处理，返回维度与输入相同 [batchSize, len, embSize]
		@author: Alpaca-Man
		@date: 2021/2/22
		@param: { }
		@return: { }
		'''

		super(AddNorm, self).__init__()


	def forward(self, old, new):
		'''
		@doc: AddNorm Calculation
		@author: Alpaca-Man
		@date: 2021/2/22
		@param: {
					old: Multi-Head Attention or Feed Forward or Masked Multi-Head Attention 前的输入 [batchSize, len, embSize]
					new: Multi-Head Attention or Feed Forward or Masked Multi-Head Attention 的输出 [batchSize, len, embSize]
			    }
		@return: {
					: [batchSize, len, embSize]
				}
		'''
		residual = old + new
		embSize = residual.size(2)
		return nn.LayerNorm(embSize).to(device)(residual)


class FeedForward(nn.Module):
	def __init__(self, embSize, hiddenSize, bias=bias):
		'''
		@doc: 该类为两层全连接神经网络，把输入投射到样本空间中，输出与输入维度相同
		@author: Alpaca-Man
		@date: 2021/2/22
		@param: {
					embSize: 词向量维度
					hiddenSize: 隐藏层维度
					bias: 偏置 default False
				}
		@return: { }
		'''

		super(FeedForward, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(embSize, hiddenSize, bias=bias), # 也可以采用一维卷积
			nn.ReLU(),
			nn.Linear(hiddenSize, embSize, bias=bias)
		)

	def forward(self, inputs):
		'''
		@doc: FeedForward Calculation
		@author: Alpaca-Man
		@date: 2021/2/22
		@param: {
					inputs: AddNorm 层的输出 [batchSize, len, embSize]
			    }
		@return: {
					: [batchSize, len, embSize]
				}
		'''

		return self.fc(inputs)


def makeData(sentences, enVocab, deVocab):
	'''
	@doc: 该函数用于处理数据，返回三个 dtype 类型的矩阵
    @author: Alpaca-Man
    @date: 2021/2/22
    @param: {
                sentences: 包含 Encoder 端输入 和 Decoder 端输入输出的全部句子
                enVocab: Encoder 端单词表
                deVocab: Decoder 端单词表
        	}
    @return: {
         		enInputs: Encoder 端输入 [batchSize, seqLen]
         		deInputs: Decoder 端输入 [batchSize, tgtLen]
         		deOutputs: Decoder 端输出 [batchSize, tgtLen]
         	}
	'''

	enInputs, deInputs, deOutputs = [], [], []
	for i in range(len(sentences)):
		enInput = [[enVocab[n] for n in sentences[i][0].split()]]
		deInput = [[deVocab[n] for n in sentences[i][1].split()]]
		deOutput = [[deVocab[n] for n in sentences[i][2].split()]]

		enInputs.extend(enInput) # [batchSize, seqLen]
		deInputs.extend(deInput) # [batchSize, tgtLen]
		deOutputs.extend(deOutput) # [batchSize, tgtLen]

	return dtype(enInputs), dtype(deInputs), dtype(deOutputs)


def paddingMask(qInputs, kvInputs, numHeads=numHeads):
	'''
	@doc: 该函数用于对填充单词 'PAD' or 'P' 进行处理,返回一个填充词位置为 True 其他位置为 False ,bool类型的 [batchSize, numHeads, len(=seqLen or tgtLen), len]
    @author: Alpaca-Man
    @date: 2021/2/22
    @param: {
                qInputs: Encoder 端的输入 [batchSize, seqLen] or Decoder 端的输入 [batchSize, tgtLen]
                kvInputs: Encoder 端的输入 [batchSize, seqLen] or Decoder 端的输入 [batchSize, tgtLen]
        	}
    @return: {
         		pMask: Encoder 的[batchSize, numHeads, seqLen, seqLen]
         				Decoder 的[batchSize, numHeads, tgtLen, tgtLen]
         				Encoder 和 Decoder 的[batchSize, numHeads, tgtLen, seqLen] type = bool
         	}
	'''

	batchSize, qLen = qInputs.size()
	kvLen = kvInputs.size(1)

	# pMask: [batchSize, 1, kvLen] -> pMask: [batchSize, q_len, kvLen] -> pMask: [batchSize, numHeads, q_len, kvLen]
	pMask = kvInputs.data.eq(0).unsqueeze(1)
	pMask = pMask.expand(batchSize, qLen, kvLen)
	pMask = pMask.unsqueeze(1).repeat(1, numHeads, 1, 1).bool()

	return pMask


def tokensMask(qInputs, kvInputs, numHeads=numHeads):
	'''
	@doc: 该函数主要用于 Decoder 端输入的 tokensMask，返回一个覆盖位置为 True 其他位置为 False ,bool类型的 [batchSize, numHeads, len(=seqLen or tgtLen), len]
    @author: Alpaca-Man
    @date: 2021/2/22
    @param: {
                qInputs: Encoder 端的输入 [batchSize, seqLen] or Decoder 端的输入 [batchSize, tgtLen]
                kvInputs: Encoder 端的输入 [batchSize, seqLen] or Decoder 端的输入 [batchSize, tgtLen]
            }
    @return: {
         		tMask: Encoder 的[batchSize, numHeads, seqLen, seqLen]
         				Decoder 的[batchSize, numHeads, tgtLen, tgtLen]
         	}
	'''

	shape = [qInputs.size(0), qInputs.size(1), kvInputs.size(1)]

	# tMask: [batchSize, q_len, kvLen] -> tMask: [batchSize, numHeads, q_len, kvLen]
	tMask = torch.from_numpy(np.triu(np.ones(shape), k=1))
	tMask = tMask.unsqueeze(1).repeat(1, numHeads, 1, 1).bool()

	return tMask


def getMask(qInputs, kvInputs, numHeads=numHeads):
	'''
	@doc: 该函数用于对输入同时进行  paddingMask 和 tokensMask 操作，返回一个Mask位置为 True 其他位置为 False ,bool类型的 [batchSize, numHeads, len(=seqLen or tgtLen), len]
	@author: Alpaca-Man
	@date: 2021/2/22
	@param: {
	            qInputs: Encoder 端的输入 [batchSize, seqLen] or Decoder 端的输入 [batchSize, tgtLen]
	            kvInputs: Encoder 端的输入 [batchSize, seqLen] or Decoder 端的输入 [batchSize, tgtLen]
	        }
	@return: {
	     		selfMask: selfMask + tokensMask
	     		 		Encoder 的[batchSize, numHeads, seqLen, seqLen]
	     				Decoder 的[batchSize, numHeads, tgtLen, tgtLen]
	     	}
	'''

	selfPMask = paddingMask(kvInputs, kvInputs, numHeads)
	selfTMask = tokensMask(kvInputs, kvInputs, numHeads)
	selfMask = torch.gt((selfPMask + selfTMask), 0).to(device)  # selfMask: [batchSize, numHeads, tgtLen, tgtLen] dtype = bool
	return selfMask


def greedyDecoding(model, enInputs, tgtLen=tgtLen, startSymbol=1):
	'''
	@doc: 该函数主要用于对输入进行贪婪解码，当前 t 时刻输出的单词只能考虑前 t-1 时刻的单词，返回 [batchSize(=1) * , tgtlen]
	@author: Alpaca-Man
	@date: 2021/2/23
	@param: {
	            model: 训练好的模型
	            enInputs: Encoder 端的输入 [1, seq_len]
	        	tgtLen: 输出序列的最大长度 default 50
	        	startSymbol: 开始标志的单词索引 default 6
	        }
	@return: {
	        	deInputs: 输出的序列的索引 [1, tgt_len]
	        }
	'''

	deInputs = torch.zeros(1, tgtLen).type_as(enInputs.data)

	# 进行 Mask 操作
	pMask = paddingMask(enInputs, enInputs, model.numHeads).to(device)
	selfMask = getMask(deInputs, deInputs, model.numHeads).to(device) # selfMask: [batchSize, numHeads, tgtLen, tgtLen] dtype = bool
	enDeMask = paddingMask(deInputs, enInputs, model.numHeads).to(device) # enDeMask: [batchSize, numHeads, tgtLen, seqLen] dtype = bool

	enOutputs = model.encoder(enInputs, pMask)
	nextSymbol = startSymbol

	for i in range(0, tgtLen):
		deInputs[0][i] = nextSymbol
		deOutputs = model.decoder(deInputs, enOutputs, selfMask, enDeMask).to(device)
		pred = model.linear(deOutputs).to(device) # pred: [batchSize(=1), tgtLen, tgt_vocab]
		pred = pred.squeeze(0).max(dim=-1, keepdim=False)[1]
		nextWord = pred.data[i]
		nextSymbol = nextWord.item()


	# deInputs: [batchSize(=1), tgtLen, deVocabSize] -> deInputs: [batchSize(=1) * tgtlen, deVocabSize]
	# deInputs = deInputs.view(-1, deInputs.size(-1))
	return deInputs