import torch
import torch.nn as nn

from Math import Position, MultiHeadAttention, AddNorm, FeedForward, paddingMask, tokensMask, getMask

'''
@doc: Transformer
@author: Alpaca-Man
@date: 2021/2/22
'''


dtype = torch.LongTensor # 数据类型
device = torch.device("cuda" if not torch.cuda.is_available() else "cpu") # 运算设备

dropout = 0.1 # 失活率
numLayers = 8 # Encoder and Decoder 层数
numHeads = 8 # 头数
bias = False # 偏置


class EncoderLayer(nn.Module):
	def __init__(self, embSize, qSize, kSize, vSize, hiddenSize, numHeads=numHeads, bias=bias):
		'''
		@doc: EncoderLayer
		@author: Alpaca-Man
		@date: 2021/2/22
		@param: {
					embSize: 词向量维度
					qSize: Q 的维度
					kSize: K 的维度(需要与 Q 的维度相等)
					vSize: V 的维度
					hiddenSize: 两层全连接层中间的维度
					numHeads: 头数 default 8
					bias: 偏置 default False
			    }
		@return: { }
		'''
		super(EncoderLayer, self).__init__()
		self.subLayer1 = MultiHeadAttention(embSize, qSize, kSize, vSize, numHeads=numHeads, bias=bias)
		self.subLayer2 = FeedForward(embSize, hiddenSize)
		self.addnorm = AddNorm()

	def forward(self, enInputs, mask):
		'''
		@doc: EncoderLayer
		@author: Alpaca-Man
		@date: 2021/2/22
		@param: {
					enInputs: Encoder 端输入 [batchSize, seqLen, embSize]
					mask: Encoder 端输入的 paddingMask [batchSize, numHeads, seqLen, seqLen]

			    }
		@return: {
					enOutputs: Single Encoder 层的输出 [batchSize, seqLen, embSize]
				}
		'''

		temp = self.subLayer1(enInputs, enInputs, mask)
		middle = self.addnorm(enInputs, temp)

		temp = self.subLayer2(middle)
		enOutputs = self.addnorm(middle, temp)

		return enOutputs


class DecoderLayer(nn.Module):
	def __init__(self, embSize, qSize, kSize, vSize, hiddenSize, numHeads=numHeads, bias=bias):
		'''
		@doc: DecoderLayer
		@author: Alpaca-Man
		@date: 2021/2/22
		@param: {
					embSize: 词向量维度
					qSize: Q 的维度
					kSize: K 的维度(需要与 Q 的维度相等)
					vSize: V 的维度
					hiddenSize: 两层全连接层中间的维度
					numHeads: 头数 default 8
					bias: 偏置 default False
			    }
		@return: { }
		'''

		super(DecoderLayer, self).__init__()
		self.subLayer1 = MultiHeadAttention(embSize, qSize, kSize, vSize, numHeads=numHeads, bias=bias)
		self.subLayer2 = MultiHeadAttention(embSize, qSize, kSize, vSize, numHeads=numHeads, bias=bias)
		self.subLayer3 = FeedForward(embSize, hiddenSize)
		self.addnorm = AddNorm()

	def forward(self, deInputs, enOutputs, selfMask, enDeMask):
		'''
		@doc: DecoderLayer
		@author: Alpaca-Man
		@date: 2021/2/22
		@param: {
					deInputs: Decoder 端的输入 [batchSize, tgtLen]
					enOutputs: Encoder 端的输出 [batchSize, seqLen, embSize]
					selfMask: Decoder 端输入的 paddingMask 和 tokensMask [batchSize, numHeads, tgtLen, tgtLen]
					enDeMask: Encoder 和 Decoder 之间的 paddingMask [batchSize, numHeads, tgtLen, seqLen]
			    }
		@return: {
					deOutputs: Single Decoder 层的输出 [batchSize, tgtLen, embSize]
				}
		'''

		temp = self.subLayer1(deInputs, deInputs, selfMask)
		middle = self.addnorm(deInputs, temp)

		temp = self.subLayer2(middle, enOutputs, enDeMask)
		middle = self.addnorm(middle, temp)

		temp = self.subLayer3(middle)
		deOutputs = self.addnorm(middle, temp)

		return deOutputs


class Encoder(nn.Module):
	def __init__(self, enVocabSize, embSize, qSize, kSize, vSize, hiddenSize, numHeads=numHeads, bias=bias, numLayers=numLayers, dropout=dropout):
		'''
		@doc: Encoder
		@author: Alpaca-Man
		@date: 2021/2/22
		@param: {
					enVocabSize: Decoder 单词表长度
					embSize: 词向量维度
					qSize: Q 的维度
					kSize: K 的维度(需要与 Q 的维度相等)
					vSize: V 的维度
					hiddenSize: 两层全连接层中间的维度
					numHeads: 头数 default 8
					bias: 偏置 default False
		            numLayers: Encoder 层数 default 8
		            dropout: 失活率 default 0.1
			    }
		@return: { }
		'''

		super(Encoder, self).__init__()
		self.enEmb = nn.Embedding(enVocabSize, embSize)
		self.enPos = Position(enVocabSize, embSize, dropout).to(device)
		self.layers = nn.ModuleList([EncoderLayer(embSize, qSize, kSize, vSize, hiddenSize, numHeads=numHeads, bias=bias) for _ in range(numLayers)]).to(device)

	def forward(self, enInputs, mask):
		'''
		@doc: Encoder
		@author: Alpaca-Man
		@date: 2021/2/22
		@param: {
					enInputs: Encoder 端的输入 [batchSize, seqLen]
					mask: Encoder 端输入的 Padding Mask [batchSize, numHeads, seqLen, seqLen]
				}
		@return: {
					enOutputs: Encoder 端的输出 [batchSize, seqLen, embSize]
				}
		'''

		enInputs = self.enEmb(enInputs)
		enInputs = self.enPos(enInputs)

		for layer in self.layers:
			enOutputs = layer(enInputs, mask)

		return  enOutputs


class Decoder(nn.Module):
	def __init__(self, deVocabSize, embSize, qSize, kSize, vSize, hiddenSize, numHeads=numHeads, bias=bias, numLayers=numLayers, dropout=dropout):
		'''
		@doc: Decoder
		@author: Alpaca-Man
		@date: 2021/2/22
		@param: {
					deVocabSize: Decoder 单词表长度
		            embSize: 词向量维度
					qSize: Q 的维度
					kSize: K 的维度(需要与 Q 的维度相等)
					vSize: V 的维度
					hiddenSize: 两层全连接层中间的维度
					numHeads: 头数 default 8
					bias: 偏置 default False
		            numLayers: Encoder 层数 default 8
			    }
		@return: { }
		'''

		super(Decoder, self).__init__()
		self.deEmb = nn.Embedding(deVocabSize, embSize)
		self.dePos = Position(deVocabSize, embSize, dropout).to(device)
		self.layers = nn.ModuleList([DecoderLayer(embSize, qSize, kSize, vSize, hiddenSize, numHeads=numHeads, bias=bias) for _ in range(numLayers)]).to(device)

	def forward(self, deInputs, enOutputs, selfMask, enDeMask):
		'''
		@doc: Decoder
		@author: Alpaca-Man
		@date: 2021/2/22
		@param: {
					deInputs: Decoder 端的输入 [batchSize, tgtLen]
					enOutputs: Encoder 端的输出 [batchSize, seqLen, embSize]
					selfMask: Decoder 端输入的 paddingMask 和 tokensMask [batchSize, numHeads, tgtLen, tgtLen]
					enDeMask: Encoder 和 Decoder 之间的 paddingMask [batchSize, numHeads, tgtLen, seqLen]
				}
		@return: {
					deOutputs: Decoder 端的输出 [batchSize, tgtLen, embSize]
				}
		'''

		deInputs = self.deEmb(deInputs)
		deInputs = self.dePos(deInputs)

		for layer in self.layers:
			deOutputs = layer(deInputs, enOutputs, selfMask, enDeMask)

		return deOutputs


class Transformer(nn.Module):
	def __init__(self, enVocabSize, deVocabSize, embSize,
				qSize, kSize, vSize, hiddenSize,
				numHeads=numHeads, bias=bias, numLayers=numLayers,
				dropout=dropout):
		'''
		@doc: Transformer
		@author: Alpaca-Man
		@date: 2021/2/22
		@param: {
		            embSize: 词向量维度
		            qSize: Q 的维度
		            kSize: K 的维度(需要与 Q 的维度相等)
		            vSize: V 的维度
		            hiddenSize: 全连接层的隐藏层维度
					numHeads: 头数 default 8
					bias: 偏置 default False
					numLayers: 层数 default 8
		            dropout: 失活率 default 0.1
		        }
		@return: { }
		'''

		super(Transformer, self).__init__()

		self.numHeads = numHeads # 用于后面 Mask 矩阵维度

		# Encoder and Decoder
		self.encoder = Encoder(enVocabSize, embSize, qSize, kSize, vSize, hiddenSize, numHeads=numHeads, bias=bias, numLayers=numLayers, dropout=dropout).to(device)
		self.decoder = Decoder(deVocabSize, embSize, qSize, kSize, vSize, hiddenSize, numHeads=numHeads, bias=bias, numLayers=numLayers, dropout=dropout).to(device)

		# Linear
		self.linear = nn.Linear(embSize, deVocabSize, bias=bias).to(device)

	def forward(self,enInputs, deInputs):
		'''
		@doc: Transformer
        @author: Alpaca-Man
        @date: 2021/2/22
        @param: {
                    enInputs: Encoder 端的输入 [batchSize, seqLen]
                    deInputs: Decoder 端的输入 [batchSize, tgtLen]
                }
        @return: {
         			deOutputs: [batchSize, tgtLen, deVocabSize]
         		}
		'''

		# Encoder 端的 Padding Mask
		mask = paddingMask(enInputs, enInputs, self.numHeads) # mask: [batchSize, numHeads, seqLen, seqLen] dtype = bool

		selfMask = getMask(deInputs, deInputs, self.numHeads) # selfMask: [batchSize, numHeads, tgtLen, tgtLen] dtype = bool

		# Encoder 给出的单词中没有 Decoder 的单词，即不存在“透题”，所以 Encoder 和 Decoder 交接处只需要 Padding Mask
		# enDeTMask = tokensMask(deInputs, enInputs)
		# enDeMask = enDePMask + enDeTMask # enDeMask: [batchSize, numHeads, tgtLen, seqLen] dtype = bool
		enDePMask = paddingMask(deInputs, enInputs, self.numHeads)
		enDeMask = torch.gt(enDePMask, 0).to(device) # enDeMask: [batchSize, numHeads, tgtLen, seqLen] dtype = bool


		enOutputs = self.encoder(enInputs, mask) # enOutputs: [batchSize, seqLen, embSize]
		deOutputs = self.decoder(deInputs, enOutputs, selfMask, enDeMask) # deOutputs: [batchSize, tgtLen, embSize]

		# deOutputs: [batchSize, tgtLen, embSize] -> deOutputs: [batchSize, tgtLen, deVocabSize]
		deOutputs = self.linear(deOutputs)

		return deOutputs



