import torch
import torch.nn as nn
import torch.nn.functional as F
from HAN import HAN

batchSize = 64 # 批次大小
seqLen = 60 # 输入语句长度
embSize = 200 # 词向量维度
hiddenSize = 50 # BiGRU 隐藏层维度
numLayers = 1 # BiGRU 层数
vocabSize = 100 # 词汇表大小

x = torch.rand(batchSize, seqLen).long()
model = HAN(vocabSize)
y = model(x)
print('模型输入 x 维度为:[{}, {}]'.format(x.size(0), x.size(1)))
print('模型输出 y 维度为:[{}, {}]'.format(y.size(0), y.size(1)))
