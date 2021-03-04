import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from Math import MyDataSet, makeData, Position, MultiHeadAttention, AddNorm, FeedForward, paddingMask, tokensMask, greedyDecoding
from Transformer import Encoder, Decoder, Transformer

'''
@doc: 主函数
@author: Alpaca-Man
@date: 2021/2/22
'''

dtype = torch.LongTensor # 数据类型
device = torch.device("cuda" if not torch.cuda.is_available() else "cpu") # 运算设备

bias = False # 偏置
batchSize = 1 # 批次大小
embSize = 512 # 词向量维度
hiddenSize = 2048 # 全连接层的中间维度
seqLen = 5 # Encoder 端输入语句的长度
tgtLen = 6 # Decoder 端输入语句的长度和输出句子的长度

enVocabSize = 100 # Encoder 单词表长度
deVocabSize = 100 # Decoder 单词表长度
dropout = 0.1 # 失活率

qSize = 64 # Q 的维度
kSize = 64 # K 的维度(需要和 Q 的维度相等)
vSize = 64 # V 的维度
numHeads = 8 # 头数
numLayers = 8 # 层数

EPOCH = 100 # 训练次数

# S: Symbol that shows starting of deInputs
# E: Symbol that shows ending of deIoutputs
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
sentences = [
	# enInputs           		deInputs         	deOutputs
	['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
	['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# Padding Should be Zero
enVocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
enIdx2Word = {i: w for i, w in enumerate(enVocab)}
enVocabSize = len(enVocab)

deVocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
deIdx2Word = {i: w for i, w in enumerate(deVocab)}
deVocabSize = len(deVocab)

#把字符串数据集转化为数字数据集
enInputs, deInputs, deOutputs = makeData(sentences, enVocab, deVocab)

loader = Data.DataLoader(MyDataSet(enInputs, deInputs, deOutputs), batchSize, True)

model = Transformer(enVocabSize, deVocabSize, embSize, qSize, kSize, vSize,
					hiddenSize, numHeads, bias, numLayers, dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

print('模型训练开始')
for epoch in range(EPOCH):
	for enInputs, deInputs, deOutputs in loader:
		enInputs, deInputs, deOutputs = enInputs.to(device), deInputs.to(device), deOutputs.to(device)
		pred = model(enInputs, deInputs)
		pred = pred.view(-1, pred.size(-1))
		loss = criterion(pred, deOutputs.view(-1))
		if (epoch + 1) % 100 == 0 :
			print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

print('模型训练完毕')

enInputs, _, _ = next(iter(loader))
pred = greedyDecoding(model, enInputs, tgtLen, 6)# pred: [batchSize, tgtLen]
pred = model(enInputs, pred)

# pred: [batchSize, tgtLen, deVocabSize] -> pred: [batchSize * tgtLen, deVocabSize]
pred = pred.view(-1, pred.size(-1))

pred = pred.data.max(1, keepdim=True)[1]
print([enIdx2Word[n.item()] for n in enInputs[0]], '->', [deIdx2Word[n.item()] for n in pred.squeeze()])
