import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from Math import Position, MultiHeadAttention, AddNorm, FeedForward, paddingMask, tokensMask, greedyDecoding
from Transformer import Encoder, Decoder, Transformer


dtype = torch.LongTensor # 数据类型
device = torch.device("cuda" if not torch.cuda.is_available() else "cpu") # 运算设备

bias = False
batchSize = 64
vocabSize = 1000
embSize = 512
hiddenSize = 2048
seqLen = 5
tgtLen = 6

enVocabSize = 100
deVocabSize = 100
dropout = 0.1

qSize = 64
kSize = 64
vSize = 64
numHeads = 8
numLayers = 8

EPOCH = 1
# 完整测试
# S: Symbol that shows starting of decoding input
# E: Symbol that shows ending of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
sentences = [
	# enc_input           dec_input         dec_output
	['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
	['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# Padding Should be Zero
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
enVocabSize = len(src_vocab)

tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
deVocabSize = len(tgt_vocab)



#把字符串数据集转化为数字数据集
def make_data(sentences):
	'''

	:param sentences: 二维列表，第一维大小是句子数量，第二维是三条句子（Encoder输入语句，Decoder输入语句，Decoder输出语句）
	:return: 都是LongTensor类型二维数组，维度分别为
	'''
	enc_inputs, dec_inputs, dec_outputs = [], [], []
	for i in range(len(sentences)):
		enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]  # [2,src_len]
		dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]  # [2,tgt_len]
		dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]  # [2,tgt_len]

		enc_inputs.extend(enc_input)
		dec_inputs.extend(dec_input)
		dec_outputs.extend(dec_output)

	return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


enc_inputs, dec_inputs, dec_outputs = make_data(sentences)

class MyDataSet(Data.Dataset):
	def __init__(self, enc_inputs, dec_inputs, dec_outputs):
		super(MyDataSet, self).__init__()
		self.enc_inputs = enc_inputs
		self.dec_inputs = dec_inputs
		self.dec_outputs = dec_outputs

	def __len__(self):
		return self.enc_inputs.shape[0]

	def __getitem__(self, idx):
		return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]
loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 1, True)

model = Transformer(enVocabSize, deVocabSize, embSize, qSize, kSize, vSize,
					hiddenSize, numHeads, bias, numLayers, dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

for epoch in range(EPOCH):
	for enc_inputs, dec_inputs, dec_outputs in loader:
		'''
		enc_inputs: [batch_size, src_len]
		dec_inputs: [batch_size, tgt_len]
		dec_outputs: [batch_size, tgt_len]
		'''
		enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
		# outputs: [batch_size * tgt_len, tgt_vocab_size]
		outputs = model(enc_inputs, dec_inputs)
		outputs = outputs.view(-1, outputs.size(-1))
		loss = criterion(outputs, dec_outputs.view(-1))
		if (epoch + 1) % 100 ==0 :
			print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

print('模型训练完毕')
enInputs, _, _ = next(iter(loader))

# pred: [batchSize, tgtLen]
pred = greedyDecoding(model, enInputs, tgtLen, 6)
pred = model(enInputs, pred)

# pred: [batchSize, tgtLen, deVocabSize] -> pred: [batchSize * tgtLen, deVocabSize]
pred = pred.view(-1, pred.size(-1))

pred = pred.data.max(1, keepdim=True)[1]
print(enc_inputs[0], '->', [idx2word[n.item()] for n in pred.squeeze()])

# greedyDecoding
'''
enInputs = torch.rand(batchSize, seqLen).long()
deInputs = torch.rand(batchSize, tgtLen).long()
model = Transformer(enVocabSize, deVocabSize, embSize, qSize, kSize, vSize,
					hiddenSize, numHeads, bias, numLayers, dropout)

deOutputs = model(enInputs, deInputs)
print('模型训练完毕')
enInputs = torch.rand(1, seqLen).long()
pred = greedyDecoding(model, enInputs, tgtLen, 1)
pred = model(enInputs, pred)
pred = pred.data.max(1, keepdim=True)[1]
print([n for n in pred.squeeze()])
'''

# Transformer
'''
enInputs = torch.rand(batchSize, seqLen).long()
deInputs = torch.rand(batchSize, tgtLen).long()
model = Transformer(enVocabSize, deVocabSize, embSize, qSize, kSize, vSize,
					hiddenSize, numHeads, bias, numLayers, dropout)
deOutputs = model(enInputs, deInputs)
print(deOutputs.shape)
'''

# mask
'''
enInputs = torch.rand(batchSize, seqLen)
enInputsEmb = torch.rand(batchSize, seqLen, embSize)
pMask = paddingMask(enInputs, enInputs, numHeads).bool()
tMask = tokensMask(enInputs, enInputs, numHeads).bool()

selfMask1 = torch.gt((pMask + tMask), 0)
selfMask1 = selfMask1.unsqueeze(1).repeat(1, numHeads, 1, 1).bool()
print('selfMask1:{}'.format(selfMask1.shape))

pMask = pMask.unsqueeze(1).repeat(1, numHeads, 1, 1).bool()
tMask = tMask.unsqueeze(1).repeat(1, numHeads, 1, 1).bool()
selfMask2 = torch.gt((pMask + tMask), 0)
print('selfMask2:{}'.format(selfMask2.shape))
print(torch.all(torch.eq(selfMask1,selfMask2)))
'''

# Encoder Decoder
'''
enOutputs = torch.rand(batchSize, seqLen, embSize)
deInputs = torch.rand(batchSize, tgtLen, embSize)
selfMask = torch.ones(batchSize, numHeads, tgtLen, tgtLen).bool()
enDeMask = torch.ones(batchSize, numHeads, tgtLen, seqLen).bool()
decoder = Decoder(embSize, qSize, kSize, vSize, hiddenSize, numHeads=numHeads, bias=bias, numLayers=numLayers)
deOutputs = decoder(deInputs, enOutputs, selfMask, enDeMask)
print(deOutputs.shape)
'''

# Decoder
'''
enOutputs = torch.rand(batchSize, seqLen, embSize)
deInputs = torch.rand(batchSize, tgtLen, embSize)
selfMask = torch.ones(batchSize, numHeads, tgtLen, tgtLen).bool()
enDeMask = torch.ones(batchSize, numHeads, tgtLen, seqLen).bool()
decoder = Decoder(embSize, qSize, kSize, vSize, hiddenSize, numHeads=numHeads, bias=bias, numLayers=numLayers)
deOutputs = decoder(deInputs, enOutputs, selfMask, enDeMask)
print(deOutputs.shape)
'''

# Encoder
'''
emb = nn.Embedding(enVocabSize, embSize)
enInputs = torch.rand(batchSize, seqLen).long()
# enInputsEmb = torch.rand(batchSize, seqLen, embSize)
enInputsEmb = emb(enInputs)
print(enInputsEmb.shape)
pMask = paddingMask(enInputs, enInputs, numHeads)
tMask = tokensMask(enInputs, enInputs, numHeads)
print(pMask.shape)
print(tMask.shape)
selfMask = torch.gt((pMask + tMask), 0)
print(selfMask.shape)
encoder = Encoder(embSize, qSize, kSize, vSize, hiddenSize, numHeads=numHeads, bias=bias, numLayers=numLayers)
enOutputs = encoder(enInputsEmb, selfMask)
print(enOutputs.shape)
'''

'''
a = torch.rand(batchSize, len, embSize)
mask = torch.ones(batchSize, numHeads, len, len).bool()
attn = MultiHeadAttention(embSize, qSize, kSize, vSize, numHeads, bias)
context = attn(a, mask)
print(context.shape)
'''
'''
a = torch.rand(batchSize, len, embSize)
fc = FeedForward(embSize, hiddenSize, bias)
b = fc(a)
print(b.shape)
'''
'''
old = torch.rand(batchSize, len, embSize)
new = torch.rand(batchSize, len, embSize)
addNorm = AddNorm().to(device)
residual = addNorm(old, new)
print(residual.shape)
'''
'''
pos = Position(vocabSize, embSize)
a = torch.rand(batchSize, seqLen, embSize)
a = pos(a)
print(a.shape)
'''
'''
a = torch.arange(0, 4, 4)
print(a)

pe = torch.arange(0,25).view(5,5)
print(pe)
print(pe[:, 1::2])
print(pe[1::2])

pos = torch.zeros(5, 6)
print(pos.shape)
pos = pos.unsqueeze(0).transpose(0, 1)
print(pos.shape)
'''