import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from Utils import makeData, train
from HAN import HAN

'''
@doc: 主函数
@author: Alpaca-Man
@date: 2021/2/26
'''

# 可调整参数
dtype = torch.LongTensor # 数据类型
device = torch.device("cuda" if not torch.cuda.is_available() else "cpu") # 运算设备

# HAN 参数
seqLen = 3 # 每一句的长度
numClasses = 2 # 标签种类数量
batchSize = 3 # 每个批次的大小

embSize = 2 # 词嵌入维度
hiddenSize = 3 # 隐藏层维度
numLayers = 1 # 双向 GRU 层数
dropout = 0 # 失活率
bidirectional = False # RNN 是否双向
lr = 1e-3 # 学习率
epoch = 100 # 训练次数

# 数据集
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

# 获取单词表和引索
wordList = " ".join(sentences).split() # 全部单词（重复）
vocab = list(set(wordList)) # 单词表（不重复）
word2idx = {w: i for i, w in enumerate(vocab)} # 单词索引
vocabSize = len(vocab) # 单词个数

# 封装成数据集 加载器
trainInput, trainTarget = makeData(sentences, word2idx, labels) # 用于把数据集处理成数组
trainInput, trainTarget = dtype(trainInput).to(device), dtype(trainTarget).to(device)
trainDataSet = Data.TensorDataset(trainInput, trainTarget)
trainDataLoader = Data.DataLoader(trainDataSet, batchSize, shuffle = True) # 打乱

# GRU
model = HAN(vocabSize, numClasses).to(device)
criterion = nn.CrossEntropyLoss().to(device)# 损失函数
optimizer = optim.Adam(model.parameters(), lr = lr) # 优化器

# Training
train(model, epoch, trainDataLoader, criterion, optimizer)

# testSet及处理
testSentences = ["i hate me","you love me"]
testLabels = [0,1]
testInput, testTarget = makeData(testSentences, word2idx, testLabels)
testInput = dtype(testInput).to(device)
testTarget = dtype(testTarget).to(device)

# 封装成数据集 加载器
testDataSet = Data.TensorDataset(testInput, testTarget)
testDataLoader = Data.DataLoader(testDataSet, 2, shuffle = False) # 不打乱

# Predict
model = model.eval()
for testInput, _ in testDataLoader:
    num = testInput.shape[0]
    predict = model(testInput).data.max(1, keepdim=True)[1]# 查一下 .max() 这个函数
    for i in range(num):
        if predict[i][0] == 0:
            print(testSentences[i],"is Bad Mean...")
        else:
            print(testSentences[i],"is Good Mean!!")