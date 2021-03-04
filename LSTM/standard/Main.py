import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from Utils import make_data, train
from LSTM import LSTM

'''
@doc: 主函数
@author: Alpaca-Man
@date: 2021/2/16
'''

# 可调整参数
dtype = torch.LongTensor # 数据类型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 运算设备

# LSTM 参数
seq_len = 3 # 每一句的长度
num_classes = 2 # 标签种类数量
batch_size = 3 # 每个批次的大小

emb_size = 2 # 词嵌入维度
hidden_size = 3 # 隐藏层维度
num_layers = 1 # LSTM 层数
dropout = 0 # 失活率
bidirectional = False # LSTM 是否双向
lr = 1e-3 # 学习率
epoch = 5000 # 训练次数

# 数据集
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

# 获取单词表和引索
word_list = " ".join(sentences).split() # 全部单词（重复）
vocab = list(set(word_list)) # 单词表（不重复）
word2idx = {w: i for i, w in enumerate(vocab)} # 单词索引
vocab_size = len(vocab) # 单词个数

# 封装成数据集 加载器
trainInput, trainTarget = make_data(sentences, word2idx, labels) # 用于把数据集处理成数组
trainInput, trainTarget = torch.LongTensor(trainInput).to(device), torch.LongTensor(trainTarget).to(device)
trainDataSet = Data.TensorDataset(trainInput, trainTarget)
trainDataLoader = Data.DataLoader(trainDataSet, batch_size, shuffle = True) # 打乱

# LSTM
model = LSTM(vocab_size, emb_size, hidden_size, num_classes, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional).to(device)
criterion = nn.CrossEntropyLoss().to(device)# 损失函数
optimizer = optim.Adam(model.parameters(), lr = lr) # 优化器

# Training
train(model, epoch, trainDataLoader, criterion, optimizer)

# testSet及处理
test_sentences = ["i hate me","you love me"]
test_labels = [0,1]
testInput, testTarget = make_data(test_sentences, word2idx, test_labels)
testInput = torch.LongTensor(testInput).to(device)
testTarget = torch.LongTensor(testTarget).to(device)

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
            print(test_sentences[i],"is Bad Mean...")
        else:
            print(test_sentences[i],"is Good Mean!!")