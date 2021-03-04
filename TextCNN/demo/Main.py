import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.model_selection import KFold

from DataProcessor import DataProcessor
import Utils
from TextCNN import TextCNN

'''
@doc: 主函数
@author: Alpaca-Man
@date: 2021/2/10
'''

# 参数
fileName = 'D:\\项目组\\模型\\data\\usual_train.xlsx' # 训练集路径

dtype = torch.LongTensor # 数据类型
device = torch.device("cuda" if not torch.cuda.is_available() else "cpu") # 运算设备

splits = 10 # 折数
shuffle = True # 是否打乱

# TextCNN 的参数
batch_size = 64 # 批次大小
seq_len = 60 # 每条语句的长度
emb_size = 64 # 词嵌入维度
in_channels = 1 # 输入的通道数
out_channels = 6 # 卷积输出的通道数
lr = 1e-3 # 学习率
epoch = 100 # 训练次数

# 加载处理数据集
dataProcessor = DataProcessor(fileName, seq_len)
corpusContents, corpusLabels = dataProcessor.preTreatMent()

# TextCNN
model = TextCNN(dataProcessor.voc_size, emb_size, in_channels, out_channels).to(device)
criterion = nn.CrossEntropyLoss().to(device)# 损失函数
optimizer = optim.Adam(model.parameters(), lr = lr) # 优化器

kf = KFold(n_splits=splits, shuffle=shuffle, random_state=None)
fold = 0

# 保留最好
bestFold = 1
bestScores = 0

for train,test in kf.split(corpusContents):
	fold += 1
	print("第{}折开始".format(fold))
	trainContents, trainTargets = dataProcessor.train(train)
	testContents, testTargets = dataProcessor.test(test)
	# train, test 分别封装成数据集迭代器
	trainDataSet = Data.TensorDataset(trainContents, trainTargets)
	trainDataLoader = Data.DataLoader(trainDataSet, batch_size=batch_size, shuffle=shuffle)  # 打乱，也可以不打乱，因为 KFold 已经打乱了
	testDataSet = Data.TensorDataset(testContents, testTargets)
	testDataLoader = Data.DataLoader(testDataSet, batch_size=batch_size, shuffle=False)  # 不用打乱

	# 训练并测试
	Utils.train(model, epoch, trainDataLoader, criterion, optimizer)
	scores = Utils.test(model, testDataLoader)
	if scores > bestScores:
		bestScores = scores
		bestFold = fold
	print('第{}折的精确度为{}'.format(fold, scores))
	print('缓冲一下')

print('结束了，最好的模型是第{}折的，准确率为:{}%'.format(bestFold,bestScores * 100))