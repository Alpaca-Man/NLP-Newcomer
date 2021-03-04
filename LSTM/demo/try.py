import numpy as np
import xlrd
import jieba
from sklearn.model_selection import KFold
import torch
import time

# 默认参数
fileName = 'D:\\项目组\\模型\\data\\usual_train.xlsx' # 训练集路径
splits = 10 # 折数

bestScores = 0.9531235
# bestScores *= 100
print('结束了，最好的模型是第{}折的，准确率为:{}%'.format(1,bestScores * 100))
'''

a = []
for i in range(10):
	temp = torch.zeros(5)
	a.append(temp)

print(a)
a = torch.LongTensor([item.cpu().detach().numpy() for item in a]).cuda()

print(a)
print(type(a))
'''
'''
a = torch.rand(5,1,5)
b = torch.rand(5,1,5)
c = torch.rand(5,1,5)

sum = torch.cat((a,b),1)
sum = torch.cat((sum,c),1)
print(sum.shape)
'''





'''


acc = 0  # 预测正确的句子数量
sum = 0  # 测试集总数

num = a.shape[0]
sum += num
a = a.data.max(1, keepdim=True)[1]
print(a)
b = b.data.max(1, keepdim=True)[1]
print(b)
for i in range(num):
	print(a[i][0], b[i][0])
	if a[i][0] == b[i][0]:
		acc += 1

scores = acc / sum
print(scores)
'''










'''
corpus = xlrd.open_workbook(fileName)

sheet1 = corpus.sheets()[0]

corpusContents = sheet1.col_values(1) # 所有语句
corpusContents.remove('文本')
corpusLabels = sheet1.col_values(2) # 所有标签
corpusLabels.remove('情绪标签')

# 获取所有语句的分词版本
label_types = list(set(corpusLabels))
print(label_types)


contents = []
for sen in corpusContents:
	contents.append([i for i in jieba.cut(sen, cut_all=True)])
corpusContents = contents

vocab = []

for i in corpusContents:
	vocab.extend(i)
vocab = list(set(vocab))

'''







'''
for i in range(10):
	start = time.time()
	word2idx = dict(zip(vocab, range(len(vocab))))
	end = time.time()
	print(list(word2idx.items())[:5])
	print('{}'.format((end - start) * 1000))

	start = time.time()
	word2idx = word2idx = {w: i for i, w in enumerate(vocab)}
	end = time.time()
	print(list(word2idx.items())[:5])
	print('{}'.format((end - start) * 1000))
	if i == 9:
		print('-------')
		start = time.time()
		word2idx = dict(zip(vocab, range(len(vocab))))
		end = time.time()
		print(list(word2idx.items())[:5])
		print('{}'.format((end - start) * 1000))

		start = time.time()
		word2idx = word2idx = {w: i for i, w in enumerate(vocab)}
		end = time.time()
		print(list(word2idx.items())[:5])
		print('{}'.format((end - start) * 1000))

start = time.time()
word2idx = dict(zip(vocab, range(len(vocab))))
end = time.time()
print(list(word2idx.items())[:5])
print('{}'.format((end-start)*1000))

start = time.time()
word2idx = word2idx = {w: i for i, w in enumerate(vocab)}
end = time.time()
print(list(word2idx.items())[:5])
print('{}'.format((end-start)*1000))

'''

'''
kf = KFold(n_splits=splits, shuffle=True, random_state=None)
fold = 0
for train,test in kf.split(corpusContents):
	fold += 1
	print("第{}折".format(fold))
	print(train[0:10])

'''