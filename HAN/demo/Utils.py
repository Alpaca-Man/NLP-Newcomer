
'''
@doc: 工具类,用于模型训练，测试
@author: Alpaca-Man
@date: 2021/2/10
'''

# 该函数用于根据给定的条件训练模型
def train(model, epoch, dataLoader, criterion, optimizer):
	'''
	@doc: 该函数用于根据给定的条件训练模型
	@author: Alpaca-Man
	@date: 2021/2/9
	@param: {
				model: 神经网络
				epoch: 训练轮次
				dataLoader: 数据装载器
				criterion: 损失函数
				optimizer: 优化器
			}
	@return:{ }
	'''
	print('-' * 20)
	print('训练开始了哟')
	for i in range(epoch):
		for batchX, batchY in dataLoader:
			pred = model(batchX)
			loss = criterion(pred, batchY)
			if (i + 1) % 10 == 0:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
		if (i + 1) % 100 == 0:
			print('Epoch:{}  Loss:{:.6f}'.format(i + 1, loss))
	print('训练结束了哟')
	print('-' * 20)


# 该函数用于评判模型性能
def test(model, dataLoader):
	'''
	@doc:
	@author: Alpaca-Man
	@date: 2021/2/9
	@param: {
				model: 神经网络
				dataLoader: 测试集装载器
			}
	@return: {
				scores: 模型得分
			}
	'''
	model = model.eval() # 需要调为测试模式，不启用 BatchNormalization 和 Dropout
	acc = 0	# 预测正确的句子数量
	sum = 0 # 测试集总数
	for batchX, batchY in dataLoader:
		num = batchX.shape[0]
		sum += num
		predict = model(batchX).data.max(1, keepdim=True)[1]
		batch_y = batchY.data.max(1, keepdim=True)[1]
		for i in range(num):
			if predict[i][0] == batch_y[i][0]: # 预测值和真实值的最大引索相同则则正确
				acc += 1

		scores = acc / sum
		return scores
