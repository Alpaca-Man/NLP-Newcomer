
'''
@doc: 工具类,用于模型训练，测试
@author: Alpaca-Man
@date: 2021/2/10
'''

# 用于把数据集处理成数组
def makeData(sentences, word2idx, labels):
    '''
    @doc: 该函数用于根据给定的条件训练模型
    @author: Alpaca-Man
    @date: 2021/2/9
    @param: {
        		sentences: 所有语句的字符串列表
        		word2idx: 单词表对应的引索
        		labels: 所有语句对应的标签列表
        	}
    @return:{ }
    '''
    inputs = []
    for sen in sentences:
        inputs.append([word2idx[n] for n in sen.split()])

    targets = []
    for out in labels:
        targets.append(out) # To using Torch Softmax Loss function
    return inputs, targets


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
    for i in range(epoch):
        for batchX, batchY in dataLoader:
            pred = model(batchX)
            loss = criterion(pred, batchY)
            if (i + 1) % 1000 == 0:
                print('Epoch:{}  Loss:{:.6f}'.format(i + 1,loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# 用于模型验证
def valid(model, dataLoader):
    pass


# 用于模型测试
def test(model, dataLoader):
    pass