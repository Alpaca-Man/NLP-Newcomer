import xlrd
import jieba
import torch

'''
@doc: 数据处理类,把字符串们转换成符合模型标准输入的格式
@author: Alpaca-Man
@date: 2021/2/10
'''

# 默认参数
fileName = 'D:\\项目组\\模型\\data\\usual_train.xlsx' # 训练集路径
dtype = torch.LongTensor # 数据类型
device = torch.device("cuda" if not torch.cuda.is_available() else "cpu") # 运算设备

class DataProcessor():
	def __init__(self, fileName, seq_len):
		self.fileName = fileName # 文件路径
		self.voc_size = 0 # 单词个数
		self.word2idx = [] # 单词引索
		self.label_types = [] # 标签种类
		self.corpusContents = [] # 所有语句（已分词）
		self.corpusLabels = [] # 所有标签
		self.seq_len = seq_len # 每条语句的长度

	# 用于从 .xlsx 文件中提取出语句（已分词）和标签并返回
	def preTreatMent(self):
		'''
		@doc: 该函数用于从 .xlsx 文件中提取出语句（已分词）和标签
		@author: Alpaca-Man
		@date: 2021/2/9
		@param: { }
		@return: {
					contents: [语句数量, 每条句子各自的长度] dtype = list
					corpusLabels: [语句数量] dtype = list
				}
		'''
		corpus = xlrd.open_workbook(self.fileName)
		sheet1 = corpus.sheets()[0]

		corpusContents = sheet1.col_values(1)  # 所有语句
		corpusContents.remove('文本')
		corpusLabels = sheet1.col_values(2)  # 所有标签
		corpusLabels.remove('情绪标签')

		# 获取所有语句的分词版本
		contents = []
		for sen in corpusContents:
			contents.append([i for i in jieba.cut(sen, cut_all=True)])

		self.label_types = list(set(corpusLabels)) # 获得所有标签类型
		self.corpusContents = contents # 获得所有语句（已分词）
		self.corpusLabels = corpusLabels # 获得所有标签

		print('生成单词表及索引')
		vocab = []
		for i in self.corpusContents:
			vocab.extend(i)
		vocab = list(set(vocab))
		self.voc_size = len(vocab)
		self.word2idx = dict(zip(vocab, range(self.voc_size)))
		print('单词表大小为:{}'.format(self.voc_size))

		print('数据集的总大小为:{}'.format(len(contents)))

		return contents, corpusLabels


	# 用于生成训练集，返回训练语句和标签
	def train(self, index):
		'''
		@doc: 该函数用于根据 KFold 折出的引索提取生成 train 语句和标签
		@author: Alpaca-Man
		@date: 2021/2/9
		@param: {
					index: KFold 折出的 train 引索
				}
		@return: {
					trainInput: [语句数量, seq_len] dtype = torch.LongTensor
					trainTarget: [语句数量] dtype = torch.LongTensor
				}
		'''
		trainInput = []
		for i in index:
			temp = torch.zeros(self.seq_len).to(device)
			for j in range(self.seq_len):
				if j < len(self.corpusContents[i]):
					temp[j] = self.word2idx[self.corpusContents[i][j]]
				else:
					temp[j] = self.word2idx['']
			trainInput.append(temp)

		trainInput = dtype([item.cpu().detach().numpy() for item in trainInput]).to(device) # 语句转换成指定数据类型
		trainTarget = []
		for i in index:
			temp = self.label_types.index(self.corpusLabels[i])
			trainTarget.append(temp)

		trainTarget = dtype(trainTarget).to(device) # 标签转换成指定数据类型
		# trainTarget = dtype([item.cpu().detach().numpy() for item in trainTarget]).to(device) # 标签转换成指定数据类型

		print('-' * 20)
		print('train语句维度为{}'.format(trainInput.shape))
		print('train标签维度为{}'.format(trainTarget.shape))
		print('-' * 20)

		return trainInput, trainTarget


	# 用于生成测试集，返回测试语句和标签
	def test(self, index):
		'''
		@doc: 该函数用于根据 KFold 折出的引索提取生成 test 语句和标签
		@author: Alpaca-Man
		@date: 2021/2/9
		@param: {
					index: KFold 折出的 test 引索
				}
		@return: {
					testInput: [语句数量, seq_len] dtype = torch.LongTensor
					testTarget: [语句数量, label_types] dtype = torch.LongTensor
				}
		'''
		testInput = []
		for i in index:
			temp = torch.zeros(self.seq_len).to(device)
			for j in range(self.seq_len):
				if j < len(self.corpusContents[i]):
					temp[j] = self.word2idx[self.corpusContents[i][j]]
				else:
					temp[j] = self.word2idx['']
			testInput.append(temp)

		testInput = dtype([item.cpu().detach().numpy() for item in testInput]).to(device)  # 语句转换成指定数据类型

		testTarget = []
		for i in index:
			temp = torch.zeros(len(self.label_types)).to(device)
			temp[self.label_types.index(self.corpusLabels[i])] = 1
			testTarget.append(temp)

		testTarget = dtype([item.cpu().detach().numpy() for item in testTarget]).to(device)  # 标签转换成指定数据类型

		print('-' * 20)
		print('test语句维度为{}'.format(testInput.shape))
		print('test标签维度为{}'.format(testTarget.shape))
		print('-' * 20)

		return testInput, testTarget


