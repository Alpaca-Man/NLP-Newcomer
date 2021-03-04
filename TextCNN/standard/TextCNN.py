import torch.nn as nn

'''
@doc: TextCNN 模型细节
@author: Alpaca-Man
@date: 2021/2/9
'''

# TextCNN 默认参数
vocab_size = 10 # 单词个数
emb_size = 2 # 词嵌入维度
seq_len = 3 # 每一句的长度
num_classes = 2 # 标签种类数量
batch_size = 3 # 每个批次的大小
conv_size = (2, emb_size) # 卷积核大小
in_channels = 1 # 输入的通道数
out_channels = 3 # 卷积输出的通道数
kernel_size = (2,1) # 池化层核大小

class TextCNN(nn.Module):
    def __init__(self, vocab_size = vocab_size, emb_size = emb_size, in_channels = in_channels, out_channels = out_channels, conv_size = conv_size, kernel_size = kernel_size):
        '''
        @doc: TextCNN 模型细节
        @author: Alpaca-Man
        @date: 2021/2/9
        @param: {
                    vocab_size: 单词个数
                    emb_size: 词嵌入维度
                    in_channels: 输入通道
                    out_channels: 输出通道
                    conv_size: 卷积核大小
                    kernel_size: 池化层核大小
                }
        @return: { }
        '''
        super(TextCNN, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.conv = nn.Sequential(
            # x => [batch_size, out_channels, height1, weight1] (height1 = seq_len - conv_size[0] + 1, weight1 = emb_size - conv_size[1] + 1)
            nn.Conv2d(in_channels, out_channels, conv_size),

            # x => [batch_size, out_channels, height1, weight1]
            nn.ReLU(),

            # x => [batch_size, out_channels, height2, weight2] (height2 = height1 - kernel_size[0] + 1, weight2 = weight1 - kernel_size[1] + 1)
            nn.MaxPool2d(kernel_size),
        )
        # x => [batch_size, num_classes]
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        '''
        @doc: 前向传播函数
        @author: Alpaca-Man
        @date: 2021/2/16
        @param: {
                    x: 输入 [batch_size, seq_len]
                }
        @return: {
                    pred: 预测值 [batch_size, num_classes]
                }
        '''
        batch_size = x.shape[0] # 获取批次大小
        x_emb = self.emb(x) # [batch_size, seq_len, emb_size]
        x_emb = x_emb.unsqueeze(1) # 增加一个维度表示通道数 [batch, in_channels(=1), seq_len, emb_size]
        conved = self.conv(x_emb) # 经过卷积层 [batch_size, out_channels, 1, 1]
        flatten = conved.view(batch_size, -1) # 展平 [batch_size, out_channels * 1 * 1]
        pred = self.fc(flatten)
        return pred

