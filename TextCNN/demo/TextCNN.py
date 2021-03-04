import torch.nn as nn
import torch

'''
@doc: TextCNN模型细节
@author: Alpaca-Man
@date: 2021/2/9
'''

seq_len = 60 # 单条语句长度
num_classes = 6 # 类别数量
k1 = (int)((seq_len - 2) / 2 + 1)
k2 = (int)((seq_len - 2) / 3 + 1)
k3 = (int)((seq_len - 2) / 4 + 1)
kernel1_size = (k1,1) # 池化层核1 (30,1)
kernel2_size = (k2,1) # 池化层核2 (20, 1)
kernel3_size = (k3,1) # 池化层核3 (15, 1)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_size, in_channels, out_channels):
        '''
        @doc: TextCNN 模型细节
        @author: Alpaca-Man
        @date: 2021/2/9
        @param: {
                    vocab_size: 单词个数
                    emb_size: 词嵌入维度
                    in_channels: 输入通道
                    out_channels: 输出通道
                }
        @return: { }
        '''
        super(TextCNN, self).__init__()
        self.emb = nn.Embedding(vocab_size+1, emb_size)
        self.conv_size = (2, emb_size) # 卷积核
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 2, self.conv_size, (2,1)),
            nn.MaxPool2d(kernel1_size),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, 2, self.conv_size, (3,1)),
            nn.MaxPool2d(kernel2_size),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, 2, self.conv_size, (4,1)),
            nn.MaxPool2d(kernel3_size),
            nn.ReLU(),
        )
        # x => [batch_size, num_classes]
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        '''
        @doc: 前向传播函数
        @author: Alpaca-Man
        @date: 2021/2/15
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
        conv1 = self.conv1(x_emb) # 经过卷积层1 [batch_size, out_channels / 3, 1, 1]
        conv2 = self.conv2(x_emb)  # 经过卷积层2 [batch_size, out_channels / 3, 1, 1]
        conv3 = self.conv3(x_emb)  # 经过卷积层3 [batch_size, out_channels / 3, 1, 1]
        conved = torch.cat((conv1,conv2),1)
        conved = torch.cat((conved, conv3), 1) # 拼接 [batch_size, out_channels, 1, 1]
        flatten = conved.view(batch_size, -1) # 展平 [batch_size, out_channels * 1 * 1]
        pred = self.fc(flatten)
        return pred

