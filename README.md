# NLP-Newcomer

## Introduction

NLP Classical Deep Learning Model (pytorch)


​	这是一个用 python 语言编写，调用深度学习库 pytorch 实现的各类 NLP 经典深度学习模型项目，同时也是一个记录我 NLP 入门之路的项目，新人们在学习有关模型时遇到代码困难时可以参考这个项目。我总是认为开源是一种非常 cool 的想法，即使我现在的能力不够不足以为开源社区做出多大贡献，但是只要这个项目能为想要了解学习 NLP 的同学带来一点帮助，我想也足够了。

​	This is a Classic NLP Deep Learning Model project written in python and implemented by calling the deep learning library pytorch. It is also a project that records my entry to NLP. It is also a project to record my entry to NLP and share it with novices for reference.I always think that the spirit of open source is a very good idea. Even if my current ability is not enough to make much contribution to the open source community, as long as this project can bring a little help to students who want to learn about NLP, I think it will be enough.



## Project Description

### 1 project contents

#### 1.1 Frame

- root
  - (models)
    - demo
    - standard
  - data
  - README.md



#### 1.2 Detail

##### 1.2.1 model

* TextCNN
* RNN
* LSTM
* GRU
* BiLSTM
* HAN
* Transformer



##### 1.2.2 standard

**Main**

主函数，模型在这里被调用

The main function, the model is used here



**Model**

模型，提供各类模型的具体实现细节封装

Models, provide specific implementation details encapsulation of various models



**try**

测试代码，为避免与 test 引起误解故命名为 try

The test code is named _try_ to avoid misunderstanding with _test_



**Utils**

工具包，提供各类需要的功能

Toolkit, providing various required functions



##### 1.2.3 demo

**Main**

主函数，模型在这里被调用

The main function, the model is used here



**Model**

模型，提供各类模型的具体实现细节封装

Models, provide specific implementation details encapsulation of various models



**DataProcessor**

数据预处理，对大数据集的数据进行清洗提取

Data preprocessing, clean and extract data from large data sets



**Utils**

工具包，提供各类需要的功能

Toolkit, providing various required functions



**try**

测试代码，为避免与 test 引起误解故命名为 try

The test code is named _try_ to avoid misunderstanding with _test_



**Math (Transformer Unique)**

数学运算，Transformer 中各类数学运算由该文件中的类与函数实现

Mathematical operations, various mathematical operations in Transformer are implemented by the classes and functions in this file



### 2 Environment

* python  3.7.4
* pytorch  1.2.0
* numpy  1.16.5
* jieba  0.38
* xlrd  1.2.0
* sklearn  0.21.3



## Author's point of view

### 1 Advantage

​	该项目提供了各类经典深度学习模型的 pytorch 代码实现，对于每个模型，提供了 `standard `标准的、简洁的模型使用样例，并且把每一个功能都单独封装出来，避免了代码冗余过长不便于理解的问题。

​	该项目还提供了一份大数据集和使用代码 `demo` 供使用者学习使用（该数据集是通过网络爬虫爬取所得，如有侵权请联系作者删除），使用者可以利用该数据集来评判模型的性能好坏或实现自己的想法。

​	对于每一份代码，作者都提供了详细的代码注释，对变量的维度变化也详细记录在旁，便于学习者感受理解。



​	The project provides pytorch code implementations of various classic deep learning models. For each model, it provides standard and concise model usage examples, and each function is individually encapsulated to avoid code redundancy.Avoid incomprehensible problems caused by code redundancy.

​	The project also provides a large data set and use code `demo` for users to learn and use (the data set is crawled through web crawlers, if there is any infringement, please contact the author to delete), users can use the data set to Judge the performance of the model or realize your own ideas.

​	For each piece of code, the author provides detailed code comments, and the dimensional changes of the variables are also recorded in detail, which is convenient for learners to feel and understand.



### 2 Problem

​	注释和变量命名风格随着作者学习进程会出现不同，早期变量命名风格采用 `功能_属性` 的格式（例如 _batch_size_, _emb_size_），后期命名风格为小驼峰命名法（例如 _batchSize_, _embSize_）

​	Github 对于 `Tab` 缩进有不同处理，作者不知道怎么处理，所以代码在官网上显示出来会有点问题。

`	demo` 中 `DataProcessor.py` 文件中使用的是绝对路径，使用者下载后只需要改动 _fileName_ 这一全局变量就行



​	The annotation and variable naming style will vary with the author’s learning process. The early variable naming style adopts the `function_attribute` format (such as _batch_size_, _emb_size_), and the later naming style is the little camel case (such as _batchSize_, _embSize_)

​	Github has different treatments for `Tab` indentation, the author does not know how to deal with it, so the code displayed on the official website will be a little problematic.

​	The absolute path is used in the `DataProcessor.py` file in `demo`, and the user only needs to change the global variable _fileName_ after downloading.



## Scores

* TextCNN 4th fold 95%
* RNN 7th fold 60%
* LSTM 4th fold 100%
* GRU 4th fold 100%
* BiLSTM 6th fold 100%
* HAN 9th fold 100%



## If you have any doubts or suggestions, please feel free to put forward to the author, thanks！

