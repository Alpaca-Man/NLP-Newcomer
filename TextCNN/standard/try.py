import torch
i = 999
loss = 222.2232323
print('Epoch:{}  Loss:{}'.format(i + 1,loss))

sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.
word_list = " ".join(sentences).split() # 全部单词（重复）
vocab = list(set(word_list)) # 单词表（不重复）
word2idx = {w: i for i, w in enumerate(vocab)} # 单词索引
vocab_size = len(vocab) # 单词个数
test_text = 'i hate me'
tests = [[word2idx[n] for n in test_text.split()]]
test_batch = torch.LongTensor(tests).to(torch.device("cuda"))
print(tests)
print(test_batch.shape)