import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import jieba
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 利用给定语料库，利用1～2 种神经语言模型（如：基于Word2Vec ， LSTM， GloVe等模型）来训练词向量，
# 通过计算词向量之间的语意距离、某一类词语的聚类、某些段落直接的语意关联、或者其他方法来验证词向量的有效性。

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx  # 以字为键
            self.idx2word[self.idx] = word  # 以数值为键
            self.idx += 1

class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()  # 继承类，初始化映射表
        self.file_list = []

    def get_file(self, filepath):
        for root, path, fil in os.walk(filepath):
            for txt_file in fil:
                self.file_list.append(root + txt_file)
        return self.file_list

    def get_data(self, batch_size):  # 读取文件，导入映射表
        # step 1
        tokens = 0
        for path in self.file_list:
            print(path)
            with open(path, 'r', encoding="ANSI") as f:
                for line in f.readlines():
                    # 把一些无意义的空格、段落符给去掉
                    line = line.replace(' ', '')
                    line = line.replace('\u3000', '')
                    line = line.replace('\t', '')
                    # jieba
                    words = jieba.lcut(line) + ['<eos>']
                    tokens += len(words)
                    for word in words:  # 构造彼此映射的关系
                        self.dictionary.add_word(word)
        # step 2
        ids = torch.LongTensor(tokens)  # 实例化一个LongTensor，命名为ids。遍历全部文本，根据映射表把单词转成索引，存入ids里
        token = 0
        for path in self.file_list:
            with open(path, 'r', encoding="ANSI") as f:
                for line in f.readlines():
                    line = line.replace(' ', '')
                    line = line.replace('\u3000', '')
                    line = line.replace('\t', '')
                    words = jieba.lcut(line) + ['<eos>']
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]  # 把每个词对应的索引存在ids里
                        token += 1
        # step 3 根据batchsize重构成一个矩阵
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        ids = ids.view(batch_size, -1)
        return ids

class LSTMmodel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMmodel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # 单词总数，每个单词的特征个数
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # 单词特征数，隐藏节点数，隐藏层数
        self.linear = nn.Linear(hidden_size, vocab_size)  # 全连接层

    def forward(self, x, h):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out = self.linear(out)
        return out, (h, c)

'''训练'''
embed_size = 256#增加每个词涵盖的特征数，提高结果精准度
hidden_size = 1024#增加神经元数量
num_layers = 3#增加隐藏层
num_epochs = 10#增加训练次数
batch_size = 50
seq_length = 30  # 序列长度，我认为是与前多少个词具有相关程度
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
corpus = Corpus()  # 构造实例
corpus.get_file('./book1/')
ids = corpus.get_data(batch_size)  # 获得数据
my_dict = corpus.dictionary.word2idx

vocab_size = len(corpus.dictionary)  # 词总数

flag = 1

if flag:
    model = LSTMmodel(vocab_size, embed_size, hidden_size, num_layers).to(device)

    cost = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, hidden_size).to(device))  # 参数矩阵初始化(h,c)

        for i in tqdm(range(0, ids.size(1) - seq_length, seq_length)):  # 打印循环中的进度条
            inputs = ids[:, i:i + seq_length].to(device)  # 训练集的输入
            targets = ids[:, (i + 1):(i + 1) + seq_length].to(device)  # 训练集的结果

            states = [state.detach() for state in states]
            # detach返回一个新的tensor，相当于可以切断反向传播的计算
            outputs, states = model(inputs, states)
            loss = cost(outputs, targets.reshape(-1))

            model.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            clip_grad_norm_(model.parameters(), 0.5)  # 避免梯度爆炸
            optimizer.step()
    '''保存模型'''
    save_path = './model_path/model.pt'
    torch.save(model, save_path)
else:
    model = torch.load('./model_path/model.pt')

embedding_weights = model.embed.weight
# print(embedding_weights[my_dict['刀']])
def cosine_similarity_tensor(tensor_a, tensor_b):
    # 将tensor转换为numpy数组
    np_a = tensor_a.detach().cpu().numpy()
    np_b = tensor_b.detach().cpu().numpy()
    np_a = np_a[:len(np_a) // 4]
    np_b = np_b[:len(np_b) // 4]
    # 计算余弦相似度
    dot_product = np.dot(np_a, np_b)
    norm_a = np.linalg.norm(np_a)
    norm_b = np.linalg.norm(np_b)
    if norm_a == 0 or norm_b == 0:
        return 0  # 如果向量是零向量，则余弦相似度为0
    similarity = dot_product / (norm_a * norm_b)
    return similarity

para = ['掌门', '帮主']
print(f"比较词“{para[0]}”和词“{para[1]}”的余弦相似度为:{cosine_similarity_tensor(embedding_weights[my_dict[para[0]]], embedding_weights[my_dict[para[1]]])}")

first_1000_kv_pairs = list(my_dict.items())[1000:2000]
out = []
# 打印结果
for key, value in first_1000_kv_pairs:
    out.append(value)
word_vectors = []
for i in range(1000):
    np = embedding_weights[out[i]].detach().cpu().numpy()
    word_vectors.append(np)
kmeans = KMeans(n_clusters=5)
kmeans.fit(word_vectors)

# 可视化（简化展示）
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(word_vectors)
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=kmeans.labels_)
plt.show()