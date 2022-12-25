# 实训三 文本向量化（基于 word2vec 方法）
from gensim.models import word2vec
import logging
import math


def get_sim(vec1, vec2):  # 计算两个向量的余弦相似度
    n = len(vec1)
    a, b, c = 0, 0, 0
    for i in range(n):
        a += vec1[i] * vec2[i]
        b += vec1[i] * vec1[i]
        c += vec2[i] * vec2[i]
    return a / (math.sqrt(b) * math.sqrt(b))


def getmodel():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    path = 'C:/Users/Ohh/Desktop/实训数据集/第7章/'
    sentences = word2vec.LineSentence(path + 'word2vec_train_words.txt')
    model = word2vec.Word2Vec(sentences, hs=0, min_count=3, window=3)

    return model


def test():
    model = getmodel()
    news1 = 'C:/Users/Ohh/Desktop/实训数据集/第7章/amnews_key.txt'
    news2 = 'C:/Users/Ohh/Desktop/实训数据集/第7章/plnews_key.txt'
    news1_read = open(news1, 'r', encoding='utf-8')
    news1_list = news1_read.readlines()
    news1_read.close()
    news1_words_vec = []
    for line in news1_list:
        line = line.split(' ')
        if line[-1] == '\n':
            line = line[:-1]
        for w in line:
            if w in model.wv.key_to_index:
                news1_words_vec.append(model.wv[w])  # 将关键词转换为词向量

    news2_read = open(news2, 'r', encoding='utf-8')
    news2_list = news2_read.readlines()
    news2_read.close()
    news2_words_vec = []
    for line in news2_list:
        line = line.split(' ')
        if line[-1] == '\n':
            line = line[:-1]
        for w in line:
            if w in model.wv.key_to_index:
                news2_words_vec.append(model.wv[w])

    news1_vec = sum(news1_words_vec)  # 将关键词的词向量相加
    news2_vec = sum(news2_words_vec)

    sim = get_sim(news1_vec, news2_vec)  # 求出两个向量的余弦相似度
    print(sim)
    return sim
