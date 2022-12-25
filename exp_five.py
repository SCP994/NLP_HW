# 实训五 文本情感分析
# 1. 基于情感词典的情感分析实战
from collections import defaultdict
import jieba
import codecs
from functools import reduce
import operator
from numpy import zeros
import pandas as pd
from sklearn.utils import shuffle


path = 'C:/Users/Ohh/Desktop/实训数据集/实训专题5/'
def seg_word(sentence):  # 使用jieba对文档分词
    seg_list = jieba.cut(sentence)
    seg_result = []
    for w in seg_list:
        seg_result.append(w)
    # 读取停用词文件
    stopwords = set()
    fr = codecs.open(path + 'stoplist.txt', 'r', 'utf-8')
    for word in fr:
        stopwords.add(word.strip())
    fr.close()
    # 去除停用词
    return list(filter(lambda x: x not in stopwords, seg_result))


def classify_words(word_dict):  # 词语分类,找出情感词、否定词、程度副词
    # 读取情感字典文件
    sen_file = open(path + 'BosonNLP_sentiment_score.txt', 'r+', encoding='utf-8')
    # 获取字典文件内容
    sen_list = sen_file.readlines()
    # 创建情感字典
    sen_dict = defaultdict()
    # 读取字典文件每一行内容，将其转换为字典对象，key为情感词，value为对应的分值
    for s in sen_list:
        # 每一行内容根据空格分割，索引0是情感词，索引1是情感分值
        if s == '\n':
            continue
        s = s.split(' ')
        sen_dict[s[0]] = s[1]

    # 读取否定词文件
    not_word_file = open(path + '否定词.txt', 'r+', encoding='utf-8')
    # 由于否定词只有词，没有分值，使用list即可
    not_word_list = not_word_file.readlines()

    # 读取程度副词文件
    degree_file = open(path + '程度副词（中文）.txt', 'r+', encoding='utf-8')
    degree_list = degree_file.readlines()
    degree_dic = defaultdict()
    # 程度副词与情感词处理方式一样，转为程度副词字典对象，key为程度副词，value为对应的程度值
    for d in degree_list:
        if d == '\n':
            continue
        d = d.split(' ')
        degree_dic[d[0]] = d[1]

    # 分类结果，词语的index作为key,词语的分值作为value，否定词分值设为-1
    sen_word = dict()
    not_word = dict()
    degree_word = dict()

    # 分类
    for word in word_dict.keys():
        if word in sen_dict.keys() and word not in not_word_list and word not in degree_dic.keys():
            # 找出分词结果中在情感字典中的词
            sen_word[word_dict[word]] = sen_dict[word]
        elif word in not_word_list and word not in degree_dic.keys():
            # 分词结果中在否定词列表中的词
            not_word[word_dict[word]] = -1
        elif word in degree_dic.keys():
            # 分词结果中在程度副词中的词
            degree_word[word_dict[word]] = degree_dic[word]
    sen_file.close()
    degree_file.close()
    not_word_file.close()
    # 将分类结果返回
    return sen_word, not_word, degree_word


def list_to_dict(word_list):
    data = {}
    for x in range(0, len(word_list)):
        data[word_list[x]] = x
    return data


def get_init_weight(sen_word, not_word, degree_word):
    # 权重初始化为1
    W = 1
    # 将情感字典的key转为list
    sen_word_index_list = list(sen_word.keys())
    if len(sen_word_index_list) == 0:
        return W
    # 获取第一个情感词的下标，遍历从0到此位置之间的所有词，找出程度词和否定词
    for i in range(0, sen_word_index_list[0]):
        if i in not_word.keys():
            W *= -1
        elif i in degree_word.keys():
            # 更新权重，如果有程度副词，分值乘以程度副词的程度分值
            W *= float(degree_word[i])
    return W


def socre_sentiment(sen_word, not_word, degree_word, seg_result):  # 计算得分
    # 权重初始化为1
    W = 1
    score = 0
    # 情感词下标初始化
    sentiment_index = -1
    # 情感词的位置下标集合
    sentiment_index_list = list(sen_word.keys())
    for i in range(0, len(seg_result)):
        if i in sen_word.keys():
            score += W * float(sen_word[i])
            sentiment_index += 1
            if sentiment_index < len(sentiment_index_list) - 1:
                for j in range(sentiment_index_list[sentiment_index], sentiment_index_list[sentiment_index + 1]):
                    if j in not_word.keys():
                        W *= -1
                    elif j in degree_word.keys():
                        W *= float(degree_word[j])
                if sentiment_index < (len(sentiment_index_list)) - 1:
                    i = sentiment_index_list[sentiment_index + 1]
    return score


def setiment_score():
    sentence = '煽情显得太尴尬'
    # 1.对文档分词
    seg_list = seg_word(sentence)
    # 2.将分词结果列表转为dic，然后找出情感词、否定词、程度副词
    sen_word, not_word, degree_word = classify_words(list_to_dict(seg_list))
    # 3.计算得分
    score = socre_sentiment(sen_word, not_word, degree_word, seg_list)
    return score


# 2. 基于朴素⻉叶斯算法的⾖瓣影评⽂本情感分析
def trainNB():
    stopwords = set()
    fr = codecs.open(path + 'stoplist.txt', 'r', 'utf-8')
    for word in fr:
        stopwords.add(word.strip())
    fr.close()

    df = pd.read_csv(path + '流浪地球.csv')
    df = shuffle(df).reset_index(drop=True)  # 打乱数据，并丢弃旧索引
    data = df.values.tolist()
    vocablist = []
    docs = []
    docsCategory = []

    for line in data:
        sentence = line[1]
        seg_list = jieba.cut(sentence)
        seg_result = []
        for w in seg_list:
            seg_result.append(w)
        seg_result = list(filter(lambda x: x not in stopwords and x != '\r\n' and x != ' ', seg_result))
        docs.append(seg_result)
        for w in seg_result:
            if w not in vocablist:
                vocablist.append(w)
        if line[2] == '推荐' or line[2] == '力荐':  # 正面文档为 0，负面文档为 1
            docsCategory.append(0)
        else:
            docsCategory.append(1)

    docs_matrix = []
    for line in docs:
        vec = [0] * len(vocablist)
        for w in line:
            if w in vocablist:
                vec[vocablist.index(w)] = 1
        docs_matrix.append(vec)

    numTrainDocs = int(len(data) * 9 / 10)
    numWords = len(vocablist)
    pAbusive = sum(docsCategory) / float(numTrainDocs)
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0
    p1Denom = 0
    for i in range(numTrainDocs):
        if docsCategory[i] == 1:
            p1Num += docs_matrix[i]
            p1Denom += sum(docs_matrix[i])
        else:
            p0Num += docs_matrix[i]
            p0Denom += sum(docs_matrix[i])
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom

    countAll = 0
    count = 0

    for i in range(numTrainDocs, len(data), 1):
        p1VectClassify = docs_matrix[i] * p1Vect
        p0VectClassify = docs_matrix[i] * p0Vect
        p1Cond = []
        p0Cond = []
        for j in range(len(p0VectClassify)):
            if p1VectClassify[j] == 0:
                continue
            else:
                p1Cond.append(p1VectClassify[j])

        for j in range(len(p0VectClassify)):
            if p0VectClassify[j] == 0:
                continue
            else:
                p0Cond.append(p0VectClassify[j])

        if len(p0Cond):
            pC0 = reduce(operator.mul, p0Cond, 1)
        else:
            pC0 = 0
        if len(p1Cond):
            pC1 = reduce(operator.mul, p1Cond, 1)
        else:
            pC1 = 0

        p1 = pC1 * pAbusive
        p0 = pC0 * (1.0 - pAbusive)

        countAll += 1
        type = 1
        if p1 < p0:
            type = 0
        if type == docsCategory[i]:
            count += 1

    print(count / countAll)
