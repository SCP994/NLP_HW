# 四、词性标注与命名实体识别
# 1. 词性标注
# 1.1 词性标注
def one_one(text):
    import jieba.posseg as psg
    # text=input()
    #任务：使用jieba模块的函数对text完成词性标注并将结果存储到result变量中
    # ********** Begin *********#
    import jieba, logging
    jieba.setLogLevel(logging.INFO)

    generator = psg.cut(text)
    result = ''
    for i, j in generator:
        result += i + '/' + j + ' '

    # ********** End **********#
    print(result)


def test_one_one():
    text = '还有什么是比jieba更好的中文分词工具吗？'
    one_one(text)


# 2. 命名实体识别
# 2.1 命名实体识别
# 2.2 中文人名识别
def two_two():
    from pyhanlp import HanLP
    text = input()
    # 任务：完成对 text 文本的人名识别并输出结果
    # ********** Begin *********#
    segment = HanLP.newSegment().enableNameRecognize(True)
    result = segment.seg(text)
    print(result)

    # ********** End **********#


def test_two_two():
    text = '张三今天没来上课'
    two_two(text)


# 2.3 地名识别
def two_three(text):
    from pyhanlp import HanLP
    # text = input()
    # 任务：完成对 text 文本的地名识别并输出结果
    # ********** Begin *********#
    segment = HanLP.newSegment().enablePlaceRecognize(True)
    result = segment.seg(text)
    print(result)

    # ********** End **********#


def test_two_three():
    text = '中国是个好地方'
    two_three(text)


# 四、关键词提取算法
# 1. TF / IDF 算法
# 1.1 去除停用词
# 停用词表加载方法
def get_stopword_list():
    # 停用词表存储路径，每一行为一个词，按行读取进行加载
    # 进行编码转换确保匹配准确率
    stop_word_path = './stopword.txt'
    stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path,encoding='utf-8').readlines()]
    return stopword_list


def one_one_():
    text = input()
    result = ""
    # 任务：使用停用词表去掉text文本中的停用词，并将结果保存至result变量
    # ********** Begin *********#
    stopword_list = get_stopword_list()
    l = len(stopword_list)
    len_list = [0] * l
    for i in range(l):
        len_list[i] = len(stopword_list[i])
    ma = max(i for i in len_list)

    start = l
    while start > 0:
        for i in range(1, ma + 1, 1):
            if start - i < 0:
                continue
            str = text[start-i:start]
            if str in stopword_list:
                text = text[0:start-i] + text[start:]
                l -= i
                start += 1
                break
        start -= 1
    result = text

    # stopword_list=get_stopword_list()  # 只考虑单个字的情况，不正确
    # for s in  text:
    #     if s not in stopword_list:
    #         result+=s

    # ********** End **********#

    print(result, end="")


# 1.2 TF / IDF 算法
# 本程序的作用是通过TF/IDF算法完成对文本的关键词提取，输出前十个关键词。
import math
import jieba
import jieba.posseg as psg
from gensim import corpora, models
from jieba import analyse
import functools


class TfIdf(object):
    # 四个参数分别是：训练好的idf字典，默认idf值，处理后的待提取文本，关键词数量
    def __init__(self, idf_dic, default_idf, word_list, keyword_num):
        self.word_list = word_list
        self.idf_dic, self.default_idf = idf_dic, default_idf
        self.tf_dic = self.get_tf_dic()
        self.keyword_num = keyword_num

    def get_tf_dic(self):
        tf_dic = {}
        # 任务：完成word_list的tf值的统计函数，将结果存储到tf_dic变量中
        # ********** Begin *********#
        for word in self.word_list:
            if word not in tf_dic:
                tf_dic[word] = 1
            else:
                tf_dic[word] += 1

        # ********** End **********#
        return tf_dic

    # 按公式计算tf-idf
    def get_tfidf(self):
        tfidf_dic = {}
        for word in self.word_list:
            idf = self.idf_dic.get(word, self.default_idf)
            tf = self.tf_dic.get(word, 0)

            tfidf = tf * idf
            tfidf_dic[word] = tfidf

        tfidf_dic.items()
        # 根据tf-idf排序，去排名前keyword_num的词作为关键词
        for k, v in sorted(tfidf_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            print(k + "/ ", end='')
        print()


# 排序函数，用于topK关键词的按值排序
def cmp(e1, e2):
    import numpy as np
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1


# 2. TextRank 算法
# 2.1 Jieba 在关键词提取中的应用
def two_one_():
    import jieba.analyse
    import warnings
    warnings.filterwarnings("ignore")
    sentence = input()

    # 任务：基于jieba中的TF-IDF算法完成对sentence的关键词提取，提取前三个关键词并以一行输出
    # ********** Begin *********#
    import logging
    jieba.setLogLevel(log_level=logging.INFO)

    # TF / IDF 算法
    # sentence 为待提取的文本；
    # topK 为返回几个 TF / IDF 权重最大的关键词，默认值为20；
    # withWeight 为是否一并返回关键词权重值，默认值为 False；
    # allowPOS 仅包括指定词性的词，默认值为空，即不筛选。
    keywords = jieba.analyse.extract_tags(sentence, topK=3, withWeight=True, allowPOS=('n', 'nr', 'ns'))
    for item in keywords:
        print(item[0], end=" ")

    # TextRank 算法
    ret = jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=("ns", "n", "vn", "v"))
    print(' '.join(ret), end='')

    # ********** End **********#


# 3. LSA / LSI 算法
# 3.1 学会使用 Gensim
def three_one():
    from gensim import corpora, models
    import jieba.posseg as jp, jieba
    from basic import get_stopword_list

    texts = []
    # 构建语料库
    for i in range(5):
        s = input()
        texts.append(s)

    flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')  # 词性
    stopwords = get_stopword_list()
    words_ls = []
    for text in texts:
        words = [word.word for word in jp.cut(text) if word.flag in flags and word.word not in stopwords]
        words_ls.append(words)

    # 去重，存到字典
    dictionary = corpora.Dictionary(words_ls)
    corpus = [dictionary.doc2bow(words) for words in words_ls]

    # 任务:基于 gensim 的models构建一个lda模型，主题数为1个
    # ********** Begin *********#
    # corpus 是一个返回 bow 向量的迭代器
    # TF-IDF 模型
    # tfidf = models.TfidfModel(corpus)

    # LSI 模型
    # tfidf_corpus 参数代表 tf-idf 模型生成的统计量；id2word 参数代表词袋向量；num_topics 表示选取的主题词个数。
    # lsi = models.LsiModel(tfidf_corpus, id2word=dictionay, num_topics=1)

    # LDA 模型
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=1)

    # ********** End **********#
    for topic in lda.print_topics(num_words=1):
        print(topic[1].split('*')[1], end="")


# 3.2 LSA / LSI 算法
from gensim import corpora, models
import functools
# from others import seg_to_list, load_data, word_filter, cmp
import math


class TopicModel(object):
    # 三个传入参数：处理后的数据集，关键词数量，具体模型（LSI、LDA），主题数量
    def __init__(self, doc_list, keyword_num, model='LSI', num_topics=4):
        # 使用gensim的接口，将文本转为向量化表示
        # 先构建词空间
        self.dictionary = corpora.Dictionary(doc_list)
        # 任务：使用BOW模型进行向量化，并保存到corpus变量中
        # ********** Begin *********#
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]

        # ********** End **********#
        # 对每个词，根据tf-idf进行加权，得到加权后的向量表示
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]
        self.keyword_num = keyword_num
        self.num_topics = num_topics
        # 选择加载的模型
        if model == 'LSI':
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()
        # 得到数据集的主题-词分布
        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    def train_lsi(self):
        lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lsi

    def train_lda(self):
        lda = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lda

    def get_wordtopic(self, word_dic):
        wordtopic_dic = {}
        for word in word_dic:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

    # 计算词的分布和文档的分布的相似度，取相似度最高的keyword_num个词作为关键词
    def get_simword(self, word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]

        # 余弦相似度计算
        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):  # l1, l2 都为向量
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1  # ？应该写错了
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim

        # 计算输入文本和每个词的主题分布相似度
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = calsim(v, senttopic)
            sim_dic[k] = sim
        for k, v in sorted(sim_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            print(k + "/ ", end='')
        print()

    # 词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法
    def word_dictionary(self, doc_list):
        dictionary = []
        for doc in doc_list:
            dictionary.extend(doc)
        dictionary = list(set(dictionary))
        return dictionary

    def doc2bowvec(self, word_list):
        vec_list = [1 if word in word_list else 0 for word in self.dictionary]
        return vec_list


def topic_extract(word_list, model, pos=False, keyword_num=10):
    doc_list = load_data(pos)
    topic_model = TopicModel(doc_list, keyword_num, model=model)
    topic_model.get_simword(word_list)


def three_two():
    text = input()
    pos = True
    seg_list = seg_to_list(text, pos)
    filter_list = word_filter(seg_list, pos)
    topic_extract(filter_list, 'LSI', pos)


# 4. LDA 算法
# 4.1 LDA 算法
def four_one():
    import jieba
    import jieba.analyse as analyse
    import gensim
    from gensim import corpora, models, similarities

    # 停用词表加载方法
    def get_stopword_list():
        # 停用词表存储路径，每一行为一个词，按行读取进行加载
        # 进行编码转换确保匹配准确率
        stop_word_path = './stopword.txt'
        stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path, encoding='utf-8').readlines()]
        return stopword_list

    # 停用词
    stop_word = get_stopword_list()
    text = input()

    # 分词
    sentences = []
    segs = jieba.lcut(text)
    segs = list(filter(lambda x: x not in stop_word, segs))
    sentences.append(segs)

    # 构建词袋模型
    dictionary = corpora.Dictionary(sentences)
    corpus = [dictionary.doc2bow(sentence) for sentence in sentences]
    result = ""
    # 任务：使用gensim模块中的函数构造LDA模型，得出最佳主题词的分析结果保存到result变量中。
    # ********** Begin *********#
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=1)
    result = lda.print_topics(num_words=1)[0][1]

    # ********** End **********#
    print(result.split('*')[1],end="")

