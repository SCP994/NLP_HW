# 八、基于机器学习的 NLP 算法
# 2. 分类器方法
# 2.2 基于朴素贝叶斯的文本分类
from functools import reduce
import operator
from numpy import array, zeros
import xlwt
import itertools
import nltk
import os
import sklearn
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import pickle


def trainNB(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 文档数量
    numWords = len(trainMatrix[0])  # 第一篇文档的长度，也就是词汇表的长度
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 负面文档占总文档比例
    p0Num = zeros(numWords)  # 初始化概率
    p1Num = zeros(numWords)
    p0Denom = 0
    p1Denom = 0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 如果是负面文档
            p1Num += trainMatrix[i]  # 文档对应的词语数量全部加1，向量相加
            p1Denom += sum(trainMatrix[i])  # 负面文档词语的总数量
        else:
            p0Num += trainMatrix[i]  # 正常文档对应的词语数量向量
            p0Denom += sum(trainMatrix[i])  # 正常文档词语的总数量

    p1Vect = p1Num / p1Denom  # 对p1Num的每个元素做除法，即负面文档中出现每个词语的概率
    p0Vect = p0Num / p0Denom  # 对p0Num的每个元素做除法，即正常文档中出现每个词语的概率
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, trainMatrix, trainCategory):
    p0Vect, p1Vect, pAb = trainNB(trainMatrix, trainCategory)
    # 计算待分类文档词条对应的条件概率
    p1VectClassify = vec2Classify * p1Vect
    p0VectClassify = vec2Classify * p0Vect
    p1Cond = []
    p0Cond = []

    for i in range(len(p1VectClassify)):
        if p1VectClassify[i] == 0:
            continue
        else:
            p1Cond.append(p1VectClassify[i])

    for i in range(len(p0VectClassify)):
        if p0VectClassify[i] == 0:
            continue
        else:
            p0Cond.append(p0VectClassify[i])
    # 任务：完成对各概率向量的计算
    # ********** Begin *********#
    if len(p0Cond):
        pC0 = reduce(operator.mul, p0Cond, 1)
    else:
        pC0 = 0
    if (len(p1Cond)):
        pC1 = reduce(operator.mul, p1Cond, 1)
    else:
        pC1 = 0
    p1 = pC1 * pAb
    p0 = pC0 * (1.0 - pAb)

    # ********** End **********#
    if p1 > p0:
        return 1
    else:
        return 0


# 2.3 基于逻辑回归的文本分类
def Text_categorization():
    file_name=input()
    dataset = pd.read_csv('src/step2/data/'+file_name)
    # nltk.download('stopwords')
    stemmer = PorterStemmer()
    words = stopwords.words("english")
    dataset['cleaned'] = dataset['text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    vectorizer = TfidfVectorizer(min_df=3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))
    final_features = vectorizer.fit_transform(dataset['cleaned']).toarray()
    from sklearn.linear_model import LogisticRegression
    X = dataset['cleaned']
    Y = dataset['category']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    # 任务：完成对逻辑回归的建模
    # ********** Begin *********#
    model = Pipeline([('vect', vectorizer),('chi', SelectKBest(chi2, k=1200)),('clf', LogisticRegression(random_state=0))])
    model.fit(X_train, y_train)

    # ********** End **********#

    ytest = np.array(y_test)
    return X_test,ytest,model


# 2.4 基于支持向量机的文本分类
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from test import get_data,get_result
df = get_data()

counter = Counter(df['variety'].tolist())
top_10_varieties = {i[0]: idx for idx, i in enumerate(counter.most_common(10))}
df = df[df['variety'].map(lambda x: x in top_10_varieties)]

description_list = df['description'].tolist()
varietal_list = [top_10_varieties[i] for i in df['variety'].tolist()]
varietal_list = np.array(varietal_list)

count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(description_list)

# 任务：完成对文本的TF-IDF值的计算
# ********** Begin *********#
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts) # 用于统计vectorizer中每个词语的TFIDF值

# ********** End *********#

train_x, test_x, train_y, test_y = train_test_split(x_train_tfidf, varietal_list, test_size=0.3)

clf = SVC(kernel='linear').fit(train_x, train_y)
y_score = clf.predict(test_x)

n_right = 0
for i in range(len(y_score)):
    if y_score[i] == test_y[i]:
        n_right += 1

get_result(n_right,test_y)


# 3. 机器学习在 NLP 中的实战
# 3.2 基于 K-Means 算法的文本聚类
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans


class KmeansClustering():
    def __init__(self, stopwords_path=None):
        self.stopwords = self.load_stopwords(stopwords_path)
        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer()

    def load_stopwords(self, stopwords=None):
        # 加载停用词

        if stopwords:
            with open(stopwords, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f]
        else:
            return []

    def preprocess_data(self, corpus_path):
        # 文本预处理，每行一个文本
        corpus = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                corpus.append(' '.join([word for word in jieba.lcut(line.strip()) if word not in self.stopwords]))
        return corpus

    def get_text_tfidf_matrix(self, corpus):

        # 获取tfidf矩阵

        tfidf = self.transformer.fit_transform(self.vectorizer.fit_transform(corpus))

        # 获取tfidf矩阵中权重
        weights = tfidf.toarray()
        return weights

    def kmeans(self, corpus_path, n_clusters=2):
        """
        KMeans文本聚类
        :param corpus_path: 语料路径（每行一篇）,文章id从0开始
        :param n_clusters: ：聚类类别数目
        :return: {cluster_id1:[text_id1, text_id2]}
        """
        corpus = self.preprocess_data(corpus_path)
        weights = self.get_text_tfidf_matrix(corpus)
        result = {}
        # 任务：完成基于K-Means算法的文本聚类，并将结果保存到result变量中。
        # ********** Begin *********#
        k_means = KMeans(n_clusters=3, random_state=10)
        k_means.fit(weights)
        result = k_means.predict(weights)

        # ********** End **********#
        return result


# 3.3 基于 DBSCAN 的文本聚类
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class DbscanClustering():
    def __init__(self, stopwords_path=None):
        self.stopwords = self.load_stopwords(stopwords_path)
        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer()

    def load_stopwords(self, stopwords=None):  # 加载停用词
        if stopwords:
            with open(stopwords, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f]
        else:
            return []

    def preprocess_data(self, corpus_path):  # 文本预处理
        corpus = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                corpus.append(' '.join([word for word in jieba.lcut(line.strip()) if word not in self.stopwords]))
        return corpus

    def get_text_tfidf_matrix(self, corpus):  # 获取tf-idf矩阵
        tfidf = self.transformer.fit_transform(self.vectorizer.fit_transform(corpus))
        weights = tfidf.toarray()  # 获取tfidf矩阵中权重
        return weights

    def pca(self, weights, n_components=2):  # PCA对数据进行降维
        pca = PCA(n_components=n_components)
        return pca.fit_transform(weights)

    def dbscan(self, corpus_path, eps=0.1, min_samples=3, fig=True):  # 基于密度的文本聚类算法

        # 任务：完成 DBSCAN 聚类算法
        # ********** Begin *********#
        corpus = self.preprocess_data(corpus_path) # 加载语料
        weights = self.get_text_tfidf_matrix(corpus) # 词向量转换
        pca_weights = self.pca(weights) # 减低维度
        clf = DBSCAN(eps=eps, min_samples=min_samples) # 构建聚类算法

        # ********** End **********#
        y = clf.fit_predict(pca_weights)

        result = {}  # 每个样本所属的簇
        for text_idx, label_idx in enumerate(y):
            if label_idx not in result:
                result[label_idx] = [text_idx]
            else:
                result[label_idx].append(text_idx)
        return result


# 3.4 基于机器学习的情感分析
pos_f = 'src/step3/pkl_data/1000/pos_review.pkl'
neg_f = 'src/step3/pkl_data/1000/neg_review.pkl'


def load_data():  # 加载训练集数据
    global pos_review, neg_review
    pos_review = pickle.load(open(pos_f, 'rb'))
    neg_review = pickle.load(open(neg_f, 'rb'))


def create_word_bigram_scores():  # 计算整个语料里面每个词和双词搭配的信息量
    posdata = pickle.load(open(pos_f, 'rb'))
    negdata = pickle.load(open(neg_f, 'rb'))

    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    bigram_finder = BigramCollocationFinder.from_words(posWords)
    posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)

    pos = posWords + posBigrams  # 词和双词搭配
    neg = negWords + negBigrams

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        word_fd[word] += 1
        cond_word_fd["pos"][word] += 1
    for word in neg:
        word_fd[word] += 1
        cond_word_fd["neg"][word] += 1

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count),
                                               total_word_count)  # 计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores


def find_best_words(word_scores, number):  # 根据信息量进行倒序排序，选择排名靠前的信息量的词
    best_vals = sorted(word_scores.items(), key=lambda w_s: w_s[1], reverse=True)[
                :number]  # 把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
    best_words = set([w for w, s in best_vals])
    return best_words


def pos_features(feature_extraction_method):  # 赋予积极的文本类标签
    posFeatures = []
    for i in pos_review:
        posWords = [feature_extraction_method(i), 'pos']  # 为积极文本赋予"pos"
        posFeatures.append(posWords)
    return posFeatures


def neg_features(feature_extraction_method):  # 赋予消极的文本类标签

    negFeatures = []
    for j in neg_review:
        negWords = [feature_extraction_method(j), 'neg']  # 为消极文本赋予"neg"
        negFeatures.append(negWords)
    return negFeatures


def best_word_features(words):  # 把选出的这些词作为特征（这就是选择了信息量丰富的特征）
    global best_words
    return dict([(word, True) for word in words if word in best_words])


def score(classifier):
    # 任务：构建分类器模型并进行训练
    # ********** Begin *********#
    classifier = nltk.SklearnClassifier(classifier)  # 在nltk 中使用scikit-learn的接口
    classifier.train(train)  #训练分类器
    pred = classifier.classify_many(dev)  # 对开发测试集的数据进行分类，给出预测的标签
    return accuracy_score(tag_dev, pred)  # 对比分类预测结果和人工标注的正确结果，给出分类器准确度

    # ********** End **********#

    pred = classifier.classify_many(dev)  # 对开发测试集的数据进行分类，给出预测的标签
    return accuracy_score(tag_dev, pred)  # 对比分类预测结果和人工标注的正确结果，给出分类器准确度


# 使用测试集测试分类器的最终效果
def use_the_best():
    word_scores = create_word_bigram_scores()  # 使用词和双词搭配作为特征
    best_words = find_best_words(word_scores, 4000)  # 特征维度1500
    load_data()
    posFeatures = pos_features(best_word_features, best_words)
    negFeatures = neg_features(best_word_features, best_words)
    cut_data(posFeatures, negFeatures)
    trainSet = posFeatures[1500:] + negFeatures[1500:]  # 使用了更多数据
    testSet = posFeatures[:500] + negFeatures[:500]
    test, tag_test = zip(*testSet)

    # 存储分类器
    def final_score(classifier):
        classifier = SklearnClassifier(classifier)
        classifier.train(trainSet)
        pred = classifier.classify_many(test)
        return accuracy_score(tag_test, pred)

    print(final_score(MultinomialNB()))  # 使用开发集中得出的最佳分类器


# 把分类器存储下来（存储分类器和前面没有区别，只是使用了更多的训练数据以便分类器更为准确）
def store_classifier():
    load_data()
    word_scores = create_word_bigram_scores()
    global best_words
    best_words = find_best_words(word_scores, 7500)

    posFeatures = pos_features(best_word_features)
    negFeatures = neg_features(best_word_features)

    trainSet = posFeatures + negFeatures

    MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
    MultinomialNB_classifier.train(trainSet)
    pickle.dump(MultinomialNB_classifier, open('src/step3/out/classifier.pkl', 'wb'))

