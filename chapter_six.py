# 六、文本向量化
# 1. 向量化算法 word2vec
# 1.1 向量化算法 word2vec
def one_one():
    import logging
    from gensim.models import word2vec

    def getmodel():
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # 加载《人民的名义》文本
        sentences = word2vec.LineSentence('./in_the_name_of_people_segment.txt')

        # 任务：使用 gensim 模块中的word2vec对sentences文本构建合适的word2vec模型，并保存到model变量中，使得文本中的人名相近度达0.85以上。
        # ********** Begin *********#
        model = word2vec.Word2Vec(sentences, hs=0, min_count=3, window=3, size=100)

        # ********** End **********#
        return model


# 2. 向量化算法 doc2vec
# 2.1