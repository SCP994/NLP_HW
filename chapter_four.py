# 四、词性标注与命名实体识别
# 1. 词性标注
# 1.1 词性标注
def four_one(text):
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


def test_four_one():
    text = '还有什么是比jieba更好的中文分词工具吗？'
    four_one(text)


# 2. 命名实体识别
# 2.1 命名实体识别
# 2.2 中文人名识别
def four_two():
    from pyhanlp import HanLP
    text = input()
    # 任务：完成对 text 文本的人名识别并输出结果
    # ********** Begin *********#
    segment = HanLP.newSegment().enableNameRecognize(True)
    result = segment.seg(text)
    print(result)

    # ********** End **********#


def test_four_two():
    text = '张三今天没来上课'
    four_two(text)


# 2.3 地名识别
def four_three():
    from pyhanlp import HanLP
    # text = input()
    # 任务：完成对 text 文本的地名识别并输出结果
    # ********** Begin *********#
    segment = HanLP.newSegment().enablePlaceRecognize(True)
    result = segment.seg(text)
    print(result)

    # ********** End **********#


def test_four_three():
    text = '中国是个好地方'
    four_three(text)
