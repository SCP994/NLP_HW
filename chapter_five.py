# 五、句法分析
# 2. 句法分析的常用方法与实战
# 2.1 Pyhanlp 的使用
def two_one(text):
    from pyhanlp import HanLP
    # text = input()
    # 任务：使用pyhanlp对text进行关键词提取并输出前两个关键词
    # ********** Begin *********#
    print(HanLP.extractKeyword(text, 2))

    # ********** End **********#


def test_two_one():
    text = '疫情是一场灾难，这场灾难甚至要比地震、海啸等自然灾害来得更为持久。疫情像一面照妖镜，照向了人性最深处。照妖镜之下，有人为了暴利制假售\
    假、哄抬物价，有人为了“便利”抗拒防控、冲击关卡，也有人伺机打劫绑架、寻衅滋事。而每当灾难来临，每个普通人在身心上也将受到不同程度的冲击，变得尤\
    为敏感，缺乏安全感与归宿感。'
    two_one(text)


def two_two(text):
    from pyhanlp import HanLP
    # text = input()
    # 任务：使用pyhanlp模块，对text文本进行句法分析并逐行输出结果，以%s --(%s)--> %s格式输出
    # ********** Begin *********#
    sentence = HanLP.parseDependency(text)
    for i in sentence:
        print('%s --(%s)--> %s' % (i.LEMMA, i.DEPREL, i.HEAD.LEMMA.strip('#')))

    # ********** End **********#


def test_two_two():
    text = '疫情不结束不下战场'
    two_two(text)

