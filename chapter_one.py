# 2. 自然语言处理的前置技术
# 2.1 正则表达式在 NLP 中的应用
def two_one():
    import re

    # 任务1：抽取years_string中所有的年份并输出
    years_string = input()
    # ********** Begin *********#
    years = re.findall(r'\d{4}', years_string)

    # ********** End **********#
    print(years)

    # 任务2：匹配text_string中包含“文本”的句子，并使用print输出，以句号作为分隔
    text_string = input()
    regex = '文本'
    # ********** Begin *********#
    text = text_string.split('。')
    for line in text:
        if re.search(regex, line) is not None:
            print(line)

    # ********** End **********#


# 2.2 Numpy 的使用
def two_two():
    import numpy as np
    matrix = np.ones((3, 2), dtype=int)  # matrix为3行2列的数组
    for i in range(0, 3):
        for j in range(0, 2):
            matrix[i][j] = input()

    print(matrix)

    # 任务1：输出matrix第二列的最大值
    # ********** Begin *********#
    print(matrix.max(axis=0)[1])

    # ********** End **********#

    # 任务2：输出matrix按行求和的结果
    # ********** Begin *********#
    print(matrix.sum(axis=1))

    # ********** End **********#
    