# 二、基于规则的词法分析
# 1. 基于正则表达式的词形分析方法
# 1.1 基于正则表达式的词形分析方法——邮箱地址和网址提取
def one_one():
    import re
    string = input()

    # 任务：采用re库函数，对string分别进行邮箱地址、网址提取，
    # 将邮箱地址依次分行显示
    # ********** Begin *********#
    rr1 = re.compile(r'\w*@\w*\.com|\w*@\w*edu\.net')
    ret = rr1.findall(string)
    if len(ret):
        print('提取邮箱地址如下：')
        for i in ret:
            print(i)

    rr2 = re.compile(r'https://[^ ]+\b')
    ret = rr2.finditer(string)
    temp = 0
    for i in ret:
        if temp == 0:
            temp += 1
            print('提取网址如下：')
        print(i.group())

    # ********** End **********#


# 1.2 基于正则表达式的词形分析方法——密码提取
def one_two():
    import re
    string = input()
    # 任务：采用re库函数，对string分别进行密码格式认证提取，
    # 验证输入密码符合要求（8位以上，字母开头，只能是字母、数字、下划线）
    # ********** Begin *********#
    rr1 = re.compile(r'[a-zA-Z][a-zA-Z0-9_]{7,}')
    ret = rr1.findall(string)
    print('提取密码是')
    for i in ret:
        print(i)

    # ********** End **********#


# 2. 划分句子的决策树算法
# 2.1 划分句子的决策树算法
def two_one():
    import re
    text = input()
    list_ret = list()
    # 任务：完成对text文本的分句并输出结果
    # ********** Begin *********#
    list_ret = re.split(r'\. (?![a-z])|! |\? ', text)
    list_ret[-1] = re.sub(r'.$', '', list_ret[-1])
    print(list_ret)

    # ********** End **********#


# 3. 计算单词之间的最小编辑距离
# 3.1 计算单词之间的最小编辑距离
def minDistance(word1, word2):
    n = len(word1)
    m = len(word2)

    # 有一个字符串为空串
    if n * m == 0:
        return n + m

    # DP 数组
    D = [[0] * (m + 1) for _ in range(n + 1)]

    # 边界状态初始化
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j

    ######## Begin ########
    for i in range(1, n + 1, 1):  # 动态规划 注意从 D[1][1] 开始
        for j in range(1, m + 1, 1):
            ch1 = word1[i - 1]
            ch2 = word2[j - 1]
            if ch1 == ch2:
                D[i][j] = D[i - 1][j - 1]  # 字符相同
            else:
                D[i][j] = min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1]) + 1  # 字符不相同时，删除或者改成相同字符

    ######## End ########
    return D[n][m]


def test_three_one():
    word1 = 'horse'
    word2 = 'ros'
    ans = minDistance(word1, word2)
    print(ans)
