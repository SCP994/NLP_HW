import matplotlib.pyplot as plt
from wordcloud import WordCloud


# 三、中文分词与 N 元模型
# 1. 规则分词法
# 1.1 正向最大匹配法
def cutA(sentence, dictA):
    # sentence：要分词的句子
    result = []
    sentenceLen = len(sentence)
    n = 0
    maxDictA = max([len(word) for word in dictA])
    # 任务：完成正向匹配算法的代码描述，并将结果保存到result变量中
    # result变量为分词结果
    # ********** Begin *********#
    i = 0
    t = 0
    while i < sentenceLen:
        for j in range(maxDictA, 0, -1):
            if i + j > sentenceLen:
                continue
            str = sentence[i:i+j]
            if j == 1:  # 一个字的情况
                result.append(str)
                i += j
                break
            for word in dictA:
                if len(word) == j and word == str:
                    result.append(str)
                    i += j
                    t = 1
                    break
            if t == 1:
                t = 0
                break

    # ********** End **********#
    print(result)  # 输出分词结果


def test_cutA():
    sentence = '南京市长江大桥'
    dictA = ['南京市', '南京市长', '长江大桥', '大桥']
    cutA(sentence, dictA)


# 1.2 逆向最大匹配法
def cutB(sentence, dictB):
    result = []
    sentenceLen = len(sentence)
    maxDictB = max([len(word) for word in dictB])
    # 任务：完成逆向最大匹配算法的代码描述
    # ********** Begin *********#
    i = sentenceLen
    t = 0
    while i > 0:
        for j in range(maxDictB, 0, -1):
            if i - j < 0:
                continue
            str = sentence[i-j:i]
            if j == 1:
                result.append(str)
                i -= j
                break
            for word in dictB:
                if len(word) == j and word == str:
                    result.append(str)
                    i -= j
                    t = 1
                    break
            if t == 1:
                t = 0
                break
    # ********** End **********#
    print(result[::-1], end="")


def test_cutB():
    sentence = '南京市长江大桥'
    dictA = ['南京市', '南京市长', '长江大桥', '大桥']
    cutB(sentence, dictA)


# 1.3 双向最大匹配法
class BiMM():
    def __init__(self):
        self.window_size = 3  # 字典中最长词数

    def MMseg(self, text, dict):  # 正向最大匹配算法
        result = []
        index = 0
        text_length = len(text)
        while text_length > index:
            for size in range(self.window_size + index, index, -1):
                piece = text[index:size]
                if piece in dict:
                    index = size - 1
                    break
            index += 1
            result.append(piece)
        return result

    def RMMseg(self, text, dict):  # 逆向最大匹配算法
        result = []
        index = len(text)
        while index > 0:
            for size in range(index - self.window_size, index):
                piece = text[size:index]
                if piece in dict:
                    index = size + 1
                    break
            index = index - 1
            result.append(piece)
        result.reverse()
        return result

    def main(self, text, r1, r2):
        # 任务：完成双向最大匹配算法的代码描述
        # ********** Begin *********#
        list1 = self.MMseg(text, r1)
        list2 = self.RMMseg(text, r2)
        if len(list1) != len(list2):
            result = list1 if len(list1) < len(list2) else list2
            print(result)
            return result

        count1 = 0
        count2 = 0
        for i in range(len(list1)):
            if len(list1[i]) == 1:
                count1 += 1
            if len(list2[i]) == 1:
                count2 += 1
        result = list1 if count1 < count2 else list2
        print(result)
        return result
        # ********** End **********#


def test_BIMM():
    text = '研究生命的起源'
    r1 = ['研究', '研究生', '生命', '的', '起源']
    r2 = ['研究', '研究生', '生命', '的', '起源']
    bimm: BiMM = BiMM()
    r = bimm.main(text, r1, r2)


# 2. 统计分词法
# 2.1 词频统计
def two_one(text):
    wc = WordCloud().generate(text=text)
    plt.imshow(wc, interpolation='bilinear')  # 显示词云
    plt.axis('off')
    plt.show()

    text = text.lower()

    # 将特殊字符替换成为空格
    for ch in '!@#$%:^&*()-.;':
        text = text.replace(ch, " ")

    # 对字符串通过空格进行分割
    words = text.split()

    counts = {}
    # 任务：完成对text文本的词频统计，将结果保存到counts字典中
    # ********** Begin *********#
    for word in words:
        if word in counts:
            counts[word] = counts[word] + 1
        else:
            counts[word] = 1

    # ********** End **********#
    items = list(counts.items())  # 返回可遍历的（键、值）元组数据
    items.sort(key=lambda x: x[1], reverse=True)  # 逆序
    # 输出词频统计的结果
    for i in range(3):
        word, count = items[i]
        if i < 2:
            print("{0}：{1}".format(word, count))
        else:
            print("{0}：{1}".format(word, count), end="")


def test_two_one():
    text = 'Got tho on super sale. Love it! Cuts my drying time in half Reckon I have had this about a year now, at \
    least 7 months. Works great, I use it 5 days a week, blows hot air, doesnt overheat, isnt to big, came quick, \
    didnt cost much. Get you one, you will like it.The styling tip does not stay on, keeps falling off in the middle \
    of blow drying and then it\'s too hot to put back'
    two_one(text)


# 2.2 统计分词原理与实战
class HMM(object):
    def __init__(self):
        self.state_list = ['B', 'M', 'E', 'S']
        self.start_p = {}
        self.trans_p = {}
        self.emit_p = {}

        self.model_file = 'hmm_model.pkl'
        self.trained = False

    def train(self, datas, model_path=None):
        if model_path == None:
            model_path = self.model_file
        # 统计状态频数
        state_dict = {}

        def init_parameters():
            for state in self.state_list:
                self.start_p[state] = 0.0
                self.trans_p[state] = {s: 0.0 for s in self.state_list}
                self.emit_p[state] = {}
                state_dict[state] = 0

        def make_label(text):
            out_text = []
            if len(text) == 1:
                out_text = ['S']
            else:
                out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']
            return out_text

        init_parameters()
        line_nb = 0

        # 监督学习方法求解参数
        for line in datas:
            line = line.strip()
            if not line:
                continue
            line_nb += 1

            word_list = [w for w in line if w != ' ']
            line_list = line.split()
            line_state = []
            for w in line_list:
                line_state.extend(make_label(w))

            assert len(line_state) == len(word_list)

            for i, v in enumerate(line_state):
                state_dict[v] += 1

                if i == 0:
                    self.start_p[v] += 1
                else:
                    self.trans_p[line_state[i - 1]][v] += 1
                    self.emit_p[line_state[i]][word_list[i]] = self.emit_p[line_state[i]].get(word_list[i], 0) + 1.0

        self.start_p = {k: v * 1.0 / line_nb for k, v in self.start_p.items()}
        self.trans_p = {k: {k1: v1 / state_dict[k1] for k1, v1 in v0.items()} for k, v0 in self.trans_p.items()}
        self.emit_p = {k: {k1: (v1 + 1) / state_dict.get(k1, 1.0) for k1, v1 in v0.items()} for k, v0 in
                       self.emit_p.items()}

        with open(model_path, 'wb') as f:
            import pickle
            pickle.dump(self.start_p, f)
            pickle.dump(self.trans_p, f)
            pickle.dump(self.emit_p, f)
        self.trained = True
        print('model train done,parameters save to ', model_path)

    # 读取参数模型
    def load_model(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.start_p = pickle.load(f)
            self.trans_p = pickle.load(f)
            self.emit_p = pickle.load(f)
        self.trained = True
        print('model parameters load done!')

    # 维特比算法求解最优路径
    def __viterbi(self, text, states, start_p, trans_p, emit_p):
        V = [{}]
        path = {}
        for y in states:
            V[0][y] = start_p[y] * emit_p[y].get(text[0], 1.0)
            path[y] = [y]

        for t in range(1, len(text)):
            V.append({})
            new_path = {}

            for y in states:
                emitp = emit_p[y].get(text[t], 1.0)

                (prob, state) = max([(V[t - 1][y0] * trans_p[y0].get(y, 0) * emitp, y0) \
                                     for y0 in states if V[t - 1][y0] > 0])
                V[t][y] = prob
                new_path[y] = path[state] + [y]
            path = new_path

        if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text[-1], 0):
            (prob, state) = max([(V[len(text) - 1][y], y) for y in ('E', "M")])
        else:
            (prob, state) = max([(V[len(text) - 1][y], y) for y in states])

        return (prob, path[state])

    def cut(self, text):
        if not self.trained:
            print('Error：please pre train or load model parameters')
            return

        prob, pos_list = self.__viterbi(text, self.state_list, self.start_p, self.trans_p, self.emit_p)
        begin_, next_ = 0, 0
        # 任务:完成 HMM 中文分词算法
        # ********* Begin *********#
        for i, char in enumerate(text):
            pos = pos_list[i]
            if pos == 'B':
                begin_ = i
            elif pos == 'E':
                yield text[begin_:i+1]
                next_ = i + 1
            elif pos == 'S':
                yield char
                next_ = i + 1
        if next_ < len(text):
            yield text[next_:]
        # ********* Begin *********#


def test_HMM():
    text = input()

    train_data = 'pku_training.utf8'
    model_file = 'hmm_model.pkl'
    hmm = HMM()
    hmm.train(open(train_data, 'r', encoding='utf-8'), model_file)
    hmm.load_model(model_file)
    print('/'.join(hmm.cut(text)))


# 3. 基于 Jieba 的中文分词挑战
# 3.1 中文分词工具——Jieba
def three_one(text):
    import jieba
    # text = input()
    seg_list1 = ''
    seg_list2 = ''
    # 任务：采用jieba库函数，对text分别进行精确模式分词和搜索引擎模式分词，
    # 将分词结果分别保存到变量seg_list1和seg_list2中
    # ********** Begin *********#
    seg_list1 = jieba.cut(text, cut_all=False)  # 精确模式（默认），cut 返回迭代器，lcut 返回列表

    seg_list2 = jieba.cut_for_search(text)  # 搜索引擎模式（在精确模式的基础上，对长词再次切分，提高召回率）

    # ********** End **********#
    print("精确模式："+'/'.join(seg_list1) +"  搜索引擎模式："+ ' /'.join(seg_list2))


def test_three_one():
    text = '我来自北京大学'
    three_one(text)


# 3.2 基于 Jieba 的词频统计
def three_two(text):
    import jieba
    # text = input()
    words = jieba.lcut(text)
    data = {}  # 词典

    # 任务：完成基于 Jieba 模块的词频统计
    # ********** Begin *********#
    for ch in words:
        if len(ch) < 2:  # 一个字的情况
            continue
        if ch in data:
            data[ch] += 1
        else:
            data[ch] = 1

    # ********** End **********#
    data = sorted(data.items(), key=lambda x: x[1], reverse=True)  # 排序
    print(data[:3], end="")


def test_three_two():
    text = '很长时间不吃水果，到时又年纪比较小，如果说很长时间不吃水果的话，可能对我的身体健康造成，比较大的一个影响，老师每人发了一个苹果给我\
    们，然后我们每个人吃了一个苹果，当时学校里发的那个苹果，还吃起来还是比较甜的，会说我当时吃起来特别好吃'
    three_two(text)


# 4. N 元语言模型
# 4.1 预测句子概率
def four_one(sentence_test):
    import jieba

    # 语料句子
    sentence_ori = "研究生物很有意思。他大学时代是研究生物的。生物专业是他的首选目标。他是研究生。"
    # 测试句子
    # sentence_test = input()
    # 任务：完成对2-gram模型的建立，计算测试句子概率并输出结果
    # ********** Begin *********#
    import logging
    # 关闭 jieba log 输出
    jieba.setLogLevel(logging.INFO)

    import sys, os
    # 可关闭控制台输出，这里用不到
    # sys.stdout = open(os.devnull, "w")
    # sys.stderr = open(os.devnull, "w")

    if sentence_ori.endswith('。'):
        sentence_ori = sentence_ori[:-1]
    sentence_modify_ori1 = sentence_ori.replace('。', 'EOSBOS')
    sentence_modify_ori2 = 'BOS' + sentence_modify_ori1 + 'EOS'

    if sentence_test.endswith('。'):
        sentence_test = sentence_test[:-1]
    sentence_test_modify_test1 = sentence_test.replace('。', 'EOSBOS')
    sentence_test_modify_test2 = 'BOS' + sentence_test_modify_test1 + 'EOS'

    jieba.suggest_freq('BOS', True)
    jieba.suggest_freq('EOS', True)

    sentence_ori = jieba.cut(sentence_modify_ori2, HMM=False)
    format_sentence_ori = '，'.join(sentence_ori)

    lists_ori = format_sentence_ori.split('，')

    dicts_ori = {}

    if not bool(dicts_ori):  # 空字典
        for word in lists_ori:
            if word not in dicts_ori:
                dicts_ori[word] = 1
            else:
                dicts_ori[word] += 1

    sentence_test = jieba.cut(sentence_test_modify_test2, HMM=False)
    format_sentence_test = '，'.join(sentence_test)

    lists_test = format_sentence_test.split('，')

    dicts_test = {}

    if not bool(dicts_test):
        for word in lists_test:
            if word not in dicts_test:
                dicts_test[word] = 1
            else:
                dicts_test[word] += 1

    count_list = [0] * (len(lists_test))
    for i in range(1, len(lists_test), 1):
        for j in range(1, len(lists_ori), 1):
            if lists_test[i - 1] == lists_ori[j - 1] and lists_test[i] == lists_ori[j]:
                count_list[i] += 1

    p = 1.0
    for i in range(1, len(lists_test), 1):
        key = lists_test[i]
        p *= (float(count_list[i]) / float(dicts_ori[key]))
    print(p)
    # ********** End **********#


def test_four_one():
    sentence_text = '研究生物专业是他的首选目标'
    four_one(sentence_text)


# 4.2 数据平滑
def four_two(sentence_test):
    import jieba

    # 语料句子
    sentence_ori = "研究生物很有意思。他大学时代是研究生物的。生物专业是他的首选目标。他是研究生。"
    # 测试句子
    # sentence_test = input()
    # 任务：编写平滑函数完成数据平滑，利用平滑数据完成对2-gram模型的建立，计算测试句子概率并输出结果
    # ********** Begin *********#
    import logging
    # 关闭 jieba log 输出
    jieba.setLogLevel(logging.INFO)
    jieba.suggest_freq('BOS', True)  # 可以把 'EOSBOS' 分开，此处应该用不到，因为直接把句子分开了
    jieba.suggest_freq('EOS', True)

    ori = sentence_ori
    test = sentence_test

    if ori.endswith('。'):
        ori = ori[:-1]
    list_ori = ori.split('。')
    for i in range(len(list_ori)):
        str = 'BOS' + list_ori[i] + 'EOS'
        list_ori[i] = jieba.lcut(str, HMM=False)

    if test.endswith('。'):
        test.endswith('。')
    test = 'BOS' + test + 'EOS'
    list_test = jieba.lcut(test, HMM=False)

    def get_dict(list_sen, dict_ori):
        for i in range(1, len(list_sen), 1):
            ch1 = list_sen[i]
            ch2 = list_sen[i - 1]
            if ch1 in dict_ori:
                if ch2 in dict_ori[ch1]:
                    dict_ori[ch1][ch2] += 1
                else:
                    dict_ori[ch1][ch2] = 1
            else:
                dict_ori[ch1] = {}
                dict_ori[ch1][ch2] = 1

    dict_ori = {}
    for i in range(len(list_ori)):  # 遍历 list_ori 中的每一个句子
        get_dict(list_ori[i], dict_ori)

    dict_test = {}
    get_dict(list_test, dict_test)

    list_Nc = [0] * 10  # 用来保存频数的频数，list_Nc[1] 代表出现次数为 1 的二元词组数量
    for i in dict_test:  # Nc 根据测试语句计算！
        for j in dict_test[i]:
            if i in dict_ori and j in dict_ori[i]:
                c = dict_ori[i][j]
                list_Nc[c] += 1
            else:
                list_Nc[0] += 1

    N = 0
    for i in range(len(list_Nc)):
        N += i * list_Nc[i]
    probs = {}
    for i in dict_test:
        probs[i] = {}
        for j in dict_test[i]:
            if i in dict_ori and j in dict_ori[i]:
                c_star = dict_ori[i][j] + 1
                if list_Nc[c_star] != 0:
                    c_star *= list_Nc[c_star] / list_Nc[c_star - 1]
                probs[i][j] = c_star / N
            else:
                probs[i][j] = 1.0 * (list_Nc[1] / list_Nc[0]) / N  # 注意这里

    p = 1.0
    for i in range(1, len(list_test), 1):
        ch1 = list_test[i]
        ch2 = list_test[i - 1]
        p *= probs[ch1][ch2]
    print(p)

    # ********** End **********#


def test_four_two():
    sentence_test = '他是研究物理的'
    sentence_test = '他研究生时代是学习物理的'
    four_two(sentence_test)
