# -*- coding: utf-8 -*-
# @Time    : 2021/10/6 14:20
# @Author  : heyee (jialeyang)


from math import log
from structure.entity_extractor.data_scripts.data_utils import *


class SegSentence(object):
    def __init__(self):
        self.seg_tags_level_0 = ['\t']
        self.seg_tags_level_1 = ['\t', '。', ',', '，', "|", "；", ";", '!', '！', '?', '？', '…', '、', '～']
        # self.seg_tags_level_1 = ['。', ',', '，']
        # self.seg_tags_level_0 = ['\t', '\n', '\r', '。', '!', '！', '?', '？', ';', '；', ',', '，']
        # self.seg_tags_level_1 = ['.', '…', '、', ':', '：', '~', '\uFF5E', ' ', '\u00A0', '#', '-']
        # self.seg_tags_level_2 = ['#', '-']
        self.FREQ = {}
        self.total = 0

    def cut(self, sentence):
        """
        old:
            (1) 遇到 level_0 级别的标点符号直接进行断句处理，对断句后长度大于 60 的子句进行步骤（2）
            (2) 用 jieba 分词思想（后向动态规划+前向贪心算法）对句子长句建模，以最终得到的最大概率路径最为断句结果
            (3) 对步骤（2）中长句仍大于 60 的子句进行此步处理
        new:
            (1) 优先对"\t"进行断句，断句后长度大于120的，再细粒度断句
        """
        # step 1
        sen_collect = []
        # split_level0_list = self.cut_helper(self.seg_tags_level_0, sentence)

        # for idx, val in enumerate(split_level0_list):
        #     if len(val) < 60:
        #         sen_collect.append(val)
        #     else:
        #         split_level1_list = self.toSentenceListHelper(val)
        #         sen_collect.extend(split_level1_list)

        split_level0_list = self.cut_helper(self.seg_tags_level_0, sentence)
        self.total = len(split_level0_list)
        res = self.cut_DAG(split_level0_list)

        for idx, val in enumerate(res):
            if len(val) < 80:
                sen_collect.append(val)
                if len(val) < 10:
                    print(res)
                    continue
            else:
                split_level1_list = self.toSentenceListHelper(val)
                self.total = len(split_level1_list)
                res_level1 = self.cut_DAG(split_level1_list, min_len=40)
                sen_collect.extend(res_level1)
                for i in res_level1:
                    if len(i) < 10 or len(i) > 120:
                        print(res_level1)
                        break

        return sen_collect

        # filter_level0_list = [[idx, val] for idx, val in enumerate(split_level0_list) if len(val) > 60]
        #
        # # step 2
        # for idx, val in filter_level0_list:
        #     self.total = len(val)
        #     val_split_level1 = self.toSentenceListHelper(val)
        #     res = self.cut_DAG(val_split_level1)
        #     split_level0_list[idx] = res
        #
        # # flat list of list to list
        # split_level0_list = [[w] if isinstance(w, str) else w for w in split_level0_list]
        # if any(isinstance(el, list) for el in split_level0_list):
        #     flat_list = [item for sublist in split_level0_list for item in sublist]
        # else:
        #     flat_list = split_level0_list
        # return flat_list

    def get_DAG(self, sentence, min_len):
        self.FREQ = self.check_initialized(sentence)
        DAG = {}
        N = len(sentence)
        for k in range(N):
            tmplist = []
            tmpsum = 0
            i = k
            while i < N and tmpsum < 80:
                tmpsum += self.FREQ[i]
                if min_len <= tmpsum < 80:
                    tmplist.append(i)
                elif i == N - 1 and tmpsum < 40:
                    tmplist.append(i)
                i += 1
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG

    def cut_DAG(self, sentence, min_len=20):
        DAG = self.get_DAG(sentence, min_len=min_len)
        route = {}
        # 自底向上，后向动态规划
        self.calc(sentence, DAG, route)
        # 前向贪心算法搜索
        res = []
        x = 0
        N = len(sentence)
        while x < N:
            y = route[x][1] + 1
            res.append("".join(sentence[x:y]))
            x = y

        return res

    def calc(self, sentence, DAG, route):
        N = len(sentence)
        route[N] = (0, 0)
        logtotal = log(self.total)
        for idx in range(N - 1, -1, -1):
            route[idx] = max((log(self.calc_helper(idx, x) or 1) -
                              logtotal + route[x + 1][0], x) for x in DAG[idx])

    def calc_helper(self, idx, x):
        return sum([self.FREQ.get(i) for i in range(idx, x + 1)])

    def cut_helper(self, seg_tags_level, sentence):
        tag = "([" + "".join(seg_tags_level) + "])"
        split_list = re.split(tag, sentence)
        # 将标点与当前句子拼接在一起
        split_list.append("")
        split_res = ["".join(i) for i in zip(split_list[0::2], split_list[1::2])]

        return split_res

    def toSentenceListHelper(self, sentence):
        sentences = []
        chars = list(sentence)
        preLenth = 0
        sb = ""
        for i in range(len(chars)):
            if len(sb) == 0 and (chars[i].isspace() or chars[i] == " "):
                continue
            preLenth += 1
            sb += chars[i]
            if chars[i] == " ":
                # if i < len(chars) - 1 and ord(chars[i + 1]) > 128:
                if i < len(chars) - 1 and preLenth > 15:
                    sentences.append(sb)
                    sb = ""
                    preLenth = 0
            elif chars[i] in self.seg_tags_level_1:
                sentences.append(sb)
                sb = ""
                preLenth = 0
        if len(sb) > 0:
            sentences.append(sb)
        return sentences

    def check_initialized(self, sentence_list):
        """
        用于返回 sentence_list 中每个 sentence 的长度

        :param sentence_list:
        :return: dict
        """
        sen_dict = {}
        for idx, val in enumerate(sentence_list):
            sen_dict[idx] = len(val)
        return sen_dict

    def fetch_data_from_log(self, path):
        with open(path, 'r') as f:
            text = f.read()

        return text

    def analysis_data(self, text):
        """
        对长度大于 60 就直接丢弃 vs 基于规则的断句
        """
        text_length = re.findall("(text.length:)+(.+?)(,)+", text)
        SegLargeThan60_length = re.findall("(SegLargeThan60.length:)+(.+?)(,)+", text)
        SegShortLargeThan60_length = re.findall("(SegShortLargeThan60.length:)+(.+?)(,|)+", text)
        text_length_list = [int(w[1]) for w in text_length]
        SegLargeThan60_length_list = [int(w[1]) for w in SegLargeThan60_length]
        SegShortLargeThan60_length_list = [int(w[1]) for w in SegShortLargeThan60_length]
        total_text = sum(text_length_list)
        ori = sum(SegLargeThan60_length_list)
        after = sum(SegShortLargeThan60_length_list)

        print("total_text:{}, ori:{}, after:{}", total_text, ori, after)
        print("对长度大于 60 就直接丢弃：{}", ori / total_text)
        print("基于规则的断句：{}", after / total_text)

        """

        """
        text_origin = re.findall(r"(##SegSentenceForJieba## text_origin:)+(.+?)(2021-03-12)", text, re.S)
        textListSplit = re.findall(r"(##SegSentenceForJieba## textListSplit:)+(.+?)(2021-03-12)", text, re.S)
        textList = re.findall(r"(##SegSentenceForJieba## textList:)+(.+?)(2021-03-12)", text, re.S)
        data_dict = {}
        data_dict["text_origin"] = [w[1] for w in text_origin]
        data_dict["textListSplit"] = [w[1][1:-2].split(",") for w in textListSplit]
        data_dict["textList"] = [w[1][1:-2].split(",") for w in textList]
        data = pd.DataFrame(data_dict)
        data["check_is_same_length"] = data.apply(lambda row: self.check_is_same_length(
            col1=row["textListSplit"],
            col2=row["textList"]), axis=1)
        data_select = data[(data["check_is_same_length"] == True)]
        data_select["cut_res"] = data_select["text_origin"].apply(lambda x: ss.cut(x))
        # data_select.loc[:, "cut_res"] = data_select[:, "text_origin"].apply(lambda x: ss.cut(x))

        flat_list = [item for sublist in data_select["cut_res"].tolist() for item in sublist]
        # select_list = [w for w in flat_list if 30 <= len(w) < 60]
        res_list = [w for w in flat_list if len(w) > 60]

        print("需要用算法再次切分的文章数为：{}", len(data_select))
        print("用算法切分后总的句子数量为：{}", len(flat_list))
        print("用算法切分后，长度仍然大于60的句子数量为：{}", len(res_list))
        print("基于算法的断句：{}", len(res_list) / total_text)
        # with open("/Users/apple/XHSworkspace/data/210312/res_jieba_list_dayu60.pickle", "w") as f:
        #     f.write("\n".join(res_list))

        return data

    def check_is_same_length(self, col1, col2):
        return True if len(col1) > 0 and len(col1) > len(col2) else False


if __name__ == '__main__':
    ss = SegSentence()
    """test cut one sentence
    """
    # res = ss.cut(
    #     "然后说手机1800 让我先首付400然后分期 然后他说他先给我寄过来 确实单号物流都有 但是都是假的 我没有查到物流信息 他还硬说到了事后拿到钱之后直接不理人 微信删了小红书也删了 就QQ留着 QQ也不知道设置了啥 反正发了很多消息打了很多电话硬是不接, 他还硬说到了事后拿到钱之后直接不理人 微信删了小红书也删了 就QQ留着 QQ也不知道设置了啥 反正发了很多消息打了很多电话硬是不接")
    # res = ss.cut(
    #     """cos的衣服穿着穿着，就成为cos野模了！😂
    #
    #     🌈:上装推荐
    #     新款圆领衬衫，杂志推荐款！👈
    #     手感是很舒服的棉，很薄，上身感觉完全是可以在夏天穿的！🍦(我脖子短，圆领就是我的衣服了！略透，入手要常备rt！)
    #     杭州的夏天可是要到10月中旬才结束！秋天时候加个外套，估计也很帅🤪
    #
    #     🌈:下装推荐
    #     新款的薄型羊毛裤(藏青色的)，版型超级好，狂显瘦！
    #     西装裤的类型，休闲穿配小白鞋，正装就配个皮鞋啥的，这裤子是有西装配套的！(下次更新哈，超帅！)
    #     ⚠️:夏天穿羊毛不热吗？谢谢，cos家做的薄羊毛系列真的夏天可以穿。当然，为了好看我愿意天天待空调间 嘻嘻.🤪
    #
    #     🌈：牛皮小白鞋
    #     cos牛皮小白鞋，年年出，年年卖断货！现在店里还有同款黑色的，我个人更中意白色的哈！我穿40的刚好，牛皮料子，不贵不贵.🤤
    #
    #     好的，我说完了🤪有兴趣可以去了解下，还有请大家看完能赏个小心心❤️鞠躬.
    #     Collection of Style[品牌]##试衣间自拍[话题]##COLLECTION OF STYLE[地点]#"""
    # )
    # print(res)
    # lines = read_file("/Users/apple/XHSworkspace/data/structure/food/000000_0.csv_cutDoc_20210929")
    lines = read_file("/Users/apple/XHSworkspace/data/structure/food/15th_1000_1008_pred_fresh")
    for i in tqdm(lines):
        # i = "5f6901e00000000001002d49/t /t 包菜炒鸡/t /t 原料:鸡腿,包菜,干香菇 ,小葱,蒜,豆瓣酱,生抽,盐,麻油/t /t 准备:鸡腿砍成小块,过水后洗去浮沫;包菜切大块;干香菇提前泡软,斜刀切片;葱蒜切末。 ##SEP## /t /t 做法:油热后放葱末爆香,放豆瓣酱炒出红油后放入焯过水的鸡块和香菇片,稍稍炒一下加一丢丢开水半没过食材就好,加生抽提味(可以加点老抽上色), ##SEP## 中 "
        # i = "5f8c0c170000000001006772/t /t 湖州美食 良心宝藏外卖店铺推荐(第二期)/t /t 最真实的照片了,因为懒得p图。虽然有点丑好吃就行了!请相信我好吗?/t 宝藏店铺一:拼一碗蛋包饭/t 推荐:香浓咖喱蛋包饭配奥尔良腿排/t ##SEP## 均价:15/t 评价:很好吃的蛋包饭,还送了一小袋肉松,加上了肉松口感很棒,饭量很足,下面的饭也很香,随意点不踩雷/t 宝藏店铺二:小当家/t ##SEP## 推荐:干锅鸡套餐/t 宫保鸡丁套餐/t 均价:11/t 评价:用了红包才十块多,里面的配菜还可以随便选超级超级划算了,味道真的很赞, ##SEP## 鱼香肉丝和宫保鸡丁的口感一样就是里面的肉不一样,肥牛的那个也很好吃哦!性价比超高,超级推荐这家!就是配送有点不准时大部分还是准时的。 ##SEP## /t 宝藏店铺三:老俞木炭烧饼/t 推荐:香酥鸡烧饼/t 均价:13/t 评价:香酥鸡份量很足,一定要加甜酱一定要加甜酱!吃过卢记的,小俞的, ##SEP## 还是觉得这家最好吃!缺点配送费有点贵,可以和室友一起点哈哈哈!冲鸭!/t 因为不同的菜价钱也不同,均价只能做参考,红包津贴减掉后价格也不一样, ##SEP## 具体的可以去美团或者饿了吗自己看看哦!每个人口味不同,仅供参考哦!喜欢的点点关注,外卖不迷路!"
        i = i.replace("/t /t ", "/t")
        i = i.replace("/t/t/t ", "/t")
        i = i.replace(" ##SEP## ", " ")
        id = i.split("/t")[0]
        text = i[len(id) + 2:]
        text = text.replace("/t", "\t")
        res = ss.cut(text)
        for jdx, jval in enumerate(res):
            if len(jval) < 10 or len(jval) > 120:
                print("ori_cut: {}".format(text))
                print("new_cut")
                print("index: {}, value: {}".format(jdx, jval))
