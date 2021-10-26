# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 10:46
# @Author  : heyee (jialeyang)


# -*- coding: utf-8 -*-
# @Time    : 2021/9/7 15:50
# @Author  : heyee (jialeyang)
import copy
import os.path as osp
import os
from collections import defaultdict
import json
# from eval_performance import *
from tqdm import tqdm
import pandas as pd
from utils.logger_helper import logger
from segsentence import SegSentence
from collections import Counter
from data_utils import *
from sklearn.model_selection import train_test_split

tqdm.pandas()


class EntityRuleExtractor:

    def __init__(self):
        self.priority = ['龚宇娟', '王悦', '黄佳妮', '万婷茹', '赵鑫梅', '王强', '汪昕', '王念慈', '汪圣皓', '邱震宇', '万峻豪',
                         "李松", "刘扬康", "李繁", "刘沛", "王婷婷"]
        self.high_quality_id = defaultdict(list)
        self.org = json.load(
            open('/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/red_tree_food_concept_v28.json', 'r',
                 encoding='utf-8'))
        self.remove_note_id = []

    def extract_keywords(self, data):
        """ 解析 jieba 分词结果
        {'type': 'FOOD', 'text': '咖啡', 'startPosition': 4, 'endPosition': 6}
        """
        df = pd.DataFrame(data=data, columns=["source"])
        # df = pd.DataFrame(data=data, columns=["text"])
        df["note_id"] = df["source"].apply(lambda x: x.split("/t/t")[0])
        df["text"] = df["source"].apply(lambda x: x.split("/t/t")[1])
        df["ner"] = df["text"].progress_apply(parse_tags)
        df["content"] = df["text"].progress_apply(remove_all_tag)
        df["ner_allocate_type"] = df["ner"].progress_apply(self.allocate_type)
        """ 双指针寻找最长序列 """
        df["longest_match"] = df["ner_allocate_type"].progress_apply(self.find_longest_match)
        df["not_longest_match"] = df.progress_apply(
            lambda row: [w for w in row["ner_allocate_type"] if
                         w not in [item for sublist in row["longest_match"] for item in sublist]], axis=1)
        """ 统计 """
        self.LONGEST_MATCH = []
        self.NOT_LONGEST_MATCH = []
        df["longest_match"].progress_apply(self.collect_type)
        df["not_longest_match"].progress_apply(self.collect_type)
        longest_match_counter = Counter(self.LONGEST_MATCH)
        not_longest_match_counter = Counter(self.NOT_LONGEST_MATCH)
        """ 规则 
        多个实体：
        1。如果常见菜式在第一个位置 => 常见菜式
        2。如果不在第一个位置 => 食品一部分
        """
        df['re_longest_match'], df['re_not_longest_match'] = zip(*df.progress_apply(
            lambda row: self.re_allocate_type(
                list1=row["longest_match"],
                list2=row["not_longest_match"]
            ), axis=1))
        self.LONGEST_MATCH = []
        self.NOT_LONGEST_MATCH = []
        df["re_longest_match"].progress_apply(self.collect_type)
        df["re_not_longest_match"].progress_apply(self.collect_type)
        re_longest_match_counter = Counter(self.LONGEST_MATCH)
        re_not_longest_match_counter = Counter(self.NOT_LONGEST_MATCH)
        df["new_food"] = df["re_longest_match"].progress_apply(self.generate_new_food)
        df["new_food_other"] = df["re_not_longest_match"].progress_apply(self.generate_new_food_other)
        df['combine_and_sort'] = df.progress_apply(
            lambda row: self.combine_and_sort(
                list1=row["new_food"],
                list2=row["new_food_other"]
            ), axis=1)

        """ 生成训练数据 """

        def select_type(pred):
            res = []
            select_list = ['主要工艺', '食疗食补', '食品', '食材', '水果', '限定小吃', '限定菜系', '菜品口味', '工具', '适宜人群', '品牌', '常见菜式']
            for i in pred:
                if i.get("type") in select_list:
                    res.append(i)
            res = sorted(res, key=lambda k: k['startPosition'])
            return res

        df["select_type"] = df["combine_and_sort"].progress_apply(select_type)

        def re_tagging_tag_list(row, chunks):
            prefix = ""
            lastfix = 0
            for idx, val in enumerate(row):
                prefix += "".join(chunks[lastfix:val["startPosition"]])
                prefix = prefix + "[" + val["text"] + "]" + "_" + val["type"] + "_"
                lastfix = val["endPosition"]
            prefix += "".join(chunks[lastfix:])
            return prefix

        df["re_tagging_sentence"] = df.progress_apply(
            lambda row: re_tagging_tag_list(
                row=row["select_type"],
                chunks=row["content"]
            ), axis=1)

        # self.write_file(
        #     file="/Users/apple/XHSworkspace/data/structure/food/dataset/000000_0.csv_cutDoc_exp_test_20210831_base_model_plain_val2_jieba_entity_match",
        #     data=df["re_tagging_sentence"].tolist())
        self.write_file(
            file="/Users/apple/XHSworkspace/data/structure/food/tagging/food_20210901_passentity_match_jieba_entity_match",
            data=df["note_id"].str.cat([df["re_tagging_sentence"]], sep='/t/t').tolist())
        return df

    def reallocate_data_type(self, ner_d, ner_allocate_type, verify):
        ner_copy = copy.deepcopy(ner_allocate_type)
        for i in ner_allocate_type:
            if i["type"] in ["主要工艺"]:
                print(ner_d)
                print("before:::::::::::::::::::::" + str(i))
                j = copy.deepcopy(i)
                ner_copy.remove(j)
                j["type"] = "工艺"
                ner_copy.append(j)
                print("after:::::::::::::::::::::" + str(j))
            elif i["type"] in ["食疗食补"]:
                print(ner_d)
                print("before:::::::::::::::::::::" + str(i))
                j = copy.deepcopy(i)
                ner_copy.remove(j)
                j["type"] = "功效"
                ner_copy.append(j)
                print("after:::::::::::::::::::::" + str(j))
            elif i["type"] in ["菜品口味"]:
                print(ner_d)
                print("before:::::::::::::::::::::" + str(i))
                j = copy.deepcopy(i)
                ner_copy.remove(j)
                j["type"] = "口味"
                ner_copy.append(j)
                print("after:::::::::::::::::::::" + str(j))
            elif i["type"] in ["食品"]:
                for key, val in verify.items():
                    if i["text"] in val and key == "美食概念词":
                        print(ner_d)
                        print("before:::::::::::::::::::::" + str(i))
                        j = copy.deepcopy(i)
                        ner_copy.remove(j)
                        j["type"] = "美食概念词"
                        ner_copy.append(j)
                        print("after:::::::::::::::::::::" + str(j))
                        break
            elif i["type"] in ["水果"]:
                for key, val in verify.items():
                    if key == "美食概念词":
                        if i["text"] in val:
                            print(ner_d)
                            print("before:::::::::::::::::::::" + str(i))
                            j = copy.deepcopy(i)
                            ner_copy.remove(j)
                            j["type"] = "美食概念词"
                            ner_copy.append(j)
                            print("after:::::::::::::::::::::" + str(j))
                            break
                        else:
                            print(ner_d)
                            print("before:::::::::::::::::::::" + str(i))
                            j = copy.deepcopy(i)
                            ner_copy.remove(j)
                            j["type"] = "食品"
                            ner_copy.append(j)
                            print("after:::::::::::::::::::::" + str(j))
                            break
            elif i["type"] in ["限定小吃", "限定菜系"]:
                print(ner_d)
                print("before:::::::::::::::::::::" + str(i))
                j = copy.deepcopy(i)
                ner_copy.remove(j)
                j["type"] = "美食概念词"
                ner_copy.append(j)
                print("after:::::::::::::::::::::" + str(j))
            elif i["type"] in ["常见菜式"]:
                for key, val in verify.items():
                    if key == "美食概念词":
                        if i["text"] in val:
                            print(ner_d)
                            print("before:::::::::::::::::::::" + str(i))
                            j = copy.deepcopy(i)
                            ner_copy.remove(j)
                            j["type"] = "美食概念词"
                            ner_copy.append(j)
                            print("after:::::::::::::::::::::" + str(j))
                            break
                        else:
                            print(ner_d)
                            print("before:::::::::::::::::::::" + str(i))
                            j = copy.deepcopy(i)
                            ner_copy.remove(j)
                            j["type"] = "食品"
                            ner_copy.append(j)
                            print("after:::::::::::::::::::::" + str(j))
                            break
        return ner_copy

    def process_pkl_file(self):
        """
        TODO:
            0、校验：NUL 情形
            1、check_tagging_sense = True
            2、校验：select_type 单字 case，考虑删除
            3、校验：select_type 是否 match re_tagging_sentence
            4、校验：re_tagging_sentence 是否是 re_tagging_sense 的子句
            5、同一笔记 id 多个标注：按标注人员排名顺序指定一个
            6、这个系统里，所有“牛油”都是“黄油”
        """
        dir_path = "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_rel4_pkl"
        path_list = os.listdir(dir_path)
        pdList = []  # List of your dataframes
        for i in path_list:
            if i.startswith("food_") and i.endswith(".pkl"):
                data_id = int(i.split("_")[1])
                if data_id > 3171:
                # if data_id > 0:
                    tmp = pd.read_pickle(osp.join(dir_path, i))
                    tmp["pkl_from"] = i
                    pdList.append(tmp)
        new_df = pd.concat(pdList)
        # new_df = pd.read_pickle(
        #     "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/test_v3_json_txt.pkl")  # 测试集修正
        df = new_df[new_df["check_tagging_sense"] == True]

        verify = {
            "功效": read_file("/Users/apple/XHSworkspace/data/structure/food/config/red_tree_food_v8.json_功效"),
            "口味": read_file("/Users/apple/XHSworkspace/data/structure/food/config/red_tree_food_v8.json_口味"),
            "工具": read_file("/Users/apple/XHSworkspace/data/structure/food/config/red_tree_food_v8.json_工具"),
            "美食概念词": read_file("/Users/apple/XHSworkspace/data/structure/food/config/red_tree_food_v8.json_美食概念词"),
            "工艺": read_file("/Users/apple/XHSworkspace/data/structure/food/config/red_tree_food_v8.json_工艺")
        }

        def check_len(select_type):
            res = []
            for i in select_type:
                if len(i["text"]) <= 1:
                    res.append(i)
            return res

        df["check_len"] = df["select_type"].progress_apply(check_len)
        res = defaultdict(list)
        for i in df["check_len"].tolist():
            if len(i) > 0:
                for j in i:
                    if j["text"] not in res[j["type"]] and j["text"] not in self.org.get(j["type"], []):
                        res[j["type"]].append(j["text"])

        def remove_single_word(ner_d, select_type):
            sel_copy = copy.deepcopy(select_type)
            remove_dict = {
                "工艺": ['画', '片', '碗', '摸', '盐', '制'],
                '食材': ['泡', '嫩', '蘑', '核', '抹', '薯', '炒', '炖', '辣', '煮'],
                '食品': ['烧', '脆', '咖', '圈', '不', '片', '晚']
            }
            keep_dict = {
                "美食概念词": [w for w in self.org.get("美食概念词", []) if len(w) == 1],
                "否定修饰": [w for w in self.org.get("否定修饰", []) if len(w) == 1],
                "口味": [w for w in self.org.get("口味", []) if len(w) == 1],
                "工具": [w for w in self.org.get("工具", []) if len(w) == 1]
            }
            for i in select_type:
                if len(i["text"]) <= 1:
                    if i["type"] == "工艺" and i["text"] in remove_dict["工艺"]:
                        logger.info("::::remove:::{}::::::::::::::".format(ner_d))
                        sel_copy.remove(i)
                        logger.info("::::remove:::{}::::::::::::::".format(i["type"]) + str(i))
                        logger.info("-" * 200)
                    elif i["type"] == "食材" and i["text"] in remove_dict["食材"]:
                        logger.info("::::remove:::{}::::::::::::::".format(ner_d))
                        sel_copy.remove(i)
                        logger.info("::::remove:::{}::::::::::::::".format(i["type"]) + str(i))
                        logger.info("-" * 200)
                    elif i["type"] == "食品" and i["text"] in remove_dict["食品"]:
                        logger.info("::::remove:::{}::::::::::::::".format(ner_d))
                        sel_copy.remove(i)
                        logger.info("::::remove:::{}::::::::::::::".format(i["type"]) + str(i))
                        logger.info("-" * 200)
                    elif i["type"] == "美食概念词" and i["text"] not in keep_dict["美食概念词"]:
                        logger.info("::::remove:::{}::::::::::::::".format(ner_d))
                        sel_copy.remove(i)
                        logger.info("::::remove:::{}::::::::::::::".format(i["type"]) + str(i))
                        logger.info("-" * 200)
                    elif i["type"] == "否定修饰" and i["text"] not in keep_dict["否定修饰"]:
                        logger.info("::::remove:::{}::::::::::::::".format(ner_d))
                        sel_copy.remove(i)
                        logger.info("::::remove:::{}::::::::::::::".format(i["type"]) + str(i))
                        logger.info("-" * 200)
                    elif i["type"] == "口味" and i["text"] not in keep_dict["口味"]:
                        logger.info("::::remove:::{}::::::::::::::".format(ner_d))
                        sel_copy.remove(i)
                        logger.info("::::remove:::{}::::::::::::::".format(i["type"]) + str(i))
                        logger.info("-" * 200)
                    elif i["type"] == "工具" and i["text"] not in keep_dict["工具"]:
                        logger.info("::::remove:::{}::::::::::::::".format(ner_d))
                        sel_copy.remove(i)
                        logger.info("::::remove:::{}::::::::::::::".format(i["type"]) + str(i))
                        logger.info("-" * 200)
                    elif i["type"] in ["品牌", "功效"]:
                        logger.info("::::remove:::{}::::::::::::::".format(ner_d))
                        sel_copy.remove(i)
                        logger.info("::::remove:::{}::::::::::::::".format(i["type"]) + str(i))
                        logger.info("-" * 200)
            return sel_copy

        # df["select_type2"] = df["select_type"].progress_apply(remove_single_word)
        df["select_type2"] = df.progress_apply(
            lambda row: remove_single_word(
                ner_d=row["ner_d"],
                select_type=row["select_type"],
            ), axis=1)

        """ 查看新词效果 """
        # res2 = defaultdict(list)
        # for i in tqdm(df["select_type2"].tolist()):
        #     if len(i) > 0:
        #         for j in i:
        #             if j["text"] not in res2[j["type"]] and j["text"] not in self.org.get(j["type"], []):
        #                 res2[j["type"]].append(j["text"])

        df["select_type2"] = df.progress_apply(
            lambda row: self.verify_data(
                ner_d=row["ner_d"],
                ner_allocate_type=row["select_type2"],
                verify=verify
            ), axis=1)

        df["select_type2"] = df.progress_apply(
            lambda row: self.verify_data_remove(
                ner_d=row["ner_d"],
                ner_allocate_type=row["select_type2"],
                verify=verify,
                note_id=row["note_id"]
            ), axis=1)

        # verify_city = {
        #     "LOC": read_file("/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/city")
        # }
        df["select_type2"] = df.progress_apply(
            lambda row: self.remove_de_in_taste(
                ner_d=row["ner_d"],
                ner_allocate_type=row["select_type2"],
                verify={}
            ), axis=1)

        """ 生成训练数据 """

        def select_type(pred):
            res = []
            select_list = ['工艺', '功效', '食品', '食材', '口味', '工具', '适宜人群', '品牌', '美食概念词', "否定修饰"]
            for i in pred:
                if i.get("type") in select_list:
                    res.append(i)
            res = sorted(res, key=lambda k: k['startPosition'])
            return res

        df["select_type2"] = df["select_type2"].progress_apply(select_type)

        def re_tagging_tag_list(row, chunks):
            prefix = ""
            lastfix = 0
            for idx, val in enumerate(row):
                prefix += "".join(chunks[lastfix:val["startPosition"]])
                prefix = prefix + "[" + val["text"] + "]" + "_" + val["type"] + "_"
                lastfix = val["endPosition"]
            prefix += "".join(chunks[lastfix:])
            return prefix

        df["re_tagging_sentence2"] = df.progress_apply(
            lambda row: re_tagging_tag_list(
                row=row["select_type2"],
                chunks=row["content"]
            ), axis=1)

        df["check_tagging_sense2"] = df["re_tagging_sentence2"].progress_apply(check_tagging_sense)
        df = df[df["check_tagging_sense2"] == True]

        df["re_tagging_rel2"] = df.progress_apply(
            lambda row: self.reconcat_rel(
                note_id=row["note_id"],
                data_ori=row["data_ori"],
                parsed_res_relation=row["parsed_res_relation"],
                re_tagging_sentence=row["re_tagging_sentence2"]
            ), axis=1)

        # write_file(
        #     file="/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/test_v3_pkl.json_txt",
        #     data=df["re_tagging_rel2"].tolist())  # 测试集修正

        final_res = defaultdict(list)
        final_res_remove = defaultdict(list)
        for index, row in df.iterrows():
            note_id = row["note_id"]
            user = row["userName"]
            text = row["content"]
            ner = row["select_type2"]
            ner_d = row["re_tagging_sentence2"]
            relation = row["re_tagging_rel2"]
            org_data = note_id + "/t/t" + ner_d
            # if note_id in final_res.keys():
            #     """ 根据 质检结果&一致率结果 挑选高质量数据 """
            #     # if note_id in self.high_quality_id:
            #     #     if user not in self.high_quality_id[note_id]:
            #     #         final_res.pop(note_id)
            #     #         continue
            #     if user not in self.priority:
            #         continue
            #     else:
            #         pre_user = final_res[note_id]["user"]
            #         if pre_user not in self.priority:
            #             continue
            #         elif self.priority.index(user) < self.priority.index(pre_user):
            #             final_res.pop(note_id)
            new_note_id = note_id + "#" + str(index)
            # if user in self.priority:
            if note_id not in self.remove_note_id:
                final_res[new_note_id] = {
                    "note_id": note_id,
                    "org_data": org_data,
                    "user": user,
                    "text": text,
                    "ner": ner,
                    "ner_d": ner_d,
                    "relation": relation
                }
            else:
                final_res_remove[note_id] = {
                    "note_id": note_id,
                    "org_data": org_data,
                    "user": user,
                    "text": text,
                    "ner": ner,
                    "ner_d": ner_d,
                    "relation": relation
                }
        with open(
                '/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_rel4_pkl/20211018_ner_nodup_21p_3171_3196.json',
                'w', encoding='utf-8') as fp:
            json.dump(final_res, fp, ensure_ascii=False, indent=4)
        with open(
                '/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_rel4_pkl/20211018_ner_nodup_21p_3171_3196_remove_note_id.json',
                'w', encoding='utf-8') as fp:
            json.dump(final_res_remove, fp, ensure_ascii=False, indent=4)
        print(final_res)

    def generate_train_data(self):
        org = json.load(open(
            # '/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/20210928_ner.json',
            '/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_rel4_pkl/20211017_ner_nodup_21p.json',
            'r', encoding='utf-8'))
        # org = read_file("/Users/apple/XHSworkspace/data/structure/food/11th_double_blind_1_new")
        test_case = json.load(
            open('/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/test_v3.json',
                 'r', encoding='utf-8'))

        """ 原断句算法：##SEP## """
        lines = []
        for i in org:
            # if org[i]["note_id"] not in test_case and org[i]["user"] in self.priority:
            if org[i]["note_id"] not in test_case:
                # if org[i]["note_id"] not in test_case:
                # if org[i]["user"] in self.priority:
                # if i not in test_case:
                tmp = org[i]["ner_d"].strip("\'")
                tmp = tmp.replace("/t", " ")
                tmp = tmp.replace(" ##SEP## ", " ##sep## ")
                lines.extend(tmp.split(" ##sep## "))
                # lines.append(i + "/t/t" + tmp)  # 测试集校正
        print(len(lines))

        """ 单独增加：dup3038_520.json """
        dup3038_520 = json.load(open(
            '/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_david_csv/dup3038_520_json_txt_pkl.json',
            'r', encoding='utf-8'))
        for i in dup3038_520:
            if dup3038_520[i]["note_id"] not in test_case:
                tmp = dup3038_520[i]["ner_d"].strip("\'")
                tmp = tmp.replace("/t", " ")
                tmp = tmp.replace(" ##SEP## ", " ##sep## ")
                lines.extend(tmp.split(" ##sep## "))
        print(len(lines))

        """ 单独增加：20211018_ner_nodup_21p_3171_3196.json """

        org3 = json.load(open(
            '/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_rel4_pkl/20211018_ner_nodup_21p_3171_3196.json',
            'r', encoding='utf-8'))
        for i in org3:
            if org3[i]["note_id"] not in test_case:
                tmp = org3[i]["ner_d"].strip("\'")
                tmp = tmp.replace("/t", " ")
                tmp = tmp.replace(" ##SEP## ", " ##sep## ")
                lines.extend(tmp.split(" ##sep## "))
        print(len(lines))

        """ 新断句算法: \t """
        # lines = []
        # for i in org:
        #     if org[i]["note_id"] not in test_case and org[i]["user"] in self.priority:
        #         # if org[i]["note_id"] not in test_case:
        #         # if org[i]["user"] in self.priority:
        #         # if i not in test_case:
        #         tmp = org[i]["ner_d"].strip("\'")
        #         tmp = tmp.replace("/t", "\t")
        #         tmp = tmp.replace(" ##SEP## ", "")
        #         tmp = tmp.replace(" ##sep## ", "")
        #         sen_collect = ss.cut(tmp)
        #         sen_collect = [w.replace("\t", " ") for w in sen_collect]
        #         lines.extend(sen_collect)
        #         # lines.append(i + "/t/t" + tmp)  # 测试集校正
        #
        # for i in self.high_quality_id:
        #     if i not in test_case:
        #         for tmp in self.high_quality_id[i]:
        #             tmp = tmp[len(i) + 4:]
        #             tmp = tmp.replace("/t", "\t")
        #             tmp = tmp.replace(" ##SEP## ", "")
        #             tmp = tmp.replace(" ##sep## ", "")
        #             sen_collect = ss.cut(tmp)
        #             sen_collect = [w.replace("\t", " ") for w in sen_collect]
        #             lines.extend(sen_collect)

        train, test = train_test_split(lines, test_size=0.1, random_state=42, shuffle=True)
        write_file(
            file="/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_rel4_pkl/20211017/20211018_ner_nodup_21p_train.txt",
            data=train)
        write_file(
            file="/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_rel4_pkl/20211017/20211018_ner_nodup_21p_val.txt",
            data=test)
        print("done")
        # with open('/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/test_v龚宇娟.json', 'w',
        #           encoding='utf-8') as fp:
        #     json.dump(lines, fp, ensure_ascii=False, indent=4)

    def type_14_to_10(self):
        """
        TODO:
            0、分析相连实体
            1、标签变化：
                 old: ['主要工艺', '食疗食补', '食品', '食材', '水果', '限定小吃', '限定菜系', '菜品口味', '工具', '适宜人群', '品牌', '常见菜式'] past -> 20210909
                 new: ['工艺', '功效', '食品', '食材', '口味', '工具', '适宜人群', '品牌', '美食概念词'] 20210913 -> now
                         主要工艺 => 工艺
                         食疗食补 => 功效
                         食品 + 水果 => 食品 + 美食概念词
                         食材 => 食材
                         限定小吃 + 限定菜系 => 美食概念词
                         菜品口味 => 口味
                         工具 => 工具
                         适宜人群 => 适宜人群
                         品牌 => 品牌
                         常见菜式 => 食品 + 美食概念词
            2、标签约束：
                （x）食品：不包含地域
                （2）口味不含"的"：甜甜的
                （x）食材：最小序列，咸味蛋黄 => 咸味+蛋黄
                （x）食品/食材均不含品牌
                 (5) 美食概念词与食品/食材冲突
                 (6) 口味与工艺冲突

        """
        dir_path = "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_david_csv"
        # data = parse_from_csv(osp.join(dir_path, "美食_正式标注_1200_17th_1011_西安_全部_20211012105250.csv"))  # 截至 3168: 20211017_ner_nodup_21p.json
        data = parse_from_csv(osp.join(dir_path, "美食_正式标注_1400_21th_1015_西安_全部_20211018113101.csv"))
        # data = parse_from_file(
        # "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/debug")  # 测试集修正
        # "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/test_v3.json_txt")  # 测试集修正
        data_dict = {
            "note_id": [w["note_id"] for w in data],
            "userName": [w.get("userName", "") for w in data],
            "content": [w["plain_text"] for w in data],
            "ner_d": [w["ner_d"] for w in data],
            "ner_allocate_type": [w["parsed_res"] for w in data],
            "data_ori": [w["data_ori"] for w in data],
            # "data_ori": [w["org_data"] for w in data],  # 测试集修正
            "parsed_res_relation": [w["parsed_res_relation"] for w in data]
        }
        df = pd.DataFrame(data_dict)
        verify = {
            "功效": read_file("/Users/apple/XHSworkspace/data/structure/food/config/red_tree_food_v8.json_功效"),
            "口味": read_file("/Users/apple/XHSworkspace/data/structure/food/config/red_tree_food_v8.json_口味"),
            "工具": read_file("/Users/apple/XHSworkspace/data/structure/food/config/red_tree_food_v8.json_工具"),
            "美食概念词": read_file("/Users/apple/XHSworkspace/data/structure/food/config/red_tree_food_v8.json_美食概念词"),
            "工艺": read_file("/Users/apple/XHSworkspace/data/structure/food/config/red_tree_food_v8.json_工艺")
        }
        """ 标签变化: type_14_to_10 """
        df["ner_allocate_type"] = df.progress_apply(
            lambda row: self.reallocate_data_type(
                ner_d=row["ner_d"],
                ner_allocate_type=row["ner_allocate_type"],
                verify=verify
            ), axis=1)
        """ 分析相连实体 """
        df["longest_match"] = df["ner_allocate_type"].progress_apply(self.find_longest_match)
        df["not_longest_match"] = df.progress_apply(
            lambda row: [w for w in row["ner_allocate_type"] if
                         w not in [item for sublist in row["longest_match"] for item in sublist]], axis=1)
        """ 校验数据：
        食品/食材 => 美食概念词 
        工艺 => 口味
        """

        # df["ner_allocate_type_fix"] = df.progress_apply(
        #     lambda row: self.verify_data(
        #         ner_d=row["ner_d"],
        #         ner_allocate_type=row["ner_allocate_type"],
        #         verify=verify
        #     ), axis=1)
        #
        # df["ner_allocate_type_fix"] = df.progress_apply(
        #     lambda row: self.verify_data_remove(
        #         ner_d=row["ner_d"],
        #         ner_allocate_type=row["ner_allocate_type_fix"],
        #         verify=verify,
        #         note_id=row["note_id"]
        #     ), axis=1)

        # verify_city = {
        #     "LOC": read_file("/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/city")
        # }
        df["ner_allocate_type_fix"] = df.progress_apply(
            lambda row: self.remove_de_in_taste(
                ner_d=row["ner_d"],
                ner_allocate_type=row["ner_allocate_type"],
                verify={}
            ), axis=1)

        """ 生成训练数据 """

        def select_type(pred):
            res = []
            select_list = ['工艺', '功效', '食品', '食材', '口味', '工具', '适宜人群', '品牌', '美食概念词', "否定修饰"]
            for i in pred:
                if i.get("type") in select_list:
                    res.append(i)
            res = sorted(res, key=lambda k: k['startPosition'])
            return res

        df["select_type"] = df["ner_allocate_type_fix"].progress_apply(select_type)

        def re_tagging_tag_list(row, chunks):
            prefix = ""
            lastfix = 0
            for idx, val in enumerate(row):
                prefix += "".join(chunks[lastfix:val["startPosition"]])
                prefix = prefix + "[" + val["text"] + "]" + "_" + val["type"] + "_"
                lastfix = val["endPosition"]
            prefix += "".join(chunks[lastfix:])
            return prefix

        df["re_tagging_sentence"] = df.progress_apply(
            lambda row: re_tagging_tag_list(
                row=row["select_type"],
                chunks=row["content"]
            ), axis=1)

        df["check_tagging_sense"] = df["re_tagging_sentence"].progress_apply(check_tagging_sense)
        df["re_tagging_rel"] = df.progress_apply(
            lambda row: self.reconcat_rel(
                note_id=row["note_id"],
                data_ori=row["data_ori"],
                parsed_res_relation=row["parsed_res_relation"],
                re_tagging_sentence=row["re_tagging_sentence"]
            ), axis=1)
        # df.to_csv(osp.join(dir_path, "food_57_rel6_2863_2_{}.csv".format(len(df))))
        # write_file(data=list(set(self.remove_note_id)),
        #            file="/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_david_csv/dup3038_520_json_txt.remove_note_id")
        # df.to_pickle(path=osp.join(
        #     "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_david_csv/dup3038_520_json_txt.pkl"))  # 测试集修正
        df.to_pickle(path=osp.join("/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_rel4_pkl",
                                   "food_3196_{}.pkl".format(len(df))))
        print(data)

    def reconcat_rel(self, note_id, data_ori, parsed_res_relation, re_tagging_sentence):
        text = ""
        text += note_id
        if "/t /t " in data_ori:
            re_tagging_sentence = re_tagging_sentence.replace("/t/t", "/t /t ")
            text += "/t /t " + re_tagging_sentence
        elif "/t/t/t " in data_ori:
            re_tagging_sentence = re_tagging_sentence.replace("/t/t", "/t/t/t ")
            text += "/t/t/t " + re_tagging_sentence
        else:
            i, j = 0, 0
            sep_text = ""
            while i < len(re_tagging_sentence) and j < len(data_ori):
                if re_tagging_sentence[i] != data_ori[len(note_id):][j]:
                    sep_text += data_ori[len(note_id):][j]
                    j += 1
                else:
                    break
            harf_sep = sep_text[:len(sep_text) // 2]
            text += sep_text + re_tagging_sentence.replace("/t", harf_sep)
        rel_co = copy.deepcopy(parsed_res_relation)
        for i in rel_co:
            tmp = copy.deepcopy(i)
            for key, val in tmp.items():
                if key == "s":
                    if val["type"] == "口味" and "的" in val["text"][-1]:
                        tmp[key]["text"] = tmp[key]["text"][:-1]
                        rel_co.remove(i)
                        rel_co.append({
                            "s": tmp[key],
                            "p": tmp["p"],
                            "o": tmp["o"]
                        })
                elif key == "o":
                    if val["type"] == "口味" and "的" in val["text"][-1]:
                        tmp[key]["text"] = tmp[key]["text"][:-1]
                        rel_co.remove(i)
                        rel_co.append({
                            "s": tmp["s"],
                            "p": tmp["p"],
                            "o": tmp[key]
                        })
        if len(rel_co) > 0:
            text += "***" + str(rel_co)
        return text

    def verify_data(self, ner_d, ner_allocate_type, verify):
        ner_copy = copy.deepcopy(ner_allocate_type)
        for i in ner_allocate_type:
            for key, val in verify.items():
                if i["text"] in val and i["type"] != key:
                    i_copy = copy.deepcopy(i)
                    # if i["type"] in ["品牌", "食品", "口味", "食材", "NUL", "不确定"] and key == "美食概念词":  # 工艺：炒菜
                    if i["type"] not in ["美食概念词", "工具"] and key == "美食概念词":  # 工艺：炒菜; 工具: 铁板
                        ner_copy.remove(i)
                        i["type"] = "美食概念词"
                        ner_copy.append(i)
                    # elif i["type"] in ["功效", "工艺", "NUL", "不确定"] and key == "口味":  # 品牌不能加：七分甜; 食材不能加：油; 食品不能加
                    elif i["type"] not in ["品牌", "食材", "食品",
                                           "工艺"] and key == "口味":  # 品牌不能加：七分甜; 食材不能加：油; 食品不能加; 工艺不能加: 烫
                        ner_copy.remove(i)
                        i["type"] = "口味"
                        ner_copy.append(i)
                    # elif i["type"] in ["美食概念词", "工艺", "功效", "口味", "食品", "食材", "NUL", "不确定"] and key == "功效":
                    elif i["type"] not in ["功效"] and key == "功效":
                        ner_copy.remove(i)
                        i["type"] = "功效"
                        ner_copy.append(i)
                    elif i["type"] not in ["工艺", "食品", "食材", "美食概念词", "口味", "工具", "功效", "品牌"] and key == "工艺":
                        ner_copy.remove(i)
                        i["type"] = "工艺"
                        ner_copy.append(i)
                    elif i["type"] not in ["工具", "食品", "食材", "工艺"] and key == "工具":
                        ner_copy.remove(i)
                        i["type"] = "工具"
                        ner_copy.append(i)
                    elif i["type"] in ["NUL", "不确定"]:
                        ner_copy.remove(i)
                        i["type"] = key
                        ner_copy.append(i)
                    # if key == "美食概念词" or i["type"] == "美食概念词":
                    #     continue
                    if i_copy != i:
                        logger.info(ner_d)
                        logger.info(key + ":::::::verify_data::::::::::::::" + str(i_copy))
                        logger.info(key + "::::::::verify_data:::::::::::::" + str(i))
                        logger.info("-" * 200)
                    #     print(ner_d)
                    #     print(key + ":::::::verify_data::::::::::::::" + str(i_copy))
                    #     print(key + "::::::::verify_data:::::::::::::" + str(i))
                    #     print("-" * 200)
        return ner_copy

    def verify_data_remove(self, ner_d, ner_allocate_type, verify, note_id):
        ner_copy = copy.deepcopy(ner_allocate_type)
        remove_dict = {
            "美食概念词": ["煮法", "摘法", "中厨", "碳水化合物", "种子", "方子", "花牛", "一食", "拼多多", "吃饭", "包吃", "减肥期", "棉花山居",
                      "广州清吧", "下饭", "快手", "快手又好吃", "无麸", "小红书", "不会", "共饮", "乳矿物质", "海底", "成都味道", "中厨", "碳循环", "聚会",
                      "抚州口味", "红烧", "甜甜的", "油炸", "非油炸", "客家", "煮饭", "冬日暖锅", "老盐系列", "京味儿", "系列茶叶", "品甜",
                      "煮法", "摘法", "食用方法", "美食做法", "制作方法", "烹调方法", "烧法", "吃法", "方法", "做饭", "做法", "喝法",
                      "金华蛋糕", "郑州蛋糕", "东莞蛋糕", "西安蛋糕", "宁波蛋糕", "昆明生日蛋糕", "绍兴蛋糕", "成都蛋糕", "网红主题蛋糕", "网红榴莲千层蛋糕", "网红羽毛月亮蛋糕", "南昌蛋糕", "蛋糕卷系列"
                      "料料", "花茶材料", "材料", "菜叶", "蘸辣酱", "奶盖系列", "泰式奶绿", "经典意式肉酱面", "lunch set",
                      "陪餐", "教程", "食谱", "周黑鸭", "set menu", "黄鸭造型菜", "/重庆火锅", "蛋白质", "维生素", "一人份", "一餐", "百菜", "药草",
                      "韩国", "杭帮", "食欲", "头盘", "果园", "韩系", "吃喝", "觅食", "一日", "烹饪", "重庆", "爆炒", "蔬菜盘", "主食盘", "上班族", "饮品杯", "主材碎", "中国菜谱", "家庭烤肉"]
        }
        shi_function = ["美式", "中式", "泰式", "英式", "菜式", "港式", "韩式", "法式", "京式", "西式", "日式", "意式", "广式",
                        "湘式", "中西式", "意大利式", "那不勒斯西式", "葡式", "缅式", "德式", "台式", "苏式", "欧陆式", "混浊美式"]
        for i in ner_allocate_type:
            if i["type"] == "美食概念词" and i["text"] not in verify.get("美食概念词", []):
                key_ttt = "美食概念词"
                logger.info(ner_d)
                logger.info(":::::::{}::::::::::::::".format(key_ttt) + str(i))
                if i["text"] in remove_dict["美食概念词"]:
                    ner_copy.remove(i)
                elif i["text"].startswith(tuple(shi_function)):
                    text = i["text"]
                    startPosition = i['startPosition']
                    endPosition = i['endPosition']
                    for j in shi_function:
                        if i["text"].startswith(j):
                            ner_copy.remove(i)
                            ner_copy.append({
                                'type': '美食概念词',
                                'text': j,
                                "startPosition": startPosition,
                                "endPosition": startPosition + len(j)
                            })
                            text_remain = text[len(j):]
                            remian_type = '美食概念词' if text_remain in verify.get("美食概念词", []) else "食品"
                            ner_copy.append({
                                'type': remian_type,
                                'text': text_remain,
                                "startPosition": startPosition + len(j),
                                "endPosition": endPosition
                            })
                            logger.info(
                                ":::{}::::split {} into {} and {}({})::::::".format(key_ttt, text, j, text_remain,
                                                                                    remian_type) + str(i))
                elif i["text"] in self.org.get("食品", []) or i["text"] in self.org.get("食材", []):
                    ner_copy.remove(i)
                    i["type"] = "食品"
                    ner_copy.append(i)
                    logger.info("::::::::{}:::::FOOD::::::::".format(key_ttt) + str(i))
                else:
                    logger.info("::::::::{}:::::::NOT::::::".format(key_ttt) + str(i))
                logger.info("-" * 200)
            elif i["type"] == "功效" and i["text"] not in verify.get("功效", []):
                key_ttt = "功效"
                logger.info(ner_d)
                logger.info(":::::::{}::::::::::::::".format(key_ttt) + str(i))
                ner_copy.remove(i)
                logger.info("-" * 200)
            elif i["type"] == "品牌":
                key_ttt = "品牌"
                if i["text"] not in self.org.get("品牌", []):
                    logger.info(ner_d)
                    logger.info(":::::::{}::::::::::::::".format(key_ttt) + str(i))
                    ner_copy.remove(i)
                    logger.info("-" * 200)
                # else:
                #     for brand in self.org.get("品牌", []):
                #         if i["text"].startswith(brand) and len(brand) < len(i["text"]):
                #             logger.info(":::::::{}::::brand::{}::::::::".format(key_ttt, brand) + str(i))
            elif i["type"] == "食品":
                key_ttt = "食品"
                if i["text"].startswith(tuple(shi_function)):
                    text = i["text"]
                    startPosition = i['startPosition']
                    endPosition = i['endPosition']
                    for j in shi_function:
                        if i["text"].startswith(j):
                            ner_copy.remove(i)
                            ner_copy.append({
                                'type': '美食概念词',
                                'text': j,
                                "startPosition": startPosition,
                                "endPosition": startPosition + len(j)
                            })
                            text_remain = text[len(j):]
                            remian_type = '美食概念词' if text_remain in verify.get("美食概念词", []) else "食品"
                            ner_copy.append({
                                'type': remian_type,
                                'text': text_remain,
                                "startPosition": startPosition + len(j),
                                "endPosition": endPosition
                            })
                            logger.info(":::{}::::split {} into {} and {}({})::::::".format(
                                key_ttt, text, j, text_remain, remian_type) + str(i))
                elif len(i["text"]) <= 2 and i["text"] not in self.org.get(key_ttt, []):
                    ner_copy.remove(i)
                    logger.info(":::::::{}:::LESS THAN 2::::NOTE_ID {}::::".format(key_ttt, "") + str(i))
                else:
                    # if len(i["text"]) >= 8 and i["text"] not in self.org.get(key_ttt, []):
                    #     # ner_copy.remove(i)
                    #     logger.info(":::::::{}:::LARGE THAN 8::::NOTE_ID {}::::".format(key_ttt, "") + str(i))
                    for brand in self.org.get("品牌", []):
                        if i["text"].startswith(brand) and len(brand) < len(i["text"]):
                            logger.info(
                                ":::::::{}::::brand::{}::::NOTE_ID {}::::".format(key_ttt, brand, note_id) + str(i))
                            self.remove_note_id.append(note_id)
            elif i["type"] == "食材":
                key_ttt = "食材"
                if i["text"].startswith(tuple(shi_function)):
                    text = i["text"]
                    startPosition = i['startPosition']
                    endPosition = i['endPosition']
                    for j in shi_function:
                        if i["text"].startswith(j):
                            ner_copy.remove(i)
                            ner_copy.append({
                                'type': '美食概念词',
                                'text': j,
                                "startPosition": startPosition,
                                "endPosition": startPosition + len(j)
                            })
                            text_remain = text[len(j):]
                            remian_type = '美食概念词' if text_remain in verify.get("美食概念词", []) else "食材"
                            ner_copy.append({
                                'type': remian_type,
                                'text': text_remain,
                                "startPosition": startPosition + len(j),
                                "endPosition": endPosition
                            })
                            logger.info(":::{}::::split {} into {} and {}({})::::::".format(
                                key_ttt, text, j, text_remain, remian_type) + str(i))
                elif len(i["text"]) <= 2 and i["text"] not in self.org.get(key_ttt, []):
                    ner_copy.remove(i)
                    logger.info(":::::::{}:::LESS THAN 2::::NOTE_ID {}::::".format(key_ttt, note_id) + str(i))
                else:
                    # if len(i["text"]) >= 6 and i["text"] not in self.org.get(key_ttt, []):
                    #     # ner_copy.remove(i)
                    #     logger.info(":::::::{}:::LARGE THAN 6::::NOTE_ID {}::::".format(key_ttt, "") + str(i))
                    for brand in self.org.get("品牌", []):
                        if i["text"].startswith(brand) and len(brand) < len(i["text"]):
                            logger.info(
                                ":::::::{}::::brand::{}::::NOTE_ID {}::::".format(key_ttt, brand, note_id) + str(i))
                            self.remove_note_id.append(note_id)
        return ner_copy

    def remove_city_in_food(self, ner_d, ner_allocate_type, verify):
        ner_copy = copy.deepcopy(ner_allocate_type)
        for i in ner_allocate_type:
            if i["type"] not in ["食品", "美食概念词", "品牌", "口味", "适宜人群"]:
                for key, val in verify.items():
                    for j in val:
                        if j in i["text"]:
                            print(ner_d)
                            print(key + "::::::::remove_city_in_food:::::::::::::" + str(i))

    def remove_de_in_taste(self, ner_d, ner_allocate_type, verify):
        ner_copy = copy.deepcopy(ner_allocate_type)
        for i in ner_allocate_type:
            if i["type"] in ["口味"]:
                if "的" in i["text"][-1]:
                    print(ner_d)
                    print("::::::::remove_de_in_taste:::::::::::::" + str(i))
                    j = copy.deepcopy(i)
                    ner_copy.remove(j)
                    j["text"] = i["text"][:-1]
                    j["endPosition"] = i["endPosition"] - 1
                    ner_copy.append(j)
                    print("::::::::remove_de_in_taste:::::::::::::" + str(j))
        return ner_copy

    def remove_prefix_in_shicai(self, ner_d, ner_allocate_type, verify):
        ner_copy = copy.deepcopy(ner_allocate_type)
        for i in ner_allocate_type:
            if i["type"] in ["食材"]:
                for key, val in verify.items():
                    for j in val:
                        if j in i["text"]:
                            print(ner_d)
                            print(key + ":::::::::::::::::::::" + str(i))

    def combine_and_sort(self, list1, list2):
        list_combine = []
        list_combine.extend(list1)
        list_combine.extend(list2)

        return list_combine

    def generate_new_food_other(self, ner_list):
        res = []
        for i in ner_list:
            type_list = i.get("type").split("|")
            X = ['限定小吃', '限定菜系', '品牌', '工具', '食疗食补', '水果', '主要工艺', '菜品口味', '适宜人群', '常见菜式', '食品', '食材']
            Y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            keydict = dict(zip(X, Y))
            type_list.sort(key=keydict.get)
            res.append({
                "type": type_list[0],
                "text": i.get("text"),
                "startPosition": i.get("startPosition"),
                "endPosition": i.get("endPosition")
            })
        return res

    def generate_new_food(self, ner_list):
        res = []
        for i in ner_list:
            tmp = []
            startPosition = float('inf')
            endPosition = -float('inf')
            for j in i:
                startPosition = min(startPosition, j.get("startPosition"))
                endPosition = max(endPosition, j.get("endPosition"))
                tmp.append(j.get("text"))
            res.append({
                "type": "食品",
                "text": "".join(tmp),
                "startPosition": startPosition,
                "endPosition": endPosition
            })
        return res

    def re_allocate_type(self, list1, list2):
        if len(list1) == 0:
            return list1, list2
        res1 = []
        res2 = list2
        for i in list1:
            tmp = []
            for jdx, j in enumerate(i):
                if jdx == 0 and ("常见菜式" in j.get("type") or
                                 "品牌" in j.get("type") or
                                 "适宜人群" in j.get("type") or
                                 "工具" in j.get("type")):
                    res2.append(j)
                    continue
                else:
                    tmp.append(j)
            if len(tmp) > 1:
                res1.append(tmp)
            else:
                res2.extend(tmp)
        return res1, res2

    def collect_type(self, input_list):
        if len(input_list) == 0:
            return []
        if any(isinstance(el, list) for el in input_list):
            input_list = [item for sublist in input_list for item in sublist]
            for i in input_list:
                self.LONGEST_MATCH.append(i.get("type"))
        else:
            for i in input_list:
                self.NOT_LONGEST_MATCH.append(i.get("type"))

    def allocate_type(self, ner_list):
        if len(ner_list) == 0:
            return []
        res = []
        for i in ner_list:
            type_list = []
            for j in self.org.keys():
                if i.get("text") in self.org[j]:
                    type_list.append(j)
            if len(type_list) == 0:
                print(i)
                continue
            res.append({
                "type": "|".join(list(set(type_list))),
                "text": i.get("text"),
                "startPosition": i.get("startPosition"),
                "endPosition": i.get("endPosition")
            })
        return res

    def find_longest_match(self, ner_list):
        if len(ner_list) == 0:
            return []
        i, j = -1, 0
        long_match = []
        tmp = [ner_list[0]]
        while i < len(ner_list) and j < len(ner_list):
            if i < 0:
                i += 1
                j += 1
            else:
                if ner_list[i]['endPosition'] == ner_list[j]['startPosition']:
                    tmp.append(ner_list[j])
                    i += 1
                    j += 1
                else:
                    if len(tmp) > 1:
                        long_match.append(tmp)
                    tmp = [ner_list[j]]
                    i += 1
                    j += 1
        return long_match

    def double_blind(self):
        # org = json.load(
        #     open(
        #         '/Users/apple/XHSworkspace/data/structure/food/tagging/double_verify/20210926_双盲/0926_uniformity_.json',
        #         'r',
        #         encoding='utf-8'))
        org = parse_from_file_ttt(
            "/Users/apple/XHSworkspace/data/structure/food/tagging/double_verify/20210926_双盲/20210926_双盲_19.txt")
        res = []
        for i in org:
            # res.append(org[i][0]["results"][0]["result"].strip("\'"))
            res.append(i)
        # write_file(data=res, file="/Users/apple/XHSworkspace/data/structure/food/tagging/double_verify/20210926_双盲/20210926_双盲_79.txt")
        ttt = parse_from_file_ttt(
            "/Users/apple/XHSworkspace/data/structure/food/tagging/double_verify/20210926_双盲/20210926_双盲_79.txt")
        # with open("/Users/apple/XHSworkspace/data/structure/food/tagging/double_verify/20210926_双盲/20210926_双盲_79.json", 'w',
        #           encoding='utf-8') as fp:
        #     json.dump(ttt, fp, ensure_ascii=False, indent=4)

        res2 = []
        for i in ttt:
            if i in res:
                continue
            else:
                res2.append(ttt[i]["note_id"] + "/t /t " + ttt[i]["text"])
        write_file(data=res2,
                   file="/Users/apple/XHSworkspace/data/structure/food/tagging/double_verify/20210926_双盲/20210926_双盲_79_plain.txt")
        print(res)

    def check_rep_data(self):
        dir_path = "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_rel4_pkl"
        path_list = os.listdir(dir_path)
        pdList = []  # List of your dataframes
        pdList2 = []  # List of your dataframes
        for i in path_list:
            if i.startswith("food_") and i.endswith(".pkl"):
                tmp = pd.read_pickle(osp.join(dir_path, i))
                tmp["pkl_from"] = i
                if "rel4" in i:
                    pdList.append(tmp)
                else:
                    pdList2.append(tmp)
        new_df = pd.concat(pdList)
        new_df2 = pd.concat(pdList2)
        new_df_all = pd.concat([new_df, new_df2])
        print("笔记数量:{}, 去重笔记数量:{}".format(len(new_df), len(list(set(new_df["note_id"])))))
        print("笔记数量:{}, 去重笔记数量:{}".format(len(new_df2), len(list(set(new_df2["note_id"])))))
        print("笔记数量:{}, 去重笔记数量:{}".format(len(new_df_all), len(list(set(new_df_all["note_id"])))))
        print()

    def re_check_tagging_data(self):
        data_50_200 = read_file(
            "/Users/apple/XHSworkspace/data/structure/food/000000_0.csv_cutDoc_exp_test_between_50-200_20210929")

        dir_path = "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_rel4_pkl"
        path_list = os.listdir(dir_path)
        pdList = []  # List of your dataframes
        for i in path_list:
            if i.startswith("food_") and i.endswith(".pkl"):
                tmp = pd.read_pickle(osp.join(dir_path, i))
                tmp["pkl_from"] = i
                pdList.append(tmp)
        new_df = pd.concat(pdList)
        note_ids = list(set(new_df["note_id"].tolist()))

        data_dict = defaultdict(list)
        for i in note_ids:
            if i not in data_dict:
                data_dict[i] = []

        data_dict_not = defaultdict(list)
        for i in data_50_200:
            noteid = i.split("/t")[0]
            if noteid not in data_dict:
                data_dict_not[noteid] = i
            else:
                data_dict[noteid] = i

        test_data = json.load(open(
            '/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/test_v3.json',
            'r', encoding='utf-8'))
        for i in test_data:
            if i in data_dict_not:
                print(data_dict_not[i])
                data_dict_not.pop(i)

        write_file(
            file="/Users/apple/XHSworkspace/data/structure/food/000000_0.csv_cutDoc_exp_test_between_50-200_20210929_unique",
            data=list(data_dict_not.values()))
        print(data_dict_not)

    def collect_high_quality_data(self):
        dir_path = "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_david_csv"
        path_list = os.listdir(dir_path)
        data_col = defaultdict(list)
        for i in path_list:
            # if i.startswith("美食_") and i.endswith("_高一致率.json"):
            if i.startswith("dup") and i.endswith("_520.json"):
                tmp = json.load(open(osp.join(dir_path, i), 'r', encoding='utf-8'))
                for j in tmp:
                    for k in tmp[j][0]['results']:
                        ner_d = k['result'].strip("\'")
                        ner_d = ner_d.split("***")[0]
                        ner_d = ner_d.replace("/t /t ", "/t/t")
                        ner_d = ner_d.replace("/t/t/t ", "/t/t")
                        if j in data_col:
                            data_col[j].append(ner_d)
                        else:
                            data_col[j] = [ner_d]
        self.high_quality_id = data_col
        # with open(osp.join(dir_path, "high_quality_id.json"), 'w',
        #           encoding='utf-8') as fp:
        #     json.dump(data_col, fp, ensure_ascii=False, indent=4)
        # print(data_col)

    def ttt(self):
        ttt = read_file("/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/ttt.tmp")
        ttt = list(set(ttt))
        ttt.sort(key=lambda s: len(s))
        write_file(data=ttt, file="/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/ttt.tmp")

        org = json.load(open(
            '/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_david_csv/dup3038_520_json_txt_pkl_remove_note_id.json',
            'r', encoding='utf-8'))
        org2 = json.load(open(
            '/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_rel4_pkl/20211017_ner_nodup_21p_remove_note_id.json',
            'r', encoding='utf-8'))
        res = []
        for i in org:
            res.append(org[i]['relation'].strip("\'").split("***")[0])
        for j in org2:
            res.append(org2[j]['relation'].strip("\'").split("***")[0])
        write_file(
            file=osp.join("/Users/apple/XHSworkspace/data/structure/food/22th_20211017_note_id_remove.txt"),
            data=res)
        print(res)


if __name__ == '__main__':
    extractor = EntityRuleExtractor()
    ss = SegSentence()
    # dir_path = "/Users/apple/XHSworkspace/data/structure/food/dataset"
    # extractor.double_blind()

    """ 杂乱 """
    # extractor.ttt()
    """ 根据质检一致率挑选高质量数据 """
    extractor.collect_high_quality_data()
    """ step1：从 david 导出数据生成可用pkl """
    # extractor.type_14_to_10()
    """ step2：汇总挑选数据json """
    # extractor.process_pkl_file()
    """ step3: 划分训练集 """
    extractor.generate_train_data()
    # extractor.check_rep_data()
    # extractor.re_check_tagging_data()

    d = NerDataset("/Users/apple/XHSworkspace/data/structure/food/config/label_index.json")
    # res = extractor.extract_keywords(data=data)
    # print(len(res))
