# -*- coding: utf-8 -*-
# @Time    : 2021/9/7 15:50
# @Author  : heyee (jialeyang)

import os.path as osp
import os
from collections import defaultdict
import json
from eval_performance import *
from tqdm import tqdm
import pandas as pd
from collections import Counter

tqdm.pandas()


class EntityRuleExtractor:

    def __init__(self):
        self.org = defaultdict(list)
        self.LONGEST_MATCH = []
        self.NOT_LONGEST_MATCH = []

    def load_red_tree(self):
        dir_path = "/Users/apple/XHSworkspace/data/structure/food/config"
        path_list = os.listdir(dir_path)
        for i in path_list:
            if i.startswith("red_tree_food_v8.json_") and not i.endswith("collect"):
                key_name = i.split("_")[-1]
                self.org[key_name] = list(set(self.read_file(osp.join(dir_path, i))))
        with open('/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/red_tree_food_concept_v3.json', 'w',
                  encoding='utf-8') as fp:
            json.dump(extractor.org, fp, ensure_ascii=False, indent=4)

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
                （1）食品：不包含地域
                （2）口味不含"的"：甜甜的
                （3）食材：最小序列，咸味蛋黄 => 咸味+蛋黄
                （4）食品/食材均不含品牌
        """

    def parse_from_file(file):
        res = []
        with open(file, 'r', encoding='utf-8') as reader:
            tmp = reader.read()
            lines = tmp.split("\n")
        for k, line in enumerate(lines):
            if len(line) > 0:
                tmp = re.compile(r'\\+').findall(line)
                tmp.sort(key=lambda s: len(s), reverse=True)
                for i in tmp:
                    line = line.replace(i, "/")
                splits = line.split("***")
                note_id = line.split("/t")[0]
                ner_d = splits[0][len(note_id):][4:]
                if len(splits) > 1:
                    relation_d = list(json.loads(splits[-1]))
                else:
                    relation_d = []
                relation_d_re = []
                for i in relation_d:
                    if i.get("p") in ["描述", "相同", "对比"]:
                        # relation_d_re.append({
                        #     "s": {"type": i.get("s")["type"],
                        #           "text": i.get("s")["text"]},
                        #     "p": i.get("p"),
                        #     "o": {"type": i.get("o")["type"],
                        #           "text": i.get("o")["text"]}
                        # })
                        """ 去除实体type """
                        relation_d_re.append({
                            "s": {"text": i.get("s")["text"]},
                            # 'startPosition': i.get("s")["startPosition"],
                            # 'endPosition': i.get("s")["endPosition"]},
                            "p": i.get("p"),
                            "o": {"text": i.get("o")["text"]},
                            # 'startPosition': i.get("o")["startPosition"],
                            # 'endPosition': i.get("o")["endPosition"]},
                        })
                res.append({
                    "note_id": line.split("/t")[0],
                    "data": line,
                    "plain_text": remove_all_tag(ner_d),
                    "ner_d": ner_d,
                    "parsed_res": parse_tags(ner_d),
                    # "parsed_res_relation": relation_d
                    "parsed_res_relation": relation_d_re
                    # "parsed_res_relation_index": process_one(parse_tags(ner_d), relation_d_re)
                })
                # relation_d = []

        # json.dump(res, open('/Users/apple/XHSworkspace/data/structure/food/config/test.json', 'w', encoding='utf-8'),
        #           ensure_ascii=False, indent=4)
        return res

    def combine_and_sort(self, list1, list2):
        list_combine = []
        list_combine.extend(list1)
        list_combine.extend(list2)
        newlist = sorted(list_combine, key=lambda k: k['startPosition'])
        return newlist

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

    def read_file(self, file):
        """ file format:   note_id \t\t property \t\t note
        """
        with open(file, 'r', encoding='utf-8') as reader:
            tmp = reader.read()
            lines = tmp.split("\n")
        return [w for w in lines if len(w) > 0]

    def write_file(self, file, data):
        with open(file, 'w', encoding='utf-8') as writer:
            for parsed_text in data:
                if isinstance(parsed_text, str) and len(parsed_text) > 0:
                    writer.write(f'{parsed_text}\n')
                else:
                    print(parsed_text)


def parse_from_file(file):
    res = []
    with open(file, 'r', encoding='utf-8') as reader:
        tmp = reader.read()
        lines = tmp.split("\n")
    for k, line in enumerate(lines):
        if len(line) > 0:
            tmp = re.compile(r'\\+').findall(line)
            tmp.sort(key=lambda s: len(s), reverse=True)
            for i in tmp:
                line = line.replace(i, "/")
            splits = line.split("***")
            note_id = line.split("/t")[0]
            ner_d = splits[0][len(note_id):][4:]
            if len(splits) > 1:
                relation_d = list(json.loads(splits[-1]))
            else:
                relation_d = []
            relation_d_re = []
            for i in relation_d:
                if i.get("p") in ["描述", "相同", "对比"]:
                    # relation_d_re.append({
                    #     "s": {"type": i.get("s")["type"],
                    #           "text": i.get("s")["text"]},
                    #     "p": i.get("p"),
                    #     "o": {"type": i.get("o")["type"],
                    #           "text": i.get("o")["text"]}
                    # })
                    """ 去除实体type """
                    relation_d_re.append({
                        "s": {"text": i.get("s")["text"]},
                        # 'startPosition': i.get("s")["startPosition"],
                        # 'endPosition': i.get("s")["endPosition"]},
                        "p": i.get("p"),
                        "o": {"text": i.get("o")["text"]},
                        # 'startPosition': i.get("o")["startPosition"],
                        # 'endPosition': i.get("o")["endPosition"]},
                    })
            res.append({
                "note_id": line.split("/t")[0],
                "data": line,
                "plain_text": remove_all_tag(ner_d),
                "ner_d": ner_d,
                "parsed_res": parse_tags(ner_d),
                # "parsed_res_relation": relation_d
                "parsed_res_relation": relation_d_re
                # "parsed_res_relation_index": process_one(parse_tags(ner_d), relation_d_re)
            })
            # relation_d = []

    # json.dump(res, open('/Users/apple/XHSworkspace/data/structure/food/config/test.json', 'w', encoding='utf-8'),
    #           ensure_ascii=False, indent=4)
    return res


if __name__ == '__main__':
    extractor = EntityRuleExtractor()
    # dir_path = "/Users/apple/XHSworkspace/data/structure/food/dataset"
    dir_path = "/Users/apple/XHSworkspace/data/structure/food/tagging"
    data = extractor.read_file(
        osp.join(dir_path, "food_20210901_passentity_match_jieba"))
    # osp.join(dir_path, "000000_0.csv_cutDoc_exp_test_20210831_base_model_plain_val2_jieba"))
    extractor.load_red_tree()
    d = NerDataset("/Users/apple/XHSworkspace/data/structure/food/config/label_index.json")
    res = extractor.extract_keywords(data=data)
    print(len(res))
