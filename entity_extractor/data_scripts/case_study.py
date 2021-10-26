# -*- coding: utf-8 -*-
# @Time    : 2021/9/24 19:18
# @Author  : heyee (jialeyang)


import copy
import os.path as osp
import os
from collections import defaultdict
from collections import OrderedDict
from operator import getitem
import json
# from eval_performance import *
from tqdm import tqdm
import pandas as pd
from collections import Counter
from data_utils import *
from sklearn.model_selection import train_test_split

tqdm.pandas()


class CaseStudy:

    def __init__(self):
        pass

    def connected_entity(self):
        # org = json.load(
        #     open('/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/20210922_ner.json', 'r',
        #          encoding='utf-8'))
        # df = pd.DataFrame(org).T.reset_index(drop=True)

        df = pd.read_pickle(
            "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/20210922_ner_ttt.pkl")
        df["ner_truth"] = df["ner_d"].progress_apply(parse_tags)
        df["longest_match_truth"] = df["ner_truth"].progress_apply(self.find_longest_match)
        df["is_connected_truth"] = df["longest_match_truth"].apply(lambda x: len(x))

        df["ner_pred"] = df.progress_apply(
            lambda row: parse_tags(
                content=row["pred_ner"]["textPredict"]
            ), axis=1)
        # df["ner_pred"] = df["pred_ner"]["textPredict"].progress_apply(parse_tags)
        df["longest_match_pred"] = df["ner_pred"].progress_apply(self.find_longest_match)
        df["is_connected_pred"] = df["longest_match_pred"].apply(lambda x: len(x))

        """ 统计相连实体类型 & 频数 """

        # note_ids = df[df["is_connected"] == True].head(100)["note_id"].tolist()
        # print(self.pre_df[self.pre_df["note_id"] == "5fe92bd90000000001008252"]["pred_ner"].values[0]["textPredict"])

        def count_not_zero(alist):
            print("实体个数：{}".format(sum(alist)))
            res = 0
            for i in alist:
                if i != 0:
                    res += 1
            print("笔记个数：{}".format(res))

        count_not_zero(df["is_connected_truth"].tolist())
        count_not_zero(df["is_connected_pred"].tolist())

        def connected_change(a, b):
            return True if a == b else False

        df["connected_change"] = df.progress_apply(
            lambda row: connected_change(
                a=row["is_connected_truth"],
                b=row["is_connected_pred"]
            ), axis=1)

        connected_matrix = defaultdict(list)

        def connected_type(longest_match, note_id):
            for i in longest_match:
                type_s = []
                text_s = []
                for j in i:
                    type_s.append(j["type"])
                    text_s.append(j["text"])
                type_s = "-".join(type_s)
                text_s = "-".join(text_s)
                if type_s in connected_matrix:
                    connected_matrix[type_s]["count"] = connected_matrix[type_s]["count"] + 1
                    connected_matrix[type_s]["entity"].append({note_id: text_s})
                else:
                    connected_matrix[type_s] = {"count": 1, "entity": [{note_id: text_s}]}

        df.progress_apply(
            lambda row: connected_type(
                longest_match=row["longest_match_truth"],
                note_id=row["note_id"]
            ), axis=1)

        connected_matrix_truth = OrderedDict(sorted(connected_matrix.items(),
                                                    key=lambda x: getitem(x[1], 'count'),
                                                    reverse=True))
        # with open('/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/20210922_connected_matrix_truth.json', 'w',
        #           encoding='utf-8') as fp:
        #     json.dump(connected_matrix_truth, fp, ensure_ascii=False, indent=4)

        connected_matrix = defaultdict(list)
        df.progress_apply(
            lambda row: connected_type(
                longest_match=row["longest_match_pred"],
                note_id=row["note_id"]
            ), axis=1)

        connected_matrix_pred = OrderedDict(sorted(connected_matrix.items(),
                                                   key=lambda x: getitem(x[1], 'count'),
                                                   reverse=True))

        # with open('/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/20210922_connected_matrix_truth.json', 'w',
        #           encoding='utf-8') as fp:
        #     json.dump(connected_matrix_truth, fp, ensure_ascii=False, indent=4)

        # for key, val in connected_matrix_truth.items():
        #     print(key + "\t" + str(val["count"]) + "\t" + str(val["entity"][0]))
        # for key, val in connected_matrix_pred.items():
        #     print(key + "\t" + str(val["count"]) + "\t" + str(val["entity"][0]))

        def connected_diff(truth, pred):
            truth_tmp = []
            for i in truth:
                tmp_i = []
                for j in i:
                    tmp_i.append({
                        "type": j["type"],
                        "text": j["text"]
                    })
                truth_tmp.append(tmp_i)
            pred_tmp = []
            for i in pred:
                tmp_j = []
                for j in i:
                    tmp_j.append({
                        "type": j["type"],
                        "text": j["text"]
                    })
                pred_tmp.append(tmp_j)
            res = []
            for i in truth_tmp:
                if i not in pred_tmp:
                    res.append(i)
            return res

        df["connected_diff"] = df.progress_apply(
            lambda row: connected_diff(
                truth=row["longest_match_truth"],
                pred=row["longest_match_pred"]
            ), axis=1)

        connected_matrix = defaultdict(list)
        df.progress_apply(
            lambda row: connected_type(
                longest_match=row["connected_diff"],
                note_id=row["note_id"]
            ), axis=1)
        connected_matrix_diff = OrderedDict(sorted(connected_matrix.items(),
                                                   key=lambda x: getitem(x[1], 'count'),
                                                   reverse=True))
        # for key, val in connected_matrix_diff.items():
        #     print(key + "\t" + str(val["count"]) + "\t" + str(val["entity"][0]))

        # note_ids = df[df["is_connected"] == True].head(100)["note_id"].tolist()
        # print(self.pre_df[self.pre_df["note_id"] == "5fe92bd90000000001008252"]["pred_ner"].values[0]["textPredict"])

        df_same = df[df["connected_change"] == True]
        connected_matrix = defaultdict(list)
        df_same.progress_apply(
            lambda row: connected_type(
                longest_match=row["longest_match_pred"],
                note_id=row["note_id"]
            ), axis=1)

        connected_matrix_same = OrderedDict(sorted(connected_matrix.items(),
                                                   key=lambda x: getitem(x[1], 'count'),
                                                   reverse=True))

        print(df)

    def select_connected_data(self):
        df = pd.read_pickle(
            "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/20210922_ner_ttt.pkl")
        df_diff = df[df["connected_change"] == False]
        df_diff_noteids = df_diff["note_id"].tolist()
        org = json.load(
            open(
                '/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/20210922_connected_matrix_same.json',
                'r',
                encoding='utf-8'))
        df_same_noteids = []
        for i in org:
            if "食品-食品" in i or "食材-食材" in i:
                entities = org[i]["entity"]
                for j in entities:
                    for key, val in j.items():
                        df_same_noteids.append(key)
        recutdoc = []
        recutdoc.extend(df_diff_noteids)
        recutdoc.extend(df_same_noteids)
        recutdoc = list(set(recutdoc))
        print(len(recutdoc))
        write_file(file='/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/20210922_recutdoc.note_id',
                   data=recutdoc)
        df_same = df[df["connected_change"] == True]
        df_same_noteids = df_same["note_id"].tolist()
        not_recutdoc = [w for w in df_same_noteids if w not in recutdoc]
        write_file(
            file='/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/20210922_not_recutdoc.note_id',
            data=not_recutdoc)
        print(len(not_recutdoc))
        data_all = read_file("/Users/apple/XHSworkspace/data/structure/food/000000_0.csv4cutDoc")
        res = []
        for i in tqdm(data_all):
            note_id = i.split("\t\t")[0]
            if note_id in recutdoc:
                res.append(i)
        write_file(
            file='/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/20210922_recutdoc.res',
            data=res)
        print(org)

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


if __name__ == '__main__':
    cases = CaseStudy()
    # cases.connected_entity()
    cases.select_connected_data()
