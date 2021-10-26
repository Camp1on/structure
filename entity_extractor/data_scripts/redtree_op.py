# -*- coding: utf-8 -*-
# @Time    : 2021/9/13 18:37
# @Author  : heyee (jialeyang)


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


class RepaireData:

    def __init__(self):
        self.labels_map = json.load(
            open("/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/label_index_food.json",
                 'r', encoding='utf-8'))
        self.labels, self.reverse_index = NerDataset.get_labels(
            "/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/label_index_food.json")
        self.labels_reverse = {value: key for (key, value) in self.labels.items()}
        self.labels_map_reverse = {value: key for (key, value) in self.labels_map.items()}
        self.org = json.load(
            open('/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/red_tree_food_concept_v28.json', 'r', encoding='utf-8'))
        self.tmp_org = defaultdict(list)

    def fetch_data(self, file):
        df = pd.read_csv(file)
        self.write_file(file=file + "_food_add", data=df["value"].tolist())

        print(df)

    def merge_redtree(self):
        org = json.load(
            open('/Users/apple/XHSworkspace/data/structure/food/config/red_tree_food_v6.json', 'r', encoding='utf-8'))
        food_add = self.read_file(
            "/Users/apple/XHSworkspace/data/structure/food/知识库更新/64phwr4i41f986qt3Du0.csv_food_add")[:-1]
        food_old = org["美食"]["食品"]
        print(len(food_old))
        org["美食"]["食品"] = []
        food_merge = list(set(food_add + food_old))
        org["美食"]["食品"] = food_merge
        print(len(food_merge))
        with open('/Users/apple/XHSworkspace/data/structure/food/config/red_tree_food_v8.json', 'w',
                  encoding='utf-8') as fp:
            json.dump(org, fp, ensure_ascii=False, indent=4)
        print(org)

    def load_red_tree(self):
        dir_path = "/Users/apple/XHSworkspace/data/structure/food/config"
        path_list = os.listdir(dir_path)
        for i in path_list:
            if i.startswith("red_tree_food_v8.json_") and not i.endswith("collect"):
                key_name = i.split("_")[-1]
                self.tmp_org[key_name] = list(set(self.read_file(osp.join(dir_path, i))))
        # self.tmp_org["品牌"] = []
        #
        # df_ttt = pd.read_pickle(
        #     "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/20210922_ner_ttt.pkl")

        # brand = []
        #
        # def get_brand(alist, type):
        #     for i in alist:
        #         if i["type"] == type:
        #             brand.append(i["text"])
        #
        # df_ttt.progress_apply(
        #     lambda row: get_brand(
        #         alist=row["parsed_res"],
        #         type="品牌"
        #     ), axis=1)
        #
        # self.tmp_org["品牌"] = list(set(brand))
        #
        # brand = []
        # df_ttt.progress_apply(
        #     lambda row: get_brand(
        #         alist=row["parsed_res"],
        #         type="食材"
        #     ), axis=1)
        # shicai = self.tmp_org["食材"]
        # shicai.extend(brand)
        # shicai = list(set(shicai))
        #
        # print(brand)
        # brand = []

        # self.write_file(data=self.tmp_org["品牌"], file="/Users/apple/XHSworkspace/data/structure/food/config/red_tree_food_v8.json_品牌")
        with open('/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/red_tree_food_concept_v28.json', 'w',
                  encoding='utf-8') as fp:
            json.dump(self.tmp_org, fp, ensure_ascii=False, indent=4)

    def pre_calculate_lattice_key(self):
        res = defaultdict(list)
        self.org['否定修饰'] = []
        for key, val in self.org.items():
            for pkey in tqdm(val):
                for pidx, pval in enumerate(pkey):
                    if pidx == 0:
                        tmp = res.get(pval, [0] * (int(len(self.labels_map)) - 3) * 2)
                        tmp[int(self.labels_map[key])] = 1
                        res[pval] = tmp
                    else:
                        tmp = res.get(pval, [0] * (int(len(self.labels_map)) - 3) * 2)
                        tmp[int(self.labels_map[key]) + int(len(self.labels_map) - 3)] = 1
                        res[pval] = tmp
        with open('/Users/apple/XHSworkspace/data/structure/food/models/word_lattice_key_food_20211018_clean.json', 'w',
                  encoding='utf-8') as fp:
            json.dump(res, fp, ensure_ascii=False, indent=4)
        return res

    def pre_calculate_lattice(self):
        res = defaultdict(list)
        # self.org["食品"] = []
        for key, val in self.org.items():
            for pkey in tqdm(val):
                for pidx, pval in enumerate(pkey):
                    if pidx == 0:
                        tmp = res.get(pval, [0] * 2)
                        tmp[0] = 1
                        res[pval] = tmp
                    else:
                        tmp = res.get(pval, [0] * 2)
                        tmp[1] = 1
                        res[pval] = tmp
        with open('/Users/apple/XHSworkspace/data/structure/food/models/word_lattice_food_20211003.json', 'w',
                  encoding='utf-8') as fp:
            json.dump(res, fp, ensure_ascii=False, indent=4)
        return res

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


if __name__ == '__main__':
    extractor = RepaireData()
    # dir_path = "/Users/apple/XHSworkspace/data/structure/food/dataset"
    dir_path = "/Users/apple/XHSworkspace/data/structure/food/知识库更新"
    # extractor.fetch_data(osp.join(dir_path, "64phwr4i41f986qt3Du0.csv"))

    # extractor.merge_redtree()
    # extractor.load_red_tree()
    # extractor.pre_calculate_lattice()
    extractor.pre_calculate_lattice_key()

    # osp.join(dir_path, "000000_0.csv_cutDoc_exp_test_20210831_base_model_plain_val2_jieba"))
