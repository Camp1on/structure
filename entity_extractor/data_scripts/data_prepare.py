# -*- coding: utf-8 -*-
# @Time    : 2021/8/3 14:56
# @Author  : heyee (jialeyang)
import sys

sys.path.append("../../..")

from structure.knowledge import Knowledge
import os.path as osp
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import random
import re
import json
import copy
from structure.entity_extractor.rule_extractor import EntityRuleExtractor
from utils.s3_utils import S3Util
from conf.config import settings
from sklearn.model_selection import train_test_split

tqdm.pandas()
BIO_PATTERN_ALL = re.compile(r'(?=\[(.+?)\](_(.+?)_))')  # 带有捕获组的正向超前断言 且 最短匹配


class DataProcess:
    def __init__(self):
        # knowledge_file = S3Util.Instance().get_latest_file(settings['note_structure']['knowledge_file'])
        # knowledge_file = "/Users/apple/XHSworkspace/data/structure/food/config/red_tree_food_v7.json"
        # self.knowledge = Knowledge(knowledge_file)
        # self.rule_extractor = EntityRuleExtractor(self.knowledge.get_all_property())
        # self.values = self.knowledge.get_all_value()
        self.MAX_LEAF_SENT = 15

    def prepare4cutdoc(self, file):
        # df = pd.read_csv(file, error_bad_lines=False)
        # df["content"] = df['reg_title'].apply(lambda x: "" if str(x) == "nan" else x). \
        #     str.cat([df["reg_content"].apply(lambda x: "" if str(x) == "nan" else x)], sep='。'). \
        #     apply(lambda x: x[1:] if len(x) > 0 and x[0] == " " else x)
        # self.write_file(file + "_txt", df["content"].tolist())

        df = pd.read_excel(file, engine='openpyxl', sheet_name='1')
        # df["norm"] = df["norm"].progress_apply(lambda x: x.replace("_x0001_", " "))
        domain_list = set(df[df["数据类型"] == "KEY"]["键"].tolist())
        key_list = set(df["值"].tolist())
        pv_value = defaultdict(list)
        for i in domain_list:
            pv_value[i].extend(df[df["键"] == i]["值"].tolist())

        with open(file + "_food_json", 'w', encoding='utf-8') as writer:
            for parsed_text in pv_value:
                if isinstance(parsed_text, str) and len(parsed_text) > 0:
                    writer.write(f'{parsed_text} : {"|".join(pv_value.get(parsed_text))}\n')
                else:
                    print(parsed_text)

        brand_beauty = df[(df["domain"] == "美妆") & (df["key"] == "品牌")]
        brand_fashion = df[(df["domain"] == "时尚") & (df["key"] == "品牌")]

        brand_beauty_list = []
        brand_beauty_list.extend(brand_beauty["value"].tolist())
        brand_beauty_list.extend(brand_beauty["norm"].tolist())
        brand_beauty_list = list(set(brand_beauty_list))  # 7322

        brand_fashion_list = []
        brand_fashion_list.extend(brand_fashion["value"].tolist())
        brand_fashion_list.extend(brand_fashion["norm"].tolist())
        brand_fashion_list = list(set(brand_fashion_list))  # 7803
        # self.write_file(file + "_txt", df["content"].tolist())

        time = df[df["key"] == "时间"]
        time_list = []
        time_list.extend(time["value"].tolist())
        time_list.extend(time["norm"].tolist())
        time_list = list(set([w for w in time_list if "日游" not in w]))

        pinlei_beauty = df[(df["domain"] == "美妆") & (df["key"] == "产品品类")]
        pinlei_fashion = df[(df["domain"] == "时尚") & (df["key"] == "产品品类")]
        pinlei_beauty_list = []
        pinlei_beauty_list.extend(pinlei_beauty["value"].tolist())
        pinlei_beauty_list.extend(pinlei_beauty["norm"].tolist())
        pinlei_beauty_list1 = list(set([w for w in pinlei_beauty_list if 1 < len(w) < 4]))
        pinlei_beauty_list2 = list(set([w for w in pinlei_beauty_list if len(w) >= 4]))

        pinlei_fashion_list = []
        pinlei_fashion_list.extend(pinlei_fashion["value"].tolist())
        pinlei_fashion_list.extend(pinlei_fashion["norm"].tolist())
        pinlei_fashion_list1 = list(set([w for w in pinlei_fashion_list if 1 < len(w) < 4]))
        pinlei_fashion_list2 = list(set([w for w in pinlei_fashion_list if len(w) >= 4]))

        print(df)

    def arrange_data_fre(self, file):
        lines = self.read_file(file)
        res = defaultdict(list)
        for line in lines:
            sents = line.split("\t\t")
            if len(sents) == 3:
                res["note_id"].append(sents[0])
                res["property"].append(sents[1])
                res["note"].append(sents[2].replace("\t", " "))  # 笔记
            else:
                print("wrong format: {}".format(sents))
        df = pd.DataFrame(res)

        leaf_value = defaultdict(list)

        def halper(note_id, property, content):
            for val in self.values:
                if val in content:
                    leaf_value[val].append((note_id, property, content))

        df.progress_apply(lambda row: halper(
            note_id=row["note_id"],
            property=row["property"],
            content=row["note"]
        ), axis=1)

        res_data = []
        for key, val in leaf_value.items():
            if len(val) <= self.MAX_LEAF_SENT:
                res_data.extend([w[0] + "\t\t" + w[1] + "\t\t" + w[2] for w in val])
            else:
                res_data.extend([w[0] + "\t\t" + w[1] + "\t\t" + w[2] for w in random.sample(val, self.MAX_LEAF_SENT)])
        print(len(res_data))
        print(len(set(res_data)))

        self.write_file(file=file + "_frq{}".format(self.MAX_LEAF_SENT),
                        data=list(set(res_data)))

    def tagging_property(self, file):
        lines = self.read_file(file)
        lines = [w.replace("[", "【") for w in lines]
        lines = [w.replace("]", "】") for w in lines]
        # train, test = train_test_split(lines, test_size=1000, random_state=42, shuffle=True)
        # self.write_file(file=file + "_exp_test",
        #                 data=train)
        lines = [w for w in lines if len(w.split("\t\t")) == 3 and 20 <= len(w.split("\t\t")[2]) <= 300]

        res = defaultdict(list)
        for line in lines:
            if len(line) > 0:
                # res["note"].append(line.split("\t\t")[1])

                # res["note"].extend(line.split("\t"))

                # res["note"].append(line)

                sents = line.split("\t\t")
                if len(sents) == 3:
                    res["note_id"].append(sents[0])
                    res["title"].append(sents[1])
                    res["note"].append(sents[2])
                else:
                    print("wrong format: {}".format(sents))
        df = pd.DataFrame(res).sample(100000)
        df["content"] = df["title"].str.cat([df["note"]], sep='\t')
        # df["content"] = df['reg_title'].apply(lambda x: "" if str(x) == "nan" else x). \
        #     str.cat([df["reg_content"].apply(lambda x: "" if str(x) == "nan" else x)], sep='。'). \
        #     apply(lambda x: x[1:] if len(x) > 0 and x[0] == " " else x)
        df["rule_predict"] = df["content"].progress_apply(self.rule_extractor.predict)

        def select_type(pred):
            res = []
            # select_list = ['品类', '场景', '品牌', '人群', '风格', '图案', '颜色', '功能', '地点', '时间', '摄影', '流行元素', '材质', '发型', '美容方法']
            select_list = ['主要工艺', '食疗食补', '食品', '食材', '水果', '限定小吃', '限定菜系', '菜品口味', '工具', '适宜人群', '品牌', '常见菜式']
            for i in pred:
                if i.get("type") in select_list:
                    res.append(i)
            return res

        df["select_type"] = df["rule_predict"].progress_apply(select_type)

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

        df["len_500"] = df["note"].progress_apply(lambda x: True if len(x) < 500 else False)
        df = df[df["len_500"] == True]

        self.write_file(file=file + "_duiqi",
                        data=df["note_id"].str.cat([df["re_tagging_sentence"]], sep='\t\t').tolist())

    def read_file(self, file):
        """ file format:   note_id \t\t property \t\t note
        """
        with open(file, 'r', encoding='utf-8') as reader:
            tmp = reader.read()
            lines = tmp.split("\n")
        return lines

    def write_file(self, file, data):
        with open(file, 'w', encoding='utf-8') as writer:
            for parsed_text in data:
                if isinstance(parsed_text, str) and len(parsed_text) > 0:
                    writer.write(f'{parsed_text}\n')
                else:
                    print(parsed_text)

    def check_consistence(self, file1, file2, file3):
        data1 = self.read_file(file1)[:-1]
        data2 = self.read_file(file2)[:-1]
        data3 = self.read_file(file3)[:-1]
        res = defaultdict(lambda: defaultdict(list))
        # for i in data1:
        #     res[remove_all_tag(i)[:98]]["ut"].append(i)
        for j in data1:
            splits = j.split("\t")
            res[remove_all_tag(splits[1])[:98]]["ut"].append(splits[1])
        for i in data2:
            res[remove_all_tag(i)[:98]]["2710"].append(i)
        for i in data3:
            res[remove_all_tag(i)[:98]]["2711"].append(i)

        res_pair = []
        for i in res.items():
            try:
                res_pair.append(i[1]["ut"][0] + "-" * 200 + i[1]["2710"][0] + "-" * 200 + i[1]["2711"][0])
            except:
                print(i)
        self.write_file("/Users/apple/XHSworkspace/data/structure/tagging_data/ut_2710_2710.review", res_pair)
        print(data1)


def remove_all_tag(content):
    res = BIO_PATTERN_ALL.findall(content)
    for i in res:
        content = content.replace("[{}]{}".format(i[0], i[1]), r'{}'.format(i[0]))
        # content = re.sub(r'\[{}\](_{}_)'.format(i[0], i[1][1:-1]), r'{}'.format(i[0]), content)
    return content


def parse_qianqian(file):
    with open(file, 'r', encoding='utf-8') as fin:
        knowledge = json.load(fin)
        print(knowledge)
        content = [w.get("data")["desc"].replace("\n", "。") for w in knowledge]
        content = [w.replace("\r", " ") for w in content]
        content = [w.replace("\t", " ") for w in content]
        data.write_file(file + "_content", content)


def prepare_data4food(file):
    """ sql
    """
    # df = pd.read_csv(file, names=["discovery_id", "reg_title", "reg_content", "id1", "time"], error_bad_lines=False)
    # df["title"] = df["reg_title"].progress_apply(lambda x: "" if str(x) == "nan" else x.replace("\n", "\t\t\t"))
    # df["content"] = df["reg_content"].progress_apply(lambda x: "" if str(x) == "nan" else x.replace("\n", "\t\t\t"))
    # # df["content"] = df['reg_title'].apply(lambda x: "" if str(x) == "nan" else x). \
    # #     str.cat([df["reg_content"].apply(lambda x: "" if str(x) == "nan" else x)], sep='。'). \
    # #     apply(lambda x: x[1:] if len(x) > 0 and x[0] == " " else x)
    # res_data = []
    # df["select_title"] = df["title"].progress_apply(lambda x: True if len(x) > 0 else False)
    # df["select_content"] = df["content"].progress_apply(lambda x: True if len(x) > 0 else False)
    # df = df[(df["select_title"] == True) & (df["select_content"] == True)]
    # for id, title, content in zip(df["discovery_id"], df["title"], df["content"]):
    #     res_data.append(id + "\t\t" + title + "\t\t" + content)
    # data.write_file(file + "4cutDoc", res_data)

    """ UtNlpBaseServiceImpl.cutDocV2
    """
    # lines = data.read_file(file)
    # res = defaultdict(list)
    # for line in lines:
    #     sents = line.split("\t\t")
    #     if len(sents) == 3:
    #         res["note_id"].append(sents[0])
    #         res["title"].append(sents[1])
    #         res["note"].append(sents[2])  # 笔记
    #     else:
    #         print("wrong format: {}".format(sents))
    # df = pd.DataFrame(res)
    # res_data = []
    # for id, title, content in zip(df["note_id"], df["title"], df["note"]):
    #     res_data.append({
    #         "data": {
    #             "discovery_id": id,
    #             "content": title + "\t\t" +content
    #         }
    #     })
    # out_file = file + "_4qianqian.json"
    # fout = open(out_file, 'w', encoding='utf-8')
    # json.dump(res_data, fout, ensure_ascii=False, indent=4)
    #
    # print(df)
    # org = json.load(
    #     open("/Users/apple/XHSworkspace/data/structure/food/VvE3m3QiO93p9m6g74rG.csv_cutDoc_4qianqian.json", 'r',
    #          encoding='utf-8'))
    # print(org)
    #
    # json.dump(org, open('redTree.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    """ 关系抽取去重
    """
    # lines = data.read_file("/Users/apple/XHSworkspace/data/structure/food/tagging/food_selection_top50.2786")
    #
    # res = []
    # id = []
    # for line in lines:
    #     if len(line) > 0:
    #         splits = line.split("***")
    #         res.append(splits[0] + "***" + splits[-1])
    #         id_splits = line.split("\\")
    #         id.append(id_splits[0])
    # data.write_file(data=res, file="/Users/apple/XHSworkspace/data/structure/food/tagging/food_selection_top50.2786_re")
    # print(res)

    """ 根据 id 精选数据 """
    # data_200 = data.read_file(
    #     "/Users/apple/XHSworkspace/data/structure/food/tagging/000000_0.csv_cutDoc_exp_test_duiqi")
    # data_id = data.read_file("/Users/apple/XHSworkspace/data/structure/food/tagging/food_selection_top50.2786_re_id")
    # res = []
    # for line in data_200:
    #     for id in data_id:
    #         if len(id) > 0 and len(line) > 0:
    #             if id in line:
    #                 res.append(line)
    #
    # data.write_file(data=res, file="/Users/apple/XHSworkspace/data/structure/food/tagging/food_selection_top50.2786_re_retagging")
    """ 排序 food_mt """
    # data_mt = data.read_file(
    #     "/Users/apple/XHSworkspace/data/structure/food/tagging/food_mt_菜名")
    # data_mt.sort(key=lambda s: len(s))
    # data.write_file(data=data_mt, file="/Users/apple/XHSworkspace/data/structure/food/tagging/food_mt_菜名")
    # print(data_mt)


if __name__ == "__main__":
    data = DataProcess()
    dir_path = "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210909"
    # prepare_data4food(osp.join(dir_path, "000000_0.csv"))

    # data.prepare4cutdoc(osp.join(dir_path, "xXzo71vk84z984501uyz.txt_cutDocV2_with_t_1st_15type_1_500"))
    # data.arrange_data_fre(osp.join(dir_path, "xXzo71vk84z984501uyz.txt_cutDocV2"))
    # data.tagging_property(osp.join(dir_path, "xXzo71vk84z984501uyz.txt_cutDocV2_with_t"))

    train_data = data.read_file(osp.join(dir_path, "val.txt"))
    res = []
    for line in train_data:
        tmp = re.compile(r'\\+').findall(line)
        tmp.sort(key=lambda s: len(s), reverse=True)
        for i in tmp:
            line = line.replace(i, "/")
        splits = line.split("***")
        note_id = line.split("/t")[0]
        ner_d = splits[0][len(note_id):][4:]
        res.extend(ner_d.split(" ##SEP## "))
    data.write_file(data=res, file=osp.join(dir_path, "val_split.txt"))
    removed = data.read_file(osp.join(dir_path, "000000_0.csv_cutDoc_exp_test_duiqi_2000_0905"))
    removed2 = data.read_file(osp.join(dir_path, "000000_0.csv_cutDoc_test_1000"))
    lines = data.read_file(osp.join(dir_path, "000000_0.csv_cutDoc_exp_test"))
    note_id = []
    for j in removed:
        note_id.append(j.split("/t")[0])
    for j in removed2:
        note_id.append(j.split("/t")[0])
    res = []
    for i in tqdm(lines):
        if len(i.split("/t /t ")) == 3:
            if 50 < len(i.split("/t /t ")[2]) < 200 and i.split("/t")[0] not in note_id:
                res.append(i)
        else:
            print(i)
    data.write_file(data=res, file=osp.join(dir_path, "000000_0.csv_cutDoc_exp_test_between_50-200"))
    print(lines)

    data.tagging_property(osp.join(dir_path, "000000_0.csv_cutDoc_exp_test"))
    # data.tagging_property(osp.join(dir_path, "structure.ut_200_t"))
    # dir_path = "/Users/apple/XHSworkspace/data/structure/tagging_data"
    # data.check_consistence(file1=osp.join(dir_path, "structure.ut_200"),
    #                        file2=osp.join(dir_path, "entity_extraction.2710"),
    #                        file3=osp.join(dir_path, "entity_extraction.2711"))

    """ parse 千千标注 from 布雷 """
    dir_path = "/Users/apple/XHSworkspace/data/structure/brand"
    parse_qianqian(osp.join(dir_path, "project-117-at-2021-08-19-13-56-32e30e58.json"))
