# -*- coding: utf-8 -*-
# @Time    : 2021/8/8 21:53
# @Author  : heyee (jialeyang)


from structure.entity_extractor.predict import Predictor
from structure.entity_extractor.data import NerDataset
import pandas as pd

from utils.logger_helper import logger
from tqdm import tqdm
from structure.knowledge import Knowledge
from structure.entity_extractor.rule_extractor import EntityRuleExtractor
import re
import os.path as osp
from seqeval.scheme import IOB2
from seqeval.metrics import f1_score, classification_report
from collections import defaultdict
from ner.universal_v2.ner_handler import UniverNerV2Serving
from ner.poi_ner.poi_ner_handler import PoiNerV2Serving

import json
import time
from structure.structure_handler import NoteStructureServing
from structure.entity_extractor.data_scripts.data_utils import *

tqdm.pandas()
BIO_PATTERN_ALL = re.compile(r'(?=\[(.+?)\](_(.+?)_))')  # 带有捕获组的正向超前断言 且 最短匹配
d = NerDataset("/Users/apple/XHSworkspace/data/structure/food/config/label_index_food.json")


def evaluate_performance_precison_recall(truth_file):
    with open(osp.join("/Users/apple/XHSworkspace/data/ner/data3/data3_hanlp_v3/cutDocV2dataset/data_com_pos",
                       "data4_com_v2.brand_yilou"), 'r', encoding='utf-8') as reader:
        tmp = reader.read()
        brand_yilou = tmp.split("\n")

    plain_list = []
    truth_text = []
    pred_text = []
    y_true_BIO = []
    y_pred_BIO = []
    with open(truth_file, 'r', encoding='utf-8') as reader:
        tmp = reader.read()
        lines = tmp.split("\n")
        for line in tqdm(lines, total=len(lines)):
            if len(line) == 0:
                continue
            # truth_text.append(line)
            # plain_list.append(remove_all_tag(line))
            # """
            res = BIO_PATTERN_ALL.findall(line)
            mark = False
            for i in res:
                if i[0] in brand_yilou:
                    mark = True
            if not mark:
                if line not in truth_text:
                    truth_text.append(line)
                    plain_list.append(remove_all_tag(line))
            # """

    for sent in tqdm(plain_list):
        pred_text.append(p.predict(sent))

    pred_text = [w.get("textPredict") for w in pred_text]

    for pred, true in zip(pred_text, truth_text):
        if len([w[1] for w in d.pre_process(pred, d.labels_map)]) == len(
                [w[1] for w in d.pre_process(true, d.labels_map)]):
            y_pred_BIO.append([w[1] for w in d.pre_process(pred, d.labels_map)])
            y_true_BIO.append([w[1] for w in d.pre_process(true, d.labels_map)])
    y_pred_BIO_reverse = reverse_bio_tag(y_pred_BIO)
    y_true_BIO_reverse = reverse_bio_tag(y_true_BIO)

    logger.info('evaluate res is: {}'.format(
        classification_report(y_true_BIO_reverse, y_pred_BIO_reverse, mode='strict', scheme=IOB2)))


def evaluate_bert_entity_precison_recall(truth_file):
    # with open(truth_file, 'r', encoding='utf-8') as reader:
    #     tmp = reader.read()
    #     lines = tmp.split("\n")
    # plain_text = []
    # pred_tags = []
    # true_tags = []
    # for line in tqdm(lines):
    #     if len(line) > 0:
    #         splits = line.split("\t")
    #         plain_text.append(remove_all_tag(splits[0]))
    #         # pred_tags.append(p.predict(remove_all_tag(splits[0])))
    #         true_tags.append(splits[1])

    with open("/Users/apple/XHSworkspace/data/structure/tagging_data/structure.ut_200_t", 'r',
              encoding='utf-8') as reader:
        tmp = reader.read()
        lines = tmp.split("\n")
    plain_text = []
    pred_tags = []
    true_tags = []
    for line in tqdm(lines):
        if len(line) > 0:
            splits = line.split("\t\t")
            tmp_plain = splits[0]
            plain_text.append(tmp_plain)
            text_beindex = [(m.start(0), m.end(0)) for m in re.finditer('\t', tmp_plain)]
            tmp_true = splits[1]
            i, j = 0, 0
            tmp_true_res = ""
            pred_end = 0
            for be in text_beindex:
                while i < len(tmp_plain) and j < len(tmp_true):
                    if i == be[0]:
                        tmp_true_res += tmp_true[pred_end:j] + "\t"
                        pred_end = j
                        break
                    end_mark = False
                    if tmp_true[j] == "[":
                        count = 2
                        while count > 0:
                            if tmp_true[j] == "]":
                                end_mark = True
                                i -= 1
                            if not end_mark:
                                i += 1
                            if tmp_true[j] == "_":
                                count -= 1
                                j += 1
                            else:
                                j += 1
                    else:
                        i += 1
                        j += 1
            tmp_true_res += tmp_true[pred_end:]
            true_tags.append(tmp_true_res)
    true_tags_res = []
    for line in true_tags:
        true_tags_res.extend(line.split("\t"))
    plain_text_res = [remove_all_tag(w) for w in true_tags_res]
    pred_score = []
    for text in tqdm(plain_text_res):
        # pred_tags.extend(ner.predict({'text_list': [text]}))
        # pred_tags.append(poi.predict({'text_list': [text]})[0].get("text_predict"))
        p_predict = p.predict(text)
        pred_tags.append(p_predict.get("textPredict"))

    # pred_tags_re = []
    # map_dict = {"COM": "品牌", "TIM": "时间", "LOC": "地点"}
    # for i in pred_tags:
    #     res = BIO_PATTERN_ALL.findall(i)
    #     for j in res:
    #         if j[2] in ["COM", "TIM", "LOC"]:
    #             i = re.sub(r'\[{}\](_{}_)'.format(j[0], j[1][1:-1]), r'[{}]_{}_'.format(j[0], map_dict[j[1][1:-1]]), i)
    #         else:
    #             i = re.sub(r'\[{}\](_{}_)'.format(j[0], j[1][1:-1]), r'[{}]_{}_'.format(j[0], "NUL"), i)
    #     pred_tags_re.append(i)
    # df = pd.DataFrame({"rule": pred_tags_re, "truth": true_tags_res})
    df = pd.DataFrame({"rule": pred_tags, "truth": true_tags_res})

    df["rule_entity"] = df["rule"].apply(parse_tags)
    df["truth_entity"] = df["truth"].apply(parse_tags)

    confusion_matrix = defaultdict(lambda: defaultdict(list))
    generate_matrix = generate_confusion_matrix(pred_list=df["rule_entity"],
                                                truth_list=df["truth_entity"],
                                                confusion_matrix=confusion_matrix)
    res_matrix = calculate_confusion_matrix(generate_matrix)
    print_matrix(res_matrix)


def evaluate_rule_entity_precison_recall(truth_file):
    with open(truth_file, 'r', encoding='utf-8') as reader:
        tmp = reader.read()
        lines = tmp.split("\n")
    rule_tags = []
    pred_tags = []
    for line in lines:
        if len(line) > 0:
            splits = line.split("\t")
            rule_tags.append(splits[0])
            pred_tags.append(splits[1])
    df = pd.DataFrame({"rule": rule_tags, "truth": pred_tags})
    df["rule_entity"] = df["rule"].apply(parse_tags)
    df["truth_entity"] = df["truth"].apply(parse_tags)
    confusion_matrix = defaultdict(lambda: defaultdict(list))
    generate_matrix = generate_confusion_matrix(pred_list=df["rule_entity"],
                                                truth_list=df["truth_entity"],
                                                confusion_matrix=confusion_matrix)
    res_matrix = calculate_confusion_matrix(generate_matrix)
    print_matrix(res_matrix)


def evaluate_ner_entity_precison_recall_by_dict(dict_true):
    res = []
    for i in tqdm(dict_true):
        tmp = defaultdict(list)
        for key, value in i.items():
            tmp[key] = value

        plain_text = tmp.get("plain_text").replace("/t", "  ")
        tmp["pred_plain"] = plain_text
        # pred = structure_ttt.predict_debug(domain="美食", text=tmp["pred_plain"])
        pred = p.predict(tmp["pred_plain"])
        pred_re = {
            # "text": pred["text"],
            # "textPredict": pred["textPredict"],
            "entities": [{
                "type": w["type"],
                "text": w['text'],
                'startPosition': w['startPosition'],
                'endPosition': w['endPosition']
            } for w in pred["entities"]]
        }
        tmp["pred_ner"] = pred_re
        res.append(tmp)
    true_tags = []
    pred_tags = []
    # rule_tags = []
    for i in res:
        true_tags.append(i.get("parsed_res"))
        pred_tags.append(i.get("pred_ner")["entities"])
        # pred_tags.append(i.get("pred_ner"))
    df = pd.DataFrame({"pred_entity": pred_tags,
                       "truth_entity": true_tags,
                       # "pred_rule": rule_tags,
                       })

    """ 融合品牌 """

    # def merge_brand(col1, col2, col3):
    #     for j in col1:
    #         if (j["type"] == "品牌") and (j not in col3):
    #             col1.remove(j)
    #             print(j)
    #     for i in col2:
    #         if (i["type"] == "品牌") and (i not in col1) and (i in col3):
    #             col1.append(i)
    #             print(i)
    #     return col1
    #
    # df['pred_entity_merge_brand'] = df.progress_apply(
    #     lambda row: merge_brand(
    #         col1=row["pred_entity"],
    #         col2=row["pred_rule"],
    #         col3=row["truth_entity"]
    #     ), axis=1)

    """ 实体抽取 """
    confusion_matrix = defaultdict(lambda: defaultdict(list))
    generate_matrix = generate_confusion_matrix(pred_list=df["pred_entity"],
                                                truth_list=df["truth_entity"],
                                                confusion_matrix=confusion_matrix)
    res_matrix = calculate_confusion_matrix(generate_matrix)
    print_matrix(res_matrix)


    confusion_matrix = defaultdict(lambda: defaultdict(list))
    generate_matrix = generate_confusion_matrix(pred_list=df["pred_rule"],
                                                truth_list=df["truth_entity"],
                                                confusion_matrix=confusion_matrix)
    res_matrix = calculate_confusion_matrix(generate_matrix)
    print_matrix(res_matrix)
    return res


def process_one_ttt(one):
    all_ner = set()
    for ner in one:
        # all_ner.add(ner['text'] + ',' + str(ner['startPosition']) + ',' + str(ner['endPosition']))
        # all_ner.add(ner['text'] + ',' + ner['type'] + ',' + str(ner['startPosition']) + ',' + str(ner['endPosition']))
        all_ner.add(ner['text'] + ',' + ner['type'])
    return all_ner


def call_predict(dict_true):
    res = []
    for i in tqdm(dict_true):
        tmp = defaultdict(list)
        for key, value in i.items():
            tmp[key] = value

        tmp["pred_plain"] = tmp.get("plain_text")
        tmp["pred_ner"] = p.predict(tmp["pred_plain"])
        tmp["pred_ner"]['textPredict'] = tmp["pred_ner"]['textPredict'].replace("\t", " ##SEP## ")
        res.append(tmp["note_id"] + "/t /t " + tmp["pred_ner"]['textPredict'])
    with open(
            "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/20210922_recutdoc.res_cutdoc_pred",
            'w', encoding='utf-8') as writer:
        for parsed_text in res:
            if isinstance(parsed_text, str) and len(parsed_text) > 0:
                writer.write(f'{parsed_text}\n')
            else:
                print(parsed_text)
    return res


def predict_new_data():
    name = "22th_2000_1018_西安"
    data_50_200 = read_file(
        "/Users/apple/XHSworkspace/data/structure/food/{}".format(name))
    res = []
    for i in tqdm(data_50_200):
        res.append(p.predict(remove_all_tag(i)))
    write_file(data=[w["textPredict"] for w in res],
               file="/Users/apple/XHSworkspace/data/structure/food/{}_pred".format(name))


def evaluate_ner_entity_precison_recall_by_dict_ttt(dict_true):
    res = []
    for i in tqdm(dict_true):
        tmp = defaultdict(list)
        for key, value in i.items():
            tmp[key] = value

        plain_text = tmp.get("plain_text").replace("/t", " ")
        tmp["pred_plain"] = plain_text
        tmp["pred_ner"] = p.predict(tmp["pred_plain"])
        first_ner = process_one_ttt(tmp["parsed_res"])  # truth
        second_ner = process_one_ttt(tmp["pred_ner"]["entities"])  # pred
        tmp["first_ner"] = first_ner
        tmp["second_ner"] = second_ner
        cur_ner_inter = len(first_ner.intersection(second_ner))
        cur_ner_union = len(first_ner.union(second_ner))
        tmp["cur_ner_inter"] = cur_ner_inter
        tmp["cur_ner_union"] = cur_ner_union
        res.append(tmp)
    #
    # with open('/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/20210922_ner_ttt.json', 'w',
    #           encoding='utf-8') as fp:
    #     json.dump(res, fp, ensure_ascii=False, indent=4)

    # df = pd.read_pickle(
    #     "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/20210922_ner_ttt.pkl")

    df = pd.DataFrame(res)
    print(res)

    def cal_consistence(cur_ner_inter, cur_ner_union):
        return cur_ner_inter / cur_ner_union if cur_ner_union != 0 else 1

    df["consistence"] = df.progress_apply(
        lambda row: cal_consistence(
            cur_ner_inter=row["cur_ner_inter"],
            cur_ner_union=row["cur_ner_union"]
        ), axis=1)

    df_05 = df[df["consistence"] < 0.6]  # 2043/7138
    df_075 = df[(0.6 <= df["consistence"]) & (df["consistence"] < 0.8)]  # 1663/7138
    df_09 = df[(0.8 <= df["consistence"]) & (df["consistence"] < 0.9)]  # 748/7138
    df_1 = df[df["consistence"] >= 0.9]  # 2684/7138
    pdList = [df_09, df_1]  # List of your dataframes
    new_df = pd.concat(pdList)  # 3432/7138
    note_ids = new_df["note_id"].tolist()

    with open(
            "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/20210922_ner_ttt_note_ids_08_1.txt",
            'w', encoding='utf-8') as writer:
        for parsed_text in note_ids:
            if isinstance(parsed_text, str) and len(parsed_text) > 0:
                writer.write(f'{parsed_text}\n')
            else:
                print(parsed_text)


def evaluate_rule_entity_precison_recall_by_two_dict(dict_pred, dict_true):
    true_tags = []
    pred_tags = []
    true_tags_relation = []
    pred_tags_relation = []
    re_learning = []
    for i in dict_pred:
        for j in dict_true:
            if i.get("note_id") == j.get("note_id"):
                re_learning.append(i.get("data").split("***")[0] + "——" * 100 + j.get("data").split("***")[0] +
                                   i.get("data").split("***")[-1])
                pred_tags.append(i.get("parsed_res"))
                true_tags.append(j.get("parsed_res"))
                pred_tags_relation.append(i.get("parsed_res_relation"))
                true_tags_relation.append(j.get("parsed_res_relation"))

    # with open("/Users/apple/XHSworkspace/data/structure/food/tagging/food_35_1st.2831_relearning", 'w', encoding='utf-8') as writer:
    #     for parsed_text in re_learning:
    #         if isinstance(parsed_text, str) and len(parsed_text) > 0:
    #             writer.write(f'{parsed_text}\n')
    #         else:
    #             print(parsed_text)

    # df = pd.DataFrame({"rule": rule_tags, "truth": pred_tags})
    # df["rule_entity"] = df["rule"].apply(parse_tags)
    # df["truth_entity"] = df["truth"].apply(parse_tags)
    df = pd.DataFrame({"pred_entity": pred_tags,
                       "truth_entity": true_tags,
                       "pred_entity_relation": pred_tags_relation,
                       "truth_entity_relation": true_tags_relation})
    """ 实体抽取 """
    confusion_matrix = defaultdict(lambda: defaultdict(list))
    generate_matrix = generate_confusion_matrix(pred_list=df["pred_entity"],
                                                truth_list=df["truth_entity"],
                                                confusion_matrix=confusion_matrix)
    res_matrix = calculate_confusion_matrix(generate_matrix)
    print_matrix(res_matrix)

    """ 关系抽取 """
    confusion_matrix_relation = defaultdict(lambda: defaultdict(list))
    generate_matrix_relation = generate_confusion_matrix_relation(pred_list=df["pred_entity_relation"],
                                                                  truth_list=df["truth_entity_relation"],
                                                                  confusion_matrix=confusion_matrix_relation)
    res_matrix_relation = calculate_confusion_matrix(generate_matrix_relation)
    print_matrix_relatiom(res_matrix_relation)
    print("hi")


def remove_all_tag(content):
    res = BIO_PATTERN_ALL.findall(content)
    for i in res:
        content = content.replace("[{}]{}".format(i[0], i[1]), r'{}'.format(i[0]))
        # content = re.sub(r'\[{}\](_{}_)'.format(i[0], i[1][1:-1]), r'{}'.format(i[0]), content)
    return content


def reverse_bio_tag(BIO_list):
    y_pred_BIO_reverse = []
    for tag in BIO_list:
        tag_reverse = []
        for idx, val in enumerate(tag):
            if val == "O":
                tag_reverse.append(val)
            else:
                tmp = val.split("-")
                tag_reverse.append(tmp[1] + "-" + tmp[0])
        y_pred_BIO_reverse.append(tag_reverse)
    return y_pred_BIO_reverse


def generate_confusion_matrix(pred_list, truth_list, confusion_matrix):
    for pred, truth in tqdm(zip(pred_list, truth_list)):
        for i in pred:
            if i in truth:
                confusion_matrix[i.get("type")]["tp"].append(i)
            else:
                confusion_matrix[i.get("type")]["fp"].append(i)
        for j in truth:
            if j not in pred:
                confusion_matrix[j.get("type")]["fn"].append(j)
    return confusion_matrix


def generate_confusion_matrix_relation(pred_list, truth_list, confusion_matrix):
    for pred, truth in tqdm(zip(pred_list, truth_list)):
        for i in pred:
            if i in truth:
                confusion_matrix[i.get("p")]["tp"].append(i)
            else:
                confusion_matrix[i.get("p")]["fp"].append(i)
        for j in truth:
            if j not in pred:
                confusion_matrix[j.get("p")]["fn"].append(j)
    return confusion_matrix


def calculate_confusion_matrix(confusion_matrix):
    for key in confusion_matrix.keys():
        tp = len(confusion_matrix[key]["tp"])
        fp = len(confusion_matrix[key]["fp"])
        fn = len(confusion_matrix[key]["fn"])
        confusion_matrix[key]["precision"] = 0 if tp + fp == 0 else tp / (tp + fp)
        confusion_matrix[key]["recall"] = 0 if tp + fn == 0 else tp / (tp + fn)
        confusion_matrix[key]["f1-score"] = 0 if \
            confusion_matrix[key]["precision"] + confusion_matrix[key]["recall"] == 0 else \
            2 * confusion_matrix[key]["precision"] * confusion_matrix[key]["recall"] \
            / (confusion_matrix[key]["precision"] + confusion_matrix[key]["recall"])
        confusion_matrix[key]["support"] = tp + fn
    return confusion_matrix


def print_matrix(confusion_matrix):
    headers = ['precision', 'recall', 'f1-score', 'support']
    head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
    report = head_fmt.format('', *headers, width=10)
    print(report)
    avg_p = 0
    avg_r = 0
    support = 0
    for key in sorted(confusion_matrix, key=lambda x: int(d.labels_map[x])):
        avg_p += confusion_matrix[key]["precision"] * confusion_matrix[key]["support"]
        avg_r += confusion_matrix[key]["recall"] * confusion_matrix[key]["support"]
        support += confusion_matrix[key]["support"]
        print("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{}".
              format(d.labels_map[key] + ":" + key,
                     confusion_matrix[key]["precision"],
                     confusion_matrix[key]["recall"],
                     confusion_matrix[key]["f1-score"],
                     confusion_matrix[key]["support"]))
    print("precision: {:.3f}".format(avg_p / support))
    print("recall: {:.3f}".format(avg_r / support))
    print("f1: {:.3f}".format(2 * (avg_p / support) * (avg_r / support)
                              / (avg_p / support + avg_r / support)))


def print_matrix_relatiom(confusion_matrix):
    headers = ['precision', 'recall', 'f1-score', 'support']
    head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
    report = head_fmt.format('', *headers, width=10)
    print(report)
    avg = 0
    support = 0
    for key in confusion_matrix:
        avg += confusion_matrix[key]["precision"] * confusion_matrix[key]["support"]
        support += confusion_matrix[key]["support"]
        print("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{}".
              format(key,
                     confusion_matrix[key]["precision"],
                     confusion_matrix[key]["recall"],
                     confusion_matrix[key]["f1-score"],
                     confusion_matrix[key]["support"]))
    print("precision: {:.2f}".format(avg / support))


def parse_tags(content):
    pred_label = [w[1] for w in d.pre_process(content, d.labels_map)]
    type = [''] * len(pred_label)
    tagging = ['O'] * len(pred_label)

    for i, p in enumerate(pred_label):
        if p == 'O':
            continue
        try:
            splits = p.split('-')
            type[i] = d.reverse_index[splits[0]]
            tagging[i] = splits[1]
        except:
            print(i)
    matchs = re.finditer('BI*', ''.join(tagging))
    tokens = remove_all_tag(content)

    one = []
    textPredict = ""
    pre_end = 0
    for match in matchs:
        s, e = match.span()
        sent_start = len(''.join(tokens[:s]))
        text = ''.join(tokens[s:e])
        textPredict += ''.join(tokens[pre_end:s]) + "[" + text + "]_" + type[s:e][0] + "_"
        pre_end = e
        one.append({
            'type': type[s],
            'text': text,
            'startPosition': sent_start,
            'endPosition': sent_start + len(text)
        })
    textPredict += ''.join(tokens[pre_end:])
    return one


def process_one(ner_rst, relation_rst):
    index = {}
    for i, cur_ner in enumerate(ner_rst):
        start = cur_ner['startPosition']
        index[start] = i
    relation = []
    for cur_relation in relation_rst:
        s_start = cur_relation['s']['startPosition']
        s_index = index[s_start]
        s_text = cur_relation['s']['text']
        o_start = cur_relation['o']['startPosition']
        o_index = index[o_start]
        o_text = cur_relation['o']['text']
        p_text = cur_relation['p']
        relation.append(s_text + ',' + str(s_index) + '-' + p_text + '-' + o_text + ',' + str(o_index))
    relation = '##'.join(relation)
    return relation


def generate_train_data(file):
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
            res.append(note_id + "/t/t" + remove_all_tag(ner_d).replace("/t", " "))
    # with open(file+"entity_match", 'w', encoding='utf-8') as writer:
    #     for parsed_text in res:
    #         if isinstance(parsed_text, str) and len(parsed_text) > 0:
    #             writer.write(f'{parsed_text}\n')
    #         else:
    #             print(parsed_text)
    return res


def parse_from_json(file):
    res = []
    org = json.load(open(file, 'r', encoding='utf-8'))
    for i in org:
        # res.append(org[i]["org_data"])
        ner_parse = org[i]["ner"]
        one = []
        for j in ner_parse:
            one.append({
                'type': j["type"],
                'text': j["text"]
                # 'startPosition': sent_start,
                # 'endPosition': sent_start + len(text)
            })
        res.append({
            "note_id": i,
            "user": org[i].get("user", ""),
            # "data": org[i]["org_data"],
            "plain_text": org[i]["text"],
            "ner_d": org[i].get("ner_d", ""),
            "parsed_res": one,
            "parsed_res_relation": []
        })
    # with open('/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/test_v1_ner.txt', 'w',
    #           encoding='utf-8') as writer:
    #     for parsed_text in res:
    #         if isinstance(parsed_text, str) and len(parsed_text) > 0:
    #             writer.write(f'{parsed_text}\n')
    #         else:
    #             print(parsed_text)
    return res


def parse_from_json_v2(file):
    res = []
    org = json.load(open(file, 'r', encoding='utf-8'))
    for i in org:
        # res.append(org[i]["org_data"])
        ner_parse = i["ner"]
        one = []
        for j in ner_parse:
            one.append({
                'type': j["type"],
                'text': j["text"]
                # 'startPosition': sent_start,
                # 'endPosition': sent_start + len(text)
            })
        res.append({
            "note_id": i,
            "user": org[i].get("user", ""),
            "data": org[i]["org_data"],
            "plain_text": org[i]["text"],
            "ner_d": org[i].get("ner_d", ""),
            "parsed_res": one,
            "parsed_res_relation": []
        })

    return res


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
            # line = line.replace("/t /t ", "/t/t")
            line = line.strip("\'")
            splits = line.split("***")
            note_id = line.split("/t")[0]
            ner_d = splits[0][len(note_id):][4:]
            # if len(splits) > 1:
            #     relation_d = list(json.loads(splits[-1]))
            # else:
            #     relation_d = []
            relation_d_re = []
            # for i in relation_d:
            #     if i.get("p") in ["描述", "相同", "对比"]:
            #         # relation_d_re.append({
            #         #     "s": {"type": i.get("s")["type"],
            #         #           "text": i.get("s")["text"]},
            #         #     "p": i.get("p"),
            #         #     "o": {"type": i.get("o")["type"],
            #         #           "text": i.get("o")["text"]}
            #         # })
            #         """ 去除实体type """
            #         relation_d_re.append({
            #             "s": {"text": i.get("s")["text"]},
            #             # 'startPosition': i.get("s")["startPosition"],
            #             # 'endPosition': i.get("s")["endPosition"]},
            #             "p": i.get("p"),
            #             "o": {"text": i.get("o")["text"]},
            #             # 'startPosition': i.get("o")["startPosition"],
            #             # 'endPosition': i.get("o")["endPosition"]},
            #         })
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


def parse_from_file_truth(file):
    res = []
    with open(file, 'r', encoding='utf-8') as reader:
        tmp = reader.read()
        lines = tmp.split("\n")
    for line in lines:
        if len(line) > 0:
            tmp = re.compile(r'\\+').findall(line)
            tmp.sort(key=lambda s: len(s), reverse=True)
            for i in tmp:
                line = line.replace(i, "/")
            splits = line.split("***")
            ner_d = splits[0]
            if len(splits) > 1:
                note_id = line.split("/t")[0]
                ner_d = splits[0][len(note_id):][4:]
                relation_d = list(json.loads(splits[-1]))
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
                        " 关系顺序无关 """
                        relation_d_re.append({
                            "o": {"text": i.get("s")["text"]},
                            # 'startPosition': i.get("o")["startPosition"],
                            # 'endPosition': i.get("o")["endPosition"]},
                            "p": i.get("p"),
                            "s": {"text": i.get("o")["text"]},
                            # 'startPosition': i.get("s")["startPosition"],
                            # 'endPosition': i.get("s")["endPosition"]},
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


def parse_from_file_v2(file):
    res = defaultdict(list)
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
            res[note_id].append({
                "org_data": line,
                "text": remove_all_tag(ner_d),
                "ner_d": ner_d,
                "ner": parse_tags(ner_d),
                "relation": relation_d
            })
    json.dump(res, open('/Users/apple/XHSworkspace/data/structure/food/config/food_20210901_pass.json', 'w',
                        encoding='utf-8'),
              ensure_ascii=False, indent=4)
    return res


def collect_parsed_res(res_dict):
    res = defaultdict(list)
    for i in res_dict:
        for j in i.get("parsed_res"):
            k = j.get("type")
            v = j.get("text")
            if res.get(k) is None:
                res[k].append(v)
            else:
                if v not in res.get(k):
                    res[k].append(v)
    # json.dump([res], open('/Users/apple/XHSworkspace/data/structure/food/config/collect_parsed_res.json', 'w', encoding='utf-8'),
    #           ensure_ascii=False, indent=4)
    return res


def generate_data_4_zhijian(pred_dict):
    res = defaultdict(list)
    for i in pred_dict:
        res["实体"].append(i.get("ner_d"))
        res["关系"].append(i.get("parsed_res_relation_index"))
    df = pd.DataFrame(res)
    df.to_csv("/Users/apple/XHSworkspace/data/structure/food/tagging/food_214_relation.2886_csv", index=False,
              header=True, encoding='utf_8_sig')
    print(df)


if __name__ == '__main__':
    # knowledge_file = "/Users/apple/XHSworkspace/data/structure/food/config/red_tree_food_v7.json"
    # knowledge = Knowledge(knowledge_file)
    # rule_extractor = EntityRuleExtractor(knowledge.get_all_property())

    label_index1 = '/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/label_index_food.json'
    # label_index = '/Users/apple/XHSworkspace/data/structure/bert_base_chinese_torch_userdefine/label_index_NER.json'
    # p = Predictor(bert_config, vocab_file, model_file, label_index)
    d = NerDataset(label_index1)

    # model_file = '/Users/apple/XHSworkspace/data/structure/food/models/20211003/20211004115740/model_steps_700_20211004115740.pt'
    model_file = '/Users/apple/XHSworkspace/data/structure/food/models/20211017/20211017183235/model_steps_1300.pt'
    # model_file = '/Users/apple/XHSworkspace/data/structure/food/models/20211017/20211018180346/model_steps_1800.pt'
    bert_config = '/Users/apple/XHSworkspace/data/structure/food/models/bert_base.json'
    vocab_file = '/Users/apple/XHSworkspace/data/structure/food/models/vocab.txt'
    label_index = '/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/label_index_food.json'
    max_len = 150
    p = Predictor(bert_config, vocab_file, model_file, label_index)

    # structure_ttt = NoteStructureServing()

    # predict_new_data()
    true_dict = parse_from_file(
        # "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/debug")  ## 测试集校正
        # "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/test_v3_pkl.json_txt")  ## 测试集校正
        "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/test_v2_pkl_20211013_3_json_txt的副本_pkl.txt")  ## 测试集校正
        # "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/test_v2_pkl.json_txt")  ## 测试集校正

    evaluate_ner_entity_precison_recall_by_dict(dict_true=true_dict)

    # pred_dict = generate_train_data("/Users/apple/XHSworkspace/data/structure/food/tagging/food_20210901_pass")
    # true_dict = parse_from_file("/Users/apple/XHSworkspace/data/structure/food/tagging/food_20210901_pass")
    evaluate_rule_entity_precison_recall_by_two_dict(dict_pred=pred_dict, dict_true=true_dict)

    """ 评估实体抽取性能 """
    dir_path = "/Users/apple/XHSworkspace/data/structure/tagging_data"
    file_true = osp.join(dir_path, "2711.2710")
    # eval_entity(truth_file=file_true)
    # analysis_entity(truth_file=file_true)
    # evaluate_entity_precison_recall(truth_file=file_true)
    # ner = UniverNerV2Serving()
    # poi = PoiNerV2Serving()
    # evaluate_bert_entity_precison_recall(truth_file=file_true)
    evaluate_rule_entity_precison_recall(truth_file=file_true)

    # """ 评估模型召回能力 """
    # dir_path1 = "/Users/apple/XHSworkspace/data/brand_classify/ner_com_2nd_tagging"
    # file_true1 = osp.join(dir_path1, "ner_com_2nd_tagging.2614")
    # dir_path = "/Users/apple/XHSworkspace/data/brand_classify/ner_tagging_wuhan_30000/ner_tagging_res/eval_com_res"
    # file_true = osp.join(dir_path, "ner_tagging_COM.true")
    # # evaluate_top_poi_notes_agg(csv_file=osp.join(dir_path, "top_poi_notes_agg.txt"))
    # # evaluate_top_poi_notes_agg(csv_file=osp.join(dir_path, "3y01H31y5478d993rbZ0.content_cutDocV2"))
    # # evaluate_top_poi_notes_agg(csv_file=osp.join(dir_path, "download1627618500612.csv"))
    # evaluate_performance_precison_recall(truth_file=file_true)
    # evaluate_performance_precison_recall(truth_file=file_true1)
