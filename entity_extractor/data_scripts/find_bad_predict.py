# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 10:32
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
from data_utils import *

tqdm.pandas()
BIO_PATTERN_ALL = re.compile(r'(?=\[(.+?)\](_(.+?)_))')  # 带有捕获组的正向超前断言 且 最短匹配
d = NerDataset("/Users/apple/XHSworkspace/data/structure/food/config/label_index_food.json")


def predict_new_data():
    data_50_200 = parse_from_file(
        "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/test_v3_pkl.json_txt")
        # "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/test_v2_pkl.json_txt")
    res = defaultdict(list)
    for i in tqdm(data_50_200):
        note_id = i["note_id"]
        pred = p.predict(i["plain_text"].replace("/t", "  "))
        pred_re = {
            "text": pred["text"],
            "textPredict": pred["textPredict"],
            "entities": [{"type": w["type"], "text": w['text'], 'startPosition': w['startPosition'],
                          'endPosition': w['endPosition']} for w in pred["entities"]]
        }
        truth = {
            "text": i["plain_text"],
            "textPredict": i["ner_d"],
            "entities": i["parsed_res"]
        }
        res[note_id] = {
            "pred": pred_re,
            "truth": truth
        }
    return res


def find_bad_data(res):
    show_dict = defaultdict(list)
    for i in res:
        precision, recall, confusion_matrix = calculate_pr_for_instance(res[i]["pred"]["entities"],
                                                                        res[i]["truth"]["entities"])
        if precision < 0.9:
            show_dict[i] = {
                "pred_text": res[i]["pred"]["textPredict"],
                "truth_text": res[i]["truth"]["textPredict"],
                "precision": precision,
                "recall": recall,
                "fp": confusion_matrix["fp"],
                "fn": confusion_matrix["fn"]
            }
    with open('/Users/apple/XHSworkspace/data/structure/food/models/bad_case_finder/find_bad_data_1017_20211017183235_test_v3_pkl.json', 'w',
              encoding='utf-8') as fp:
        json.dump(show_dict, fp, ensure_ascii=False, indent=4)
    return show_dict


def calculate_pr_for_instance(pred, truth):
    confusion_matrix = defaultdict(list)
    for i in pred:
        if i in truth:
            confusion_matrix["tp"].append(i)
        else:
            confusion_matrix["fp"].append(i)
    for j in truth:
        if j not in pred:
            confusion_matrix["fn"].append(j)
    tp = len(confusion_matrix["tp"])
    fp = len(confusion_matrix["fp"])
    fn = len(confusion_matrix["fn"])
    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    return precision, recall, confusion_matrix


if __name__ == '__main__':
    label_index1 = '/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/label_index_food.json'
    d = NerDataset(label_index1)

    # model_file = '/Users/apple/XHSworkspace/data/structure/food/models/20211017/20211017183235/model_steps_1300.pt'  # 最佳模型：0.872, 0.818, 0.844
    model_file = '/Users/apple/XHSworkspace/data/structure/food/models/20211017/20211018180346/model_steps_1800.pt'  # 最佳模型：0.872, 0.818, 0.844
    # model_file = '/Users/apple/XHSworkspace/data/structure/food/models/20210923162459/model_steps_700.pt'
    bert_config = '/Users/apple/XHSworkspace/data/structure/food/models/bert_base.json'
    vocab_file = '/Users/apple/XHSworkspace/data/structure/food/models/vocab.txt'
    label_index = '/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/label_index_food.json'
    max_len = 150
    p = Predictor(bert_config, vocab_file, model_file, label_index)

    res = predict_new_data()
    find_bad_data(res)
