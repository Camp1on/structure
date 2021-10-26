# -*- coding: utf-8 -*-
# @Time    : 2021/10/14 11:16
# @Author  : heyee (jialeyang)


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
from collections import defaultdict
from ner.universal_v2.ner_handler import UniverNerV2Serving
from ner.poi_ner.poi_ner_handler import PoiNerV2Serving

import json
import time
from structure.structure_handler import NoteStructureServing
from data_utils import *

tqdm.pandas()
BIO_PATTERN_ALL = re.compile(r'(?=\[(.+?)\](_(.+?)_))')  # 带有捕获组的正向超前断言 且 最短匹配


def predict_new_data():
    name = "20th_1400_1014_西安"
    dir_path = "/Users/apple/XHSworkspace/data/structure/food/"
    data_50_200 = read_file("{}{}".format(dir_path, name))
    res = []
    for i in tqdm(data_50_200):
        res.append(p.predict(remove_all_tag(i)))
    write_file(data=[w["textPredict"] for w in res],
               file="{}{}_pred".format(dir_path, name))


def remove_all_tag(content):
    res = BIO_PATTERN_ALL.findall(content)
    for i in res:
        content = content.replace("[{}]{}".format(i[0], i[1]), r'{}'.format(i[0]))
        # content = re.sub(r'\[{}\](_{}_)'.format(i[0], i[1][1:-1]), r'{}'.format(i[0]), content)
    return content


if __name__ == '__main__':
    model_path = S3Util.Instance().get_latest_model_path(settings['note_structure']['entity_extract_model_path'])
    model_file = osp.join(model_path, 'model.pt')
    bert_config = osp.join(model_path, 'bert_base.json')
    vocab_file = osp.join(model_path, 'vocab.txt')
    label_index = osp.join(model_path, 'label_index_food.json')
    d = NerDataset(label_index)
    p = Predictor(bert_config, vocab_file, model_file, label_index)

    predict_new_data()
