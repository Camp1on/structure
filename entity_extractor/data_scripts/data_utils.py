# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 10:50
# @Author  : heyee (jialeyang)
import copy

from structure.entity_extractor.data import NerDataset
import pandas as pd

from tqdm import tqdm
import re
import os.path as osp
import json
from utils.s3_utils import S3Util
import os.path as osp
from conf.config import settings

tqdm.pandas()
BIO_PATTERN_ALL = re.compile(r'(?=\[(.+?)\](_(.+?)_))')  # 带有捕获组的正向超前断言 且 最短匹配
MODEL_path = S3Util.Instance().get_latest_model_path(settings['note_structure']['entity_extract_model_path'])
LABEL_index = osp.join(MODEL_path, 'label_index_food.json')
d = NerDataset(LABEL_index)
from collections import defaultdict


# d = NerDataset("/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/label_index_food_old.json")


def read_file(file):
    """ file format:   note_id \t\t property \t\t note
    """
    with open(file, 'r', encoding='utf-8') as reader:
        tmp = reader.read()
        lines = tmp.split("\n")
    return [w for w in lines if len(w) > 0]


def write_file(file, data):
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
            result_ori = copy.deepcopy(line)
            for i in tmp:
                line = line.replace(i, "/")
            splits = line.split("***")
            note_id = line.split("/t")[0]
            ner_d = splits[0][len(note_id):][4:]
            if len(splits) > 1:
                relation_d = list(json.loads(splits[-1]))
            else:
                relation_d = []
            res.append({
                "note_id": line.split("/t")[0],
                "data": line,
                "org_data": result_ori,
                "plain_text": remove_all_tag(ner_d),
                "ner_d": ner_d,
                "parsed_res": parse_tags(ner_d),
                "parsed_res_relation": relation_d
            })

    # json.dump(res, open('/Users/apple/XHSworkspace/data/structure/food/config/test.json', 'w', encoding='utf-8'),
    #           ensure_ascii=False, indent=4)
    return res


def parse_from_file_ttt(file):
    res = defaultdict(list)
    with open(file, 'r', encoding='utf-8') as reader:
        tmp = reader.read()
        lines = tmp.split("\n")
    for k, line in enumerate(lines):
        if len(line) > 0:
            tmp = re.compile(r'\\+').findall(line)
            tmp.sort(key=lambda s: len(s), reverse=True)
            result_ori = copy.deepcopy(line)
            for i in tmp:
                line = line.replace(i, "/")
            line = line.replace("/t /t ", "/t/t")
            line = line.replace("/t/t/t ", "/t/t")
            splits = line.split("***")
            note_id = line.split("/t")[0]
            ner_d = splits[0][len(note_id):][4:]
            if len(splits) > 1:
                relation_d = list(json.loads(splits[-1]))
            else:
                relation_d = []
            res[note_id] = {
                "note_id": line.split("/t")[0],
                "data": line,
                "org_data": result_ori,
                "user": "",
                "text": remove_all_tag(ner_d),
                "ner_d": ner_d,
                "ner": parse_tags(ner_d),
                "relation": relation_d
            }

    # json.dump(res, open('/Users/apple/XHSworkspace/data/structure/food/config/test.json', 'w', encoding='utf-8'),
    #           ensure_ascii=False, indent=4)
    return res


def parse_from_file_ttt_helper(line):
    res = defaultdict(list)
    if len(line) > 0:
        tmp = re.compile(r'\\+').findall(line)
        tmp.sort(key=lambda s: len(s), reverse=True)
        result_ori = copy.deepcopy(line)
        for i in tmp:
            line = line.replace(i, "/")
        line = line.strip("\'")
        line = line.replace("/t /t ", "/t/t")
        line = line.replace("/t/t/t ", "/t/t")
        splits = line.split("***")
        note_id = line.split("/t")[0]
        ner_d = splits[0][len(note_id):][4:]
        if len(splits) > 1:
            try:
                relation_d = json.loads(splits[-1].replace("\'", "\""))
            except:
                relation_d = []
        else:
            relation_d = []
        res = {
            "note_id": line.split("/t")[0],
            "data": line,
            "org_data": result_ori,
            "user": "",
            "text": remove_all_tag(ner_d),
            "ner_d": ner_d,
            "ner": parse_tags(ner_d),
            "relation": relation_d
        }

    return res


def parse_from_csv(file):
    res = []
    data = pd.read_csv(file, error_bad_lines=False)

    for k, line in data.iterrows():
        if len(line) > 0 and line["label"] == "完成":
            """
            \'5ffa7b1a000000000100364c/t /t #demo自律打卡#day48/t /t 自律生活/t [早餐]_美食概念词_分享/t 晨重:44.1(我怀疑称坏了,要么昨晚电热毯一晚上把我水分蒸发了)/t 晨起拉伸/t 晨起翻译/t 阅读打卡/t 碎碎念:昨天拔完牙后,对我没啥影响, ##sep## 但还是谨遵医嘱没吃过分的,但吃的也不少,n个[虎皮蛋糕]_食品_ [豆浆]_食品_ [牛奶]_食品_ [燕麦]_食品_ 反正没饿着自己,今天还下降了体重 真的奇怪了哈哈哈今早也是软绵, ##sep## 下午要去考试我考试顺利呀***[{"s":{"startPosition":58,"endPosition":60,"type":"美食概念词","text":"早餐","index":"0"},"p":"描述","o":{"startPosition":171,"endPosition":175,"type":"食品","text":"虎皮蛋糕","index":"1"}},{"s":{"startPosition":58,"endPosition":60,"type":"美食概念词","text":"早餐","index":"0"},"p":"描述","o":{"startPosition":176,"endPosition":178,"type":"食品","text":"豆浆","index":"2"}},{"s":{"startPosition":58,"endPosition":60,"type":"美食概念词","text":"早餐","index":"0"},"p":"描述","o":{"startPosition":179,"endPosition":181,"type":"食品","text":"牛奶","index":"3"}},{"s":{"startPosition":58,"endPosition":60,"type":"美食概念词","text":"早餐","index":"0"},"p":"描述","o":{"startPosition":182,"endPosition":184,"type":"食品","text":"燕麦","index":"4"}}]\'
            """
            result = line["result"].strip("\'")
            result_ori = copy.deepcopy(result)
            label = line["label"]
            userName = line["userName"]
            tmp = re.compile(r'\\+').findall(result)
            tmp.sort(key=lambda s: len(s), reverse=True)
            for i in tmp:
                result = result.replace(i, "/")
            result = result.replace("/t/t/t ", "/t/t")
            result = result.replace("/t /t ", "/t/t")
            splits = result.split("***")
            note_id = result.split("/t")[0]
            ner_d = splits[0][len(note_id):][4:]
            if len(splits) > 1:
                relation_d = list(json.loads(splits[-1]))
            else:
                relation_d = []
            res.append({
                "note_id": note_id,
                "label": label,
                "userName": userName,
                "data_ori": result_ori,
                "data": result,
                "plain_text": remove_all_tag(ner_d),
                "ner_d": ner_d,
                "parsed_res": parse_tags(ner_d),
                "parsed_res_relation": relation_d
            })

    # json.dump(res, open('/Users/apple/XHSworkspace/data/structure/food/config/test.json', 'w', encoding='utf-8'),
    #           ensure_ascii=False, indent=4)
    return res


def remove_all_tag(content):
    content = content.replace("(", "<<")
    content = content.replace(")", ">>")
    res = BIO_PATTERN_ALL.findall(content)
    for i in res:
        content = content.replace("[{}]{}".format(i[0], i[1]), r'{}'.format(i[0]))
        # content = re.sub(r'\[{}\](_{}_)'.format(i[0], i[1][1:-1]), r'{}'.format(i[0]), content)
    content = content.replace("<<", "(")
    content = content.replace(">>", ")")
    return content


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
    try:
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
    except:
        return []


def check_kuohao(s):
    res = True
    stack = []
    for item in list(s):
        if item != "[" and item != "]":
            continue
        elif item == "[":
            if len(stack) != 0:
                res = False
                break
            else:
                stack.append(item)
        else:
            if len(stack) != 1:
                res = False
                break
            else:
                stack.pop()
    return res and len(stack) == 0


def check_tagging_sense(text):
    """ bad case, 强制选用 y_true 替换
        99 / 2492 = 3 %
    """
    res_all = BIO_PATTERN_ALL.findall(text)
    res_notsense = re.compile(r'(?=\[(.+?)\](_(.+?)_))').findall(text)
    return True if len(res_all) == len(res_notsense) and check_kuohao(text) else False


if __name__ == '__main__':
    test = "海宁探店[小高烧烤]_品牌_,[美食]_美食概念词_探店分享/t/t位置比较偏的一家烧烤店,半夜也有营业。菜的种类特别多,是见过的[烧烤]_工艺_品类比较多的一家店了。/t  味道中规中矩,店面非常大,还有一个大荧幕, ##sep## 半夜看球赛的优秀聚集点没有错了。/t  地址:[小高烧烤店]_品牌_,海宁市石泾路14 9号。"
    parse_tags(test)
