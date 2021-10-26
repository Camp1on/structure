# -*- coding: utf-8 -*-
# @Time    : 2021/9/11 10:51
# @Author  : heyee (jialeyang)

import re
import json
from collections import defaultdict
import os.path as osp
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
BIO_PATTERN_ALL = re.compile(r'(?=\[(.+?)\](_(.+?)_))')


def disambiguation(target, ner_index):
    if len(ner_index[target['text']]) == 1:
        return ner_index[target['text']][0]
    # print(target['text'])
    cand = ner_index[target['text']]
    tmp = cand[0]
    min_length = abs(tmp['startPosition'] - target['startPosition'])
    for c in cand:
        cur_length = abs(c['startPosition'] - target['startPosition'])
        if cur_length < min_length:
            tmp = c
            min_length = cur_length
    return tmp


# dir_path = "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_david_csv"
dir_path = "/Users/apple/XHSworkspace/data/structure/food/tagging/double_verify/20211013"
# dir_path = "/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_david_csv"
# output_dir = "/Users/apple/XHSworkspace/data/structure/food/tagging/double_verify/high_quality_id"
# fin = open(osp.join(dir_path, 'food.3019'), 'r', encoding='utf-8')

# with open(osp.join(dir_path, "20210926_双盲_19.txt"), 'r', encoding='utf-8') as reader:
#     tmp = reader.read()
#     lines = tmp.split("\n")

lines = []
# org = json.load(
#     open('/Users/apple/XHSworkspace/data/structure/food/tagging/double_verify/20210926_双盲/0926_uniformity_.json',
#          'r', encoding='utf-8'))
# for i in org:
#     lines.append(org[i][0]["results"][0]["result"])

file_name_ttt = "美食_正式标注_1100_19th_1013_西安_全部_20211013232447.csv"
pd2 = pd.read_csv(osp.join(dir_path, file_name_ttt))
# pd1 = pd.read_csv(osp.join(dir_path, '双盲标注_185_20210930-1_全部_20211008194552.csv'))
# df = pd.concat([pd1, pd2])
df = pd2
cnt = 0
rst = defaultdict(list)

for i, line in enumerate(lines):
    try:
        line = line.replace("/t/t/t ", "/t/t")
        line = line.replace("/t /t ", "/t/t")
        one_data = {}
        one_data['org_data'] = line
        line = line.strip().strip('\\\'')
        splits = line.split('***')
        if len(splits) == 0:
            continue
        while line.find('\\\\') != -1:
            line = re.sub('\\\\\\\\', '\\\\', line)
        org_line = line
        ner_rst = []
        ner_index = defaultdict(list)
        match = re.search(BIO_PATTERN_ALL, line)
        ner_start = []
        while match is not None:
            text_span = match.regs[1]
            type_span = match.regs[3]
            text = line[text_span[0]:text_span[1]]
            type = line[type_span[0]:type_span[1]]
            one = {'startPosition': match.regs[0][0], 'endPosition': match.regs[0][0] + len(text), 'type': type,
                   'text': text}
            ner_rst.append(one)
            ner_index[text].append(one)
            ner_start.append(match.regs[0][0])
            replace_text = line[match.regs[0][0]: match.regs[2][1]]
            line = line.replace(replace_text, text, 1)
            match = re.search(BIO_PATTERN_ALL, line)
        new_line = re.sub('\\\\t', '/t', line)
        splits1 = new_line.split('/t/t', 1)
        note_id = splits1[0]
        org_data = splits1[1]
        new_splits = org_data.split('***')
        text = new_splits[0]
        one_data['text'] = text
        for cur in ner_rst:
            cur['startPosition'] = cur['startPosition'] - (len(note_id) + 4)
            cur['endPosition'] = cur['endPosition'] - (len(note_id) + 4)
        one_data['ner'] = ner_rst
        if len(splits) == 1 or splits[1] == '':
            rst[note_id].append(one_data)
            continue

        relation_start = []
        relation_rst = json.loads(splits[len(splits) - 1])
        for cur in relation_rst:
            cur['s']['startPosition'] = cur['s']['startPosition'] - (len(note_id) + 4)
            cur['s']['endPosition'] = cur['s']['endPosition'] - (len(note_id) + 4)
            cur['o']['startPosition'] = cur['o']['startPosition'] - (len(note_id) + 4)
            cur['o']['endPosition'] = cur['o']['endPosition'] - (len(note_id) + 4)
        for one in relation_rst:
            try:
                one['s'] = disambiguation(one['s'], ner_index)
                one['o'] = disambiguation(one['o'], ner_index)
                relation_start.append(one['s']['startPosition'])
                relation_start.append(one['o']['startPosition'])
            except:
                print()
        relation_start = list(set(relation_start))
        relation_start.sort()
        # for cur in relation_rst:
        #     cur['s']['startPosition'] = cur['s']['startPosition'] - (len(note_id)+4)
        #     cur['s']['endPosition'] = cur['s']['endPosition'] - (len(note_id)+4)
        #     cur['o']['startPosition'] = cur['o']['startPosition'] - (len(note_id) + 4)
        #     cur['o']['endPosition'] = cur['o']['endPosition'] - (len(note_id) + 4)
        one_data['relation'] = relation_rst
        rst[note_id].append(one_data)
    except:
        print(line)

for index, row in df.iterrows():
    if row["label"] != "完成":
        continue
    try:
        line = row["result"].strip("\'")
        line = line.replace("/t/t/t ", "/t/t")
        line = line.replace("/t /t ", "/t/t")
        one_data = {}
        one_data['org_data'] = line
        line = line.strip().strip('\\\'')
        splits = line.split('***')
        if len(splits) == 0:
            continue
        while line.find('\\\\') != -1:
            line = re.sub('\\\\\\\\', '\\\\', line)
        org_line = line
        ner_rst = []
        ner_index = defaultdict(list)
        match = re.search(BIO_PATTERN_ALL, line)
        ner_start = []
        while match is not None:
            text_span = match.regs[1]
            type_span = match.regs[3]
            text = line[text_span[0]:text_span[1]]
            type = line[type_span[0]:type_span[1]]
            one = {'startPosition': match.regs[0][0], 'endPosition': match.regs[0][0] + len(text), 'type': type,
                   'text': text}
            ner_rst.append(one)
            ner_index[text].append(one)
            ner_start.append(match.regs[0][0])
            replace_text = line[match.regs[0][0]: match.regs[2][1]]
            line = line.replace(replace_text, text, 1)
            match = re.search(BIO_PATTERN_ALL, line)
        new_line = re.sub('\\\\t', '/t', line)
        splits1 = new_line.split('/t/t', 1)
        note_id = splits1[0]
        org_data = splits1[1]
        new_splits = org_data.split('***')
        text = new_splits[0]
        one_data['text'] = text
        for cur in ner_rst:
            cur['startPosition'] = cur['startPosition'] - (len(note_id) + 4)
            cur['endPosition'] = cur['endPosition'] - (len(note_id) + 4)
        one_data['ner'] = ner_rst
        if len(splits) == 1 or splits[1] == '':
            rst[note_id].append(one_data)
            continue

        relation_start = []
        relation_rst = json.loads(splits[len(splits) - 1])
        for cur in relation_rst:
            cur['s']['startPosition'] = cur['s']['startPosition'] - (len(note_id) + 4)
            cur['s']['endPosition'] = cur['s']['endPosition'] - (len(note_id) + 4)
            cur['o']['startPosition'] = cur['o']['startPosition'] - (len(note_id) + 4)
            cur['o']['endPosition'] = cur['o']['endPosition'] - (len(note_id) + 4)
        for one in relation_rst:
            try:
                one['s'] = disambiguation(one['s'], ner_index)
                one['o'] = disambiguation(one['o'], ner_index)
                relation_start.append(one['s']['startPosition'])
                relation_start.append(one['o']['startPosition'])
            except:
                print()
        relation_start = list(set(relation_start))
        relation_start.sort()
        # for cur in relation_rst:
        #     cur['s']['startPosition'] = cur['s']['startPosition'] - (len(note_id)+4)
        #     cur['s']['endPosition'] = cur['s']['endPosition'] - (len(note_id)+4)
        #     cur['o']['startPosition'] = cur['o']['startPosition'] - (len(note_id) + 4)
        #     cur['o']['endPosition'] = cur['o']['endPosition'] - (len(note_id) + 4)
        one_data['relation'] = relation_rst
        rst[note_id].append(one_data)
    except:
        print(row)


def process_one(one):
    all_ner = set()
    for ner in one['ner']:
        # all_ner.add(ner['text'] + ',' + str(ner['startPosition']) + ',' + str(ner['endPosition']))
        all_ner.add(ner['text'] + ',' + ner['type'] + ',' + str(ner['startPosition']) + ',' + str(ner['endPosition']))
    all_relation = set()
    if 'relation' in one:
        for relation in one['relation']:
            first_entity = relation['s']['text'] + ',' + relation['s']['type'] + ',' + str(
                relation['s']['startPosition']) + ',' + str(relation['s']['endPosition'])
            second_entity = relation['o']['text'] + ',' + relation['o']['type'] + ',' + str(
                relation['o']['startPosition']) + ',' + str(relation['o']['endPosition'])
            # first_entity = relation['s']['text'] + ',' + str(
            #     relation['s']['startPosition']) + ',' + str(relation['s']['endPosition'])
            # second_entity = relation['o']['text'] + ','+ str(
            #     relation['o']['startPosition']) + ',' + str(relation['o']['endPosition'])
            all_relation.add(first_entity + ',' + relation['p'] + ',' + second_entity)
    return all_ner, all_relation


all_cnt = 0
ner_inter = 0
ner_union = 0
relation_inter = 0
relation_union = 0

fout = open(osp.join(dir_path, '0929_uniformity_verify.txt'), 'w',
            encoding='utf-8')
user_df = df
user_df["note_id"] = user_df["content"].apply(lambda x: x.split("/t")[0])
ana_res = defaultdict(list)
name = ""

res_ttt = defaultdict(list)
res_ttt_entity = defaultdict(lambda: defaultdict(list))

for note_id in rst:
    cur = rst[note_id]
    if len(rst[note_id]) != 2:
        continue
    # if note_id!="60275f2c0000000001028688":
    #     continue
    all_cnt += 1
    first_ner, first_relation = process_one(rst[note_id][0])
    second_ner, second_relation = process_one(rst[note_id][1])
    if len(first_relation) or len(first_relation) > 0:
        print()
    for i in first_ner:
        if i not in second_ner:
            i_type = i.split(",")[1]
            res_ttt[i_type] = res_ttt.get(i_type, 0) + 1
            res_ttt_entity[i_type][note_id].append("1st: {}".format(i))
    for j in second_ner:
        if j not in first_ner:
            j_type = j.split(",")[1]
            res_ttt[j_type] = res_ttt.get(j_type, 0) + 1
            res_ttt_entity[j_type][note_id].append("2nd: {}".format(j))
    cur_ner_inter = len(first_ner.intersection(second_ner))
    cur_ner_union = len(first_ner.union(second_ner))
    ner_inter += cur_ner_inter
    ner_union += cur_ner_union
    ner_uni = 'None' if cur_ner_union == 0 else round((cur_ner_inter * 1.0 / cur_ner_union), 3)
    if cur_ner_union > 0 and cur_ner_inter * 1.0 / cur_ner_union < 0.5:
        print('ner: ' + note_id)

    cur_relation_inter = len(first_relation.intersection(second_relation))
    cur_relation_union = len(first_relation.union(second_relation))
    relation_inter += cur_relation_inter
    relation_union += cur_relation_union
    relation_uni = 'None' if cur_relation_union == 0 else round((cur_relation_inter * 1.0 / cur_relation_union), 3)
    if cur_relation_union > 0 and cur_relation_inter * 1.0 / cur_relation_union < 0.5:
        print('relation: ' + note_id)

    fout.write(note_id + '\t' + str(ner_uni) + '\t' + str(relation_uni) + '\n')

    if (cur_ner_union > 0 and cur_ner_inter * 1.0 / cur_ner_union < 0.7) \
            and (cur_relation_union > 0 and cur_relation_inter * 1.0 / cur_relation_union < 0.5):
        # if (cur_ner_union > 0 and cur_ner_inter * 1.0 / cur_ner_union > 0.7) and \
        #         ((cur_relation_union > 0 and cur_relation_inter * 1.0 / cur_relation_union > 0)
        #          or cur_relation_union == 0):
        res_results = []
        for idi, i in user_df[user_df["note_id"] == note_id].iterrows():
            res_results.append(
                {
                    "userName": i["userName"],
                    "label": i["label"],
                    "result": i["result"].split("***")[0]
                    # "result": i["result"]
                }
            )
            name = i["userName"]
        ana_res[note_id].append({
            "ner_uni": str(ner_uni),
            "relation_uni": str(relation_uni),
            "results": res_results,
            "参考答案": [w for w in lines if note_id in w]
        })

with open(osp.join(dir_path, '{}_一致率质检结果.json'.format(file_name_ttt.replace(".", "_"))), 'w',
          encoding='utf-8') as fp:
    json.dump(ana_res, fp, ensure_ascii=False, indent=4)
# with open(osp.join(dir_path, '0929_uniformity.json'.format(name)), 'w',
#           encoding='utf-8') as fp:
#     json.dump(ana_res, fp, ensure_ascii=False, indent=4)
print(all_cnt)
print(ner_inter * 1.0 / ner_union)
print(relation_inter * 1.0 / relation_union)
