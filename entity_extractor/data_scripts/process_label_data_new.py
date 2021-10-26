import random
import re
import os
import json
import pandas as pd
from data_utils import *
from collections import defaultdict


class LabelDataProcessor:

    def __init__(self):
        self.priority = ['龚宇娟', '黄佳妮', '王念慈', '万婷茹']
        # self.priority = ['龚宇娟','王悦','黄佳妮','万婷茹','赵鑫梅','王强','汪昕','王念慈','汪圣皓','邱震宇','万峻豪']
        self.ner_type = {"工艺": "0", "功效": "1", "食品": "2", "食材": "3", "口味": "4", "工具": "5", "适宜人群": "6", "品牌": "7",
                         "美食概念词": "8", "否定修饰": "9"}
        self.BIO_PATTERN_ALL = re.compile(r'(?=\[(.+?)\](_(.+?)_))')
        self.segs = ['/t/t/t ', '/t /t ', '/t/t']

    def disambiguation(self, target, ner_index):
        if len(ner_index[target['text']]) == 1:
            return ner_index[target['text']][0]
        if target['text'] not in ner_index or len(ner_index[target['text']]) == 0:
            print(target['text'])
            return None
        cand = ner_index[target['text']]
        tmp = cand[0]
        min_length = abs(tmp['startPosition'] - target['startPosition'])
        for c in cand:
            cur_length = abs(c['startPosition'] - target['startPosition'])
            if cur_length < min_length:
                tmp = c
                min_length = cur_length
        return tmp

    def process_file_from_json(self):
        rst = {}
        org = json.load(
            open(
                # '/Users/apple/XHSworkspace/data/structure/food/tagging/double_verify/20210928_双盲/0928_uniformity_rel.json',
                '/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_rel4_pkl/20211014_ner_nodup_20210928_large_3100_21p.json',
                'r', encoding='utf-8'))
        org2 = json.load(
            open(
                # '/Users/apple/XHSworkspace/data/structure/food/tagging/double_verify/20210928_双盲/0928_uniformity_rel.json',
                '/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_rel4_pkl/20211014_ner_nodup_20210928_less_3100.json',
                'r', encoding='utf-8'))
        for i in org2:
            org[i] = org2[i]

        test_case = json.load(
            open('/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/test_v3.json',
                 'r', encoding='utf-8'))

        for i in org:

            # print(i)
            one_data = {}
            row = org[i]  ## david 处理后的 json

            # if row["note_id"] != "602bb701000000000102b4f1":
            #     continue

            # row = org[i][0]  ## 双盲json
            # if row['relation_uni'] != "None" and 0 < float(row['relation_uni']) < 0.5:
            #     continue
            # text_ttt = row["results"][0]["result"].strip("\'")
            text_ttt = row["relation"]
            parse_ttt = parse_from_file_ttt_helper(text_ttt)
            line = parse_ttt['data']
            one_data['org_data'] = line
            # one_data['user'] = row['results'][0]["userName"]
            one_data['user'] = row['user']
            splits = line.split('***')
            if len(splits) == 0:
                continue
            if splits[0].split("/t")[0] in test_case:
                continue
            ner_rst = []
            ner_index = defaultdict(list)
            line = splits[0]
            match = re.search(self.BIO_PATTERN_ALL, line)
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
                match = re.search(self.BIO_PATTERN_ALL, line)
            seg = ''
            for s in self.segs:
                if line.find(s) != -1:
                    seg = s
                    break
            if seg == '':
                continue
            splits1 = line.split(seg, 1)
            note_id = splits1[0]
            org_data = splits1[1]
            one_data['text'] = org_data
            for cur in ner_rst:
                cur['startPosition'] = cur['startPosition'] - (len(note_id) + len(seg))
                cur['endPosition'] = cur['endPosition'] - (len(note_id) + len(seg))
            one_data['ner'] = ner_rst
            if len(splits) == 1 or splits[1] == '':
                rst[note_id] = one_data
                continue
            sss = splits[len(splits) - 1]
            sss = re.sub('\'', '\"', sss)
            try:
                relation_rst = json.loads(sss)
            except:
                continue
            for cur in relation_rst:
                cur['s']['startPosition'] = cur['s']['startPosition'] - (len(note_id) + len(seg))
                cur['s']['endPosition'] = cur['s']['endPosition'] - (len(note_id) + len(seg))
                cur['o']['startPosition'] = cur['o']['startPosition'] - (len(note_id) + len(seg))
                cur['o']['endPosition'] = cur['o']['endPosition'] - (len(note_id) + len(seg))
            useless = []
            for j, one in enumerate(relation_rst):
                s = self.disambiguation(one['s'], ner_index)
                o = self.disambiguation(one['o'], ner_index)
                if s is None or o is None:
                    useless.append(j)
                one['s'] = s
                one['o'] = o
            for j in useless[::-1]:
                relation_rst.pop(j)
            one_data['relation'] = relation_rst
            # if user in self.priority or note_id not in rst:
            rst[i] = one_data
        return rst

    def process_file(self, file):
        rst = {}
        df = pd.read_pickle(file)
        for i, row in df.iterrows():
            print(i)
            one_data = {}
            if row['note_id'] == '6013e2ef000000000101dcc3':
                print()
            if not row['check_tagging_sense']:
                continue
            line = row['re_tagging_rel']
            user = 'default'
            if "userName" in row:
                user = row['userName']
            line = line.strip()
            one_data['org_data'] = line
            one_data['user'] = user
            splits = line.split('***')
            if len(splits) == 0:
                continue
            ner_rst = []
            ner_index = defaultdict(list)
            line = splits[0]
            match = re.search(self.BIO_PATTERN_ALL, line)
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
                match = re.search(self.BIO_PATTERN_ALL, line)
            seg = ''
            for s in self.segs:
                if line.find(s) != -1:
                    seg = s
                    break
            if seg == '':
                continue
            splits1 = line.split(seg, 1)
            note_id = splits1[0]
            org_data = splits1[1]
            one_data['text'] = org_data
            for cur in ner_rst:
                cur['startPosition'] = cur['startPosition'] - (len(note_id) + len(seg))
                cur['endPosition'] = cur['endPosition'] - (len(note_id) + len(seg))
            one_data['ner'] = ner_rst
            if len(splits) == 1 or splits[1] == '':
                rst[note_id] = one_data
                continue
            sss = splits[len(splits) - 1]
            sss = re.sub('\'', '\"', sss)
            try:
                relation_rst = json.loads(sss)
            except:
                continue
            for cur in relation_rst:
                cur['s']['startPosition'] = cur['s']['startPosition'] - (len(note_id) + len(seg))
                cur['s']['endPosition'] = cur['s']['endPosition'] - (len(note_id) + len(seg))
                cur['o']['startPosition'] = cur['o']['startPosition'] - (len(note_id) + len(seg))
                cur['o']['endPosition'] = cur['o']['endPosition'] - (len(note_id) + len(seg))
            useless = []
            for j, one in enumerate(relation_rst):
                s = self.disambiguation(one['s'], ner_index)
                o = self.disambiguation(one['o'], ner_index)
                if s is None or o is None:
                    useless.append(j)
                one['s'] = s
                one['o'] = o
            for j in useless[::-1]:
                relation_rst.pop(j)
            one_data['relation'] = relation_rst
            if user in self.priority or note_id not in rst:
                rst[note_id] = one_data
        return rst


if __name__ == '__main__':
    p = LabelDataProcessor()

    rst = p.process_file_from_json()

    with open('/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/20210922/20211015_rel_all.json', 'w',
              encoding='utf-8') as fp:
        json.dump(rst, fp, ensure_ascii=False, indent=4)

    dir = 'label_data_new/20210922_rel4_pkl'
    files = os.listdir(dir)
    rst = {}
    for file_name in files:
        print(file_name)
        file = os.path.join(dir, file_name)
        file = 'label_data_new/food_57_rel6_2863_2_57.pkl'
        cur_rst = p.process_file(file)
        for id in cur_rst:
            rst[id] = cur_rst[id]
        break
    fout = open('label_data_new/test_v0.json', 'w', encoding='utf-8')
    json.dump(rst, fout, ensure_ascii=False, indent=4)
    # ids = []
    # for note_id in rst:
    #     if rst[note_id]['user'] in p.priority:
    #         ids.append(note_id)
    # test_ids = random.sample(ids, 200)
    # test = {}
    # train = {}
    # for note_id in rst:
    #     if note_id in test_ids:
    #         test[note_id] = rst[note_id]
    #     else:
    #         train[note_id] = rst[note_id]
    # fout = open('label_data_new/0922.json', 'w', encoding='utf-8')
    # json.dump(train, fout, ensure_ascii=False, indent=4)
    # fout = open('label_data_new/test_v1.json', 'w', encoding='utf-8')
    # json.dump(test, fout, ensure_ascii=False, indent=4)
    # print(len(rst))
    # user_cnt = defaultdict(int)
    # cnt = 0
    # for id in rst:
    #     if 'relation' in rst[id]:
    #         cnt += len(rst[id]['relation'])
    #     user = rst[id]['user']
    #     if user in p.priority:
    #         user_cnt[user] += 1
    # print(user_cnt)
    # print(cnt)
