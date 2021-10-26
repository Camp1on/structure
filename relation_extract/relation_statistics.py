import json
import time

from relation_extract_handler import RelationExtractHandler, clear_duplicate

constrain = ['食材-描述-食品', '品牌-描述-食品', '口味-描述-食品', '美食概念词-描述-食品', '工艺-描述-食品', '功效-描述-食品', '适宜人群-描述-食品',
             '口味-描述-食材', '品牌-描述-食材', '功效-描述-食材', '品牌-描述-美食概念词', '口味-描述-美食概念词', '美食概念词-描述-美食概念词',
             '适宜人群-描述-美食概念词', '功效-描述-美食概念词', '食材-描述-美食概念词', '否定修饰-描述-口味', '否定修饰-描述-功效']


def clear_real(real_relation):
    rst = []
    for relation in real_relation:
        key = relation['s']['type'] + '-' + '描述' + '-' + relation['o']['type']
        if key in constrain:
            rst.append(relation)
    return rst


def relation_process(relation_rst):
    rst = set()
    for one in relation_rst:
        text = one['s']['text'] + '-' + one['o']['text']
        rst.add(text)
    return rst


def process_one(ner_rst, relation_rst):
    index = {}
    for i, cur_ner in enumerate(ner_rst):
        start = cur_ner['startPosition']
        index[start] = i
    relation = []
    relation_type = []
    for cur_relation in relation_rst:
        s_start = cur_relation['s']['startPosition']
        s_index = index[s_start]
        s_text = cur_relation['s']['text']
        o_start = cur_relation['o']['startPosition']
        o_index = index[o_start]
        o_text = cur_relation['o']['text']
        p_text = cur_relation['p']
        relation.append(s_text + ',' + str(s_index) + '-' + p_text + '-' + o_text + ',' + str(o_index))
        relation_type.append(cur_relation['s']['type'] + '-' + p_text + '-' + cur_relation['o']['type'])
    return relation, relation_type


if __name__ == '__main__':
    test_file = '/Users/liukangping/ut-caesar-server/ut-caesar/structure/relation_extract/test_v2.json'
    fin = open(test_file, 'r', encoding='utf-8')
    data = json.load(fin)
    domain = '美食'
    relationExtractHandler = RelationExtractHandler()
    num_correct = 0
    num_real = 0
    num_pred = 0
    detail_cnt = {}
    i = 0
    for note_id in data:
        i += 1
        # note_id = '5faf52b3000000000101c7a4'
        if note_id == '5f8461a00000000001004398':
            print()
        print(note_id)
        text = data[note_id]['text']
        ner = data[note_id]['ner']
        ner = sorted(ner, key=lambda d: d['startPosition'])
        ner_list = [cur['type'] + '-' + cur['text'] for cur in ner]
        if 'relation' not in data[note_id]:
            continue
        relation_real = data[note_id]['relation']
        relation_real = clear_real(relation_real)
        relation_real = clear_duplicate(relation_real)
        relation_real, relation_type_real = process_one(ner, relation_real)
        relation_real = list(set(relation_real))
        relation_real.sort()

        relation_pred = relationExtractHandler.predict(domain, text, ner)
        relation_pred = clear_duplicate(relation_pred)
        relation_pred, relation_type_pred = process_one(ner, relation_pred)
        relation_pred = list(set(relation_pred))
        relation_pred.sort()

        num_real += len(relation_real)
        num_pred += len(relation_pred)
        print(text)
        print(ner_list)
        print(relation_real)
        print(relation_pred)
        inter = set(relation_real).intersection(set(relation_pred))
        num_correct += len(inter)

        for type_real in relation_type_real:
            if type_real not in detail_cnt:
                detail_cnt[type_real] = [0, 0, 0]
            detail_cnt[type_real][2] += 1
        for type_pred in relation_type_pred:
            if type_pred not in detail_cnt:
                detail_cnt[type_pred] = [0, 0, 0]
            detail_cnt[type_pred][1] += 1
        for i, type_pred in enumerate(relation_type_pred):
            for j, type_real in enumerate(relation_type_real):
                if relation_pred[i] == relation_real[j]:
                    detail_cnt[type_pred][0] += 1

    p = num_correct*1.0 / num_pred
    r = num_correct*1.0 / num_real
    f1 = (2.0*p*r)/(r+p)
    print('precision: ' + str(round(p, 3)))
    print('recall: ' + str(round(r, 3)))
    print('F1: ' + str(round(f1, 3)))
    tmp = sorted(detail_cnt.items(), key=lambda d: d[1][2], reverse=True)
    for t, cnt in tmp:
        num_correct = cnt[0]
        if cnt[1] != 0 and cnt[2] != 0:
            p = num_correct * 1.0 / cnt[1]
            r = num_correct * 1.0 / cnt[2]
        elif cnt[1] == 0 and cnt[2] != 0:
            p = 0.0
            r = num_correct * 1.0 / cnt[2]
        elif cnt[1] != 0 and cnt[2] == 0:
            p = num_correct * 1.0 / cnt[1]
            r = 0.0
        else:
            p = 0.0
            r = 0.0
        if p < 0.00001 and p < 0.00001:
            f1 = 0.0
        else:
            f1 = (2.0 * p * r) / (r + p)
        print(t + '\t' + str(round(p, 3)) + '\t' + str(round(r, 3)) + '\t' + str(round(f1, 3)) + '\t' + str(detail_cnt[t][0]) + '\t' + str(detail_cnt[t][1]) + '\t' + str(detail_cnt[t][2]))









