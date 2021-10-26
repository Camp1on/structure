import json
import re
import jieba
import sys
sys.path.append('../..')



# 使用近邻的思想进行关系抽取
# 1、同一行保留最先出现的品类，其它忽略（todo 可尝试同一行的近邻方法）
# 2、以品类实体对所有句子进行切分，第i个品类词到第i+1个品类词之前的属性都是描述第i个品类ci
class FoodRuleExtractor:

    def __init__(self):
        self.obj_type = ['食品']
        self.MAX_DIS = 300
        # self.ddp = DDParser()

    # 返回结果类型 spo
    def predict(self, text, entity_rst):
        strict_rst = self.predict_strict(text, entity_rst)
        split_punct = ['。', '？', '！', '?', '!', '\n', '/t']
        rst = []
        obj = []
        for i, cur in enumerate(entity_rst):
            if cur['type'] in self.obj_type:
                # 记录
                obj.append(i)
                continue
            if cur in strict_rst:
                continue
            right_obj = None
            for j in range(i+1, len(entity_rst)):
                if entity_rst[j]['startPosition'] - cur['startPosition'] > self.MAX_DIS // 3:
                    break
                if entity_rst[j]['type'] in self.obj_type:
                    right_obj = entity_rst[j]
                    break
            left_obj = None
            if len(obj) > 0 and cur['startPosition'] - entity_rst[obj[len(obj)-1]]['startPosition'] < self.MAX_DIS // 3:
                left_obj = entity_rst[obj[len(obj)-1]]

            # 找不到被描述对象
            if left_obj is None and right_obj is None:
                continue
            # 只找到一个候选被描述对象
            if left_obj is None or right_obj is None:
                o = left_obj if left_obj is not None else right_obj
                dis = abs(o['startPosition'] - cur['startPosition'])
                if dis < self.MAX_DIS // 3:
                    rst.append({'s': cur, 'p': '描述', 'o': o, 'score': self.get_score(cur, o)})
                continue
            # 判断是否跨句
            left_flag = False
            for l in range(left_obj['endPosition'], cur['startPosition']):
                if text[l] in split_punct or text[l:l+2] in split_punct:
                    left_flag = True
                    break
            right_flag = False
            for r in range(cur['endPosition'], right_obj['startPosition']):
                if text[r] in split_punct or text[r:r+2] in split_punct:
                    right_flag = True
                    break
            # 两边都未跨句或者都跨句，取最近
            if (not left_flag and not right_flag) or (left_flag and right_obj):
                left_dis = abs(left_obj['startPosition'] - cur['startPosition'])
                right_dis = abs(right_obj['startPosition'] - cur['startPosition'])
                o = left_obj if left_dis//2.5 <= right_dis else right_obj
                rst.append({'s': cur, 'p': '描述', 'o': o, 'score': self.get_score(cur, o)})
                # rst.append({'s': cur, 'p': '描述', 'o': right_obj, 'score': self.get_score(cur, o)})
                # rst.append({'s': cur, 'p': '描述', 'o': left_obj, 'score': self.get_score(cur, o)})
                continue
            # 一边跨句一边未跨句，取未跨句
            o = left_obj if right_flag else right_obj
            rst.append({'s': cur, 'p': '描述', 'o': o, 'score': self.get_score(cur, o)})
        rst.extend(strict_rst)
        return rst

    def get_score(self, s, o):
        dis = abs(s['startPosition'] - o['startPosition'])
        score = (self.MAX_DIS - dis) * 1.0 / self.MAX_DIS
        return score

    # 严格的规则方法，准确率高，但是只能召回符合严格条件的关系，召回率低
    # 1. 使用先验关系知识获取关系
    # 2. 使用食品名发掘关系
    def predict_strict(self, text, entity_rst):
        rst = []
        for cur_entity in entity_rst:
            # if cur_entity['type'] not in self.obj_type or cur_entity['text'] not in self.kg_relation:
            #     continue
            for tmp in entity_rst:
                dis = abs(tmp['startPosition'] - cur_entity['startPosition'])
                if dis > self.MAX_DIS or tmp['type'] in self.obj_type:
                    continue
                # 关系出现在离线知识中
                # for key in self.kg_relation[cur_entity['text']]:
                #     if tmp['text'] in self.kg_relation[cur_entity['text']][key]:
                #         rst.append({'s': tmp, 'p': '描述', 'o': cur_entity, 'score': self.get_score(tmp, cur_entity)})
                #         continue
                # 头实体出现在菜名中
                if cur_entity['text'].find(tmp['text']) != -1:
                    rst.append({'s': tmp, 'p': '描述', 'o': cur_entity, 'score': self.get_score(tmp, cur_entity)})
        rst = []
        # same_relation = self.predict_same(entity_rst)
        # rst.extend(same_relation)

        affixes_relation = self.predict_affixes(text, entity_rst)
        rst.extend(affixes_relation)

        # negative_relation = self.predict_negative(text, entity_rst)
        # rst.extend(negative_relation)
        return rst

    def predict_same(self, entity_rst):
        rst = []
        for i in range(len(entity_rst)):
            cur = entity_rst[i]
            if cur['type'] not in self.obj_type:
                continue
            for j in range(i+1, len(entity_rst)):
                if entity_rst[j]['type'] in self.obj_type and entity_rst[j]['text'] == cur['text']:
                    score = self.get_score(entity_rst[j], cur)
                    rst.append({'s': entity_rst[j], 'p': '相同', 'o': cur, 'score': score})
                    break
        return rst

    def predict_affixes(self, text, entity_rst):
        rst = []
        obj = ['食材']
        conjunctions = ['是', '就是', '是由', '由', '也是', '属于', '可以', '适合', '非常', '很', '特别', '选用', '用', '-', '的', ' ', '',
                        '：', ':']
        betweens = ['+', '、']
        obj.extend(self.obj_type)
        for i in range(len(entity_rst)-1):
            inner_text = text[entity_rst[i]['endPosition']: entity_rst[i+1]['startPosition']]
            flag1 = (inner_text in conjunctions)
            # 前缀修饰模式
            flag2 = entity_rst[i]['type'] not in obj and entity_rst[i+1]['type'] in obj
            # 后序谓语模式
            flag3 = entity_rst[i]['type'] in obj and entity_rst[i + 1]['type'] not in obj
            if flag1 and (flag2 or flag3):
                pres = []
                tmp = entity_rst[i]
                j = i - 1
                while j >= 0:
                    between_text = text[entity_rst[j]['endPosition']: tmp['startPosition']]
                    if between_text not in betweens:
                        break
                    pres.append(entity_rst[j])
                    tmp = entity_rst[j]
                    j -= 1
                pres.append(entity_rst[i])

                afters = [entity_rst[i+1]]
                tmp = entity_rst[i+1]
                j = i + 2
                while j < len(entity_rst):
                    between_text = text[tmp['endPosition']: entity_rst[j]['startPosition']]
                    if between_text not in betweens:
                        break
                    afters.append(entity_rst[j])
                    tmp = entity_rst[j]
                    j += 1
                if flag2:
                    all_o = afters
                    all_s = pres
                else:
                    all_o = pres
                    all_s = afters
                for o in all_o:
                    for s in all_s:
                        # if o not in self.obj_type:
                        #     continue
                        rst.append({'s': s, 'p': '描述', 'o': o, 'score': self.get_score(s, o)})
        return rst

    def predict_negative(self, text, entity_rst):
        rst = []
        for i in range(len(entity_rst)-1):
            cur_entity = entity_rst[i]
            if cur_entity['type'] != '否定修饰':
                continue
            after = entity_rst[i+1]
            inner_text = text[cur_entity['endPosition']: after['startPosition']]
            if len(inner_text) < 3:
                rst.append({'s': cur_entity, 'p': '描述', 'o': after, 'score': self.get_score(cur_entity, after)})
        return rst


if __name__ == '__main__':
    # text = '裤子属于灯芯绒的面料吧。很有光泽而且很柔软。背带阔腿裤的设计很遮肉啊腿粗也不怕。上衣是针织的，夹杂了小亮片。'
    text = '很闪的一套 裤子属于灯芯绒的面料吧。很有光泽而且很柔软。背带阔腿裤的设计很遮肉啊腿粗也不怕。裤子是九分裤 矮个子也不怕太长。背带有三个扣子可以调节长度，不怕会扯档。是均码的我穿腰围有些松但不影响。所以胖一点姑娘也没问题。上衣是针织的 夹杂了小亮片。'
    entity_rst = [{'startPosition': 6, 'endPosition': 8, 'type': '品类', 'text': '裤子'}, {'startPosition': 10, 'endPosition': 13, 'type': '材质', 'text': '灯芯绒'}, {'startPosition': 20, 'endPosition': 22, 'type': '功能', 'text': '光泽'}, {'startPosition': 25, 'endPosition': 27, 'type': '功能', 'text': '柔软'}, {'startPosition': 30, 'endPosition': 33, 'type': '裤型', 'text': '阔腿裤'}, {'startPosition': 30, 'endPosition': 33, 'type': '品类', 'text': '阔腿裤'}, {'startPosition': 37, 'endPosition': 39, 'type': '功能', 'text': '遮肉'}, {'startPosition': 46, 'endPosition': 48, 'type': '品类', 'text': '裤子'}, {'startPosition': 49, 'endPosition': 52, 'type': '品类', 'text': '九分裤'}, {'startPosition': 49, 'endPosition': 52, 'type': '裤长', 'text': '九分裤'}, {'startPosition': 53, 'endPosition': 56, 'type': '人群', 'text': '矮个子'}, {'startPosition': 110, 'endPosition': 112, 'type': '品类', 'text': '上衣'}, {'startPosition': 113, 'endPosition': 115, 'type': '材质', 'text': '针织'}, {'startPosition': 121, 'endPosition': 123, 'type': '品类', 'text': '亮片'}, {'startPosition': 121, 'endPosition': 123, 'type': '流行元素', 'text': '亮片'}]

    extractor = FoodRuleExtractor()
    rst = extractor.predict(text, entity_rst)
    print(rst)

