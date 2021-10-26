import re

from collections import defaultdict


# 使用近邻的思想进行关系抽取
# 1、同一行保留最先出现的品类，其它忽略（todo 可尝试同一行的近邻方法）
# 2、以品类实体对所有句子进行切分，第i个品类词到第i+1个品类词之前的属性都是描述第i个品类ci
class RelationRuleExtractor:

    def __init__(self):
        self.PINLEI = '品类'
        self.p = {'修饰': 1, '否定': 2, '搭配': 3, '相同': 4}

    def has_cate(self, extract_rst):
        for one in extract_rst:
            if one['type'] == self.PINLEI:
                return True
        return False

    def get_group(self, extract_rst):
        rst = defaultdict(list)
        tmp = []
        last = -1
        for i, one in enumerate(extract_rst):
            flag = self.has_cate(extract_rst[i])
            if not flag and len(rst) == 0:
                tmp.append(one)
                continue
            if flag:
                rst[i].append(one)
                last = i
                continue
            rst[last].append(one)
        # 无品类词则认为整个笔记描述同一个未知品类，返回所有属性即可
        if len(rst) == 0:
            return [extract_rst]
        first = min(rst.keys())
        rst[first].extend(tmp)
        rst = [rst[row] for row in rst]
        return rst

    def split_sent(self, text, entity_rst):
        sents = re.split('。|！|？|\?|!|\n', text.strip())
        margins = [-1]
        last = -1
        for i, sent in enumerate(sents):
            margins.append(last+1+len(sent))
            last += len(sent) + 1
        rst = [[] for _ in range(len(sents))]
        for e in entity_rst:
            start = e['startPosition']
            end = e['endPosition']
            for i in range(len(margins)-1):
                if margins[i] <= start <= margins[i+1] and margins[i] <= end <= margins[i+1]:
                    rst[i].append(e)
                    break
        return sents, rst

    # 处理一个entity句子抽取的结果, 解决实体抽取的部分问题问题，
    # 同一个句子中，对于同一个entity，既有属性又有'品类'，则删除属性
    # 同一个句子中有多个'品类'则保留句子中的最前面一个
    # todo  1、使用上下位进行组织关系；2、主体词识别保留重要主体；3、实体抽取效果OK后可删除此逻辑
    def clear_extract(self, extract_rst):
        rst = []
        for i, cur in enumerate(extract_rst):
            flag = False
            if cur['type'] == self.PINLEI:
                for one in extract_rst[i + 1:]:
                    if one['text'] == cur['text']:
                        flag = True
                        break
            if not flag:
                rst.append(cur)
        first_cate = -1
        for i, one in enumerate(rst):
            if one['type'] == self.PINLEI:
                first_cate = i
                break
        length = len(rst)
        for i in range(first_cate + 1, length)[::-1]:
            if rst[i]['type'] == self.PINLEI:
                rst.pop(i)
        return rst

    # 返回结果类型 spo
    def predict(self, text, entity_rst):
        sents, entity_rst = self.split_sent(text, entity_rst)
        entity_rst = [self.clear_extract(e) for e in entity_rst]
        assert len(sents) == len(entity_rst)
        sent_group = self.get_group(entity_rst)
        rst = []
        for group in sent_group:
            cur_object = None
            cur_subjects = []
            for sent_rst in group:
                for one in sent_rst:
                    if one['type'] == self.PINLEI:
                        cur_object = one
                        continue
                    cur_subjects.append(one)
            if cur_object is None or len(cur_subjects) == 0:
                continue
            for s in cur_subjects:
                rst.append({'s': s, 'p': self.p['修饰'], 'o': cur_object, 'score': 1.0})
        return rst

    def predict_rigorous(self, text, entity_rst):
        rst = []
        for i in range(len(entity_rst)-1):
            inner_text = text[entity_rst[i]['endPosition']: entity_rst[i+1]['startPosition']]

            # 前缀修饰模式
            adjuncts = ['的', ' ', '']
            flag1 = inner_text in adjuncts
            flag2 = entity_rst[i]['type'] != '品类' and entity_rst[i+1]['type'] == '品类'
            if flag1 and flag2:
                s = entity_rst[i]
                o = entity_rst[i+1]
                rst.append({'s': s, 'p': self.p['修饰'], 'o': o, 'score': 1.0})
                j = i - 1
                after = entity_rst[i]
                while j >= 0:
                    if entity_rst[j]['type'] == '品类':
                        break
                    if entity_rst[j]['endPosition'] == after['startPosition']:
                        rst.append({'s': entity_rst[j], 'p': self.p['修饰'], 'o': o, 'score': 1.0})
                        after = entity_rst[j]
                    j -= 1

            # 后序谓语模式
            flag3 = entity_rst[i]['type'] == '品类' and entity_rst[i+1]['type'] != '品类'
            subjects = ['是', '就是', '是由', '由', '也是', '属于', '可以', '：', '适合', '非常', '很', '特别', '选用', '用', '-']
            flag4 = inner_text in subjects
            if flag3 and flag4:
                o = entity_rst[i]
                s = entity_rst[i+1]
                rst.append({'s': s, 'p': self.p['修饰'], 'o': o, 'score': 1.0})
                j = i + 2
                pre = s
                while j < len(entity_rst):
                    if entity_rst[j]['type'] == '品类':
                        break
                    if entity_rst[j]['startPosition'] == pre['endPosition']:
                        rst.append({'s': entity_rst[j], 'p': self.p['修饰'], 'o': o, 'score': 1.0})
                        pre = entity_rst[j]
                    j += 1
        return rst


if __name__ == '__main__':
    # text = '裤子属于灯芯绒的面料吧。很有光泽而且很柔软。背带阔腿裤的设计很遮肉啊腿粗也不怕。上衣是针织的，夹杂了小亮片。'
    text = '很闪的一套 裤子属于灯芯绒的面料吧。很有光泽而且很柔软。背带阔腿裤的设计很遮肉啊腿粗也不怕。裤子是九分裤 矮个子也不怕太长。背带有三个扣子可以调节长度，不怕会扯档。是均码的我穿腰围有些松但不影响。所以胖一点姑娘也没问题。上衣是针织的 夹杂了小亮片。'
    entity_rst = [{'startPosition': 6, 'endPosition': 8, 'type': '品类', 'text': '裤子'}, {'startPosition': 10, 'endPosition': 13, 'type': '材质', 'text': '灯芯绒'}, {'startPosition': 20, 'endPosition': 22, 'type': '功能', 'text': '光泽'}, {'startPosition': 25, 'endPosition': 27, 'type': '功能', 'text': '柔软'}, {'startPosition': 30, 'endPosition': 33, 'type': '裤型', 'text': '阔腿裤'}, {'startPosition': 30, 'endPosition': 33, 'type': '品类', 'text': '阔腿裤'}, {'startPosition': 37, 'endPosition': 39, 'type': '功能', 'text': '遮肉'}, {'startPosition': 46, 'endPosition': 48, 'type': '品类', 'text': '裤子'}, {'startPosition': 49, 'endPosition': 52, 'type': '品类', 'text': '九分裤'}, {'startPosition': 49, 'endPosition': 52, 'type': '裤长', 'text': '九分裤'}, {'startPosition': 53, 'endPosition': 56, 'type': '人群', 'text': '矮个子'}, {'startPosition': 110, 'endPosition': 112, 'type': '品类', 'text': '上衣'}, {'startPosition': 113, 'endPosition': 115, 'type': '材质', 'text': '针织'}, {'startPosition': 121, 'endPosition': 123, 'type': '品类', 'text': '亮片'}, {'startPosition': 121, 'endPosition': 123, 'type': '流行元素', 'text': '亮片'}]
    # text = '番茄炒蛋。1.番茄切块备用，鸡蛋加少许盐打散。2.锅中放油，倒入鸡蛋，炒散盛出。3.锅中放油，倒入蒜末爆香，倒入番茄。4.加少许白糖，' \
    #        '一勺生抽，翻炒出汤汁。5.倒入鸡蛋，翻炒均匀盛出。火腿土豆片。1.土豆、火腿切片备用（切好的土豆用水浸泡）。2.锅中放油，倒入蒜末爆香，' \
    #        '倒入火腿、土豆翻炒3分钟。3.加一勺盐，少许白糖，一勺生抽，一勺蚝油，翻炒均匀。4.倒入青椒，再翻炒2分钟盛出#上班族带饭[话题]#'
    # entity_rst = []
    extractor = RelationRuleExtractor()
    rst = extractor.predict(text, entity_rst)
    print(rst)

