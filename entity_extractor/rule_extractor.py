import sys
import jieba
sys.path.append('..')

from collections import defaultdict
from structure.util import Singleton
from structure.entity_extractor.trie import AhoCorasickAutoMation


@Singleton
class EntityRuleExtractor:

    def __init__(self, properties):
        self.properties = properties
        self.trie = self.construct_trie(properties)
        self.reverse_index = self.reverse(properties)

    def construct_trie(self, properties):
        words = []
        for p in properties:
            words.extend(properties[p])
        words = list(set(words))
        words.sort()
        return AhoCorasickAutoMation(words, words, False)

    def reverse(self, properties):
        reverse_index = defaultdict(set)
        for key in properties:
            for v in properties[key]:
                reverse_index[v].add(key)
        return reverse_index

    def postproecss(self, rst):
        has_only_cate_entities = False
        for i, cur in enumerate(rst):
            # if cur['type'] != '品类':  # 时尚美妆
            if cur['type'] != '食品':  # 美食
                continue
            flag = True
            for j, t in enumerate(rst):
                if j == i:
                    continue
                if cur['startPosition'] == t['startPosition'] and cur['endPosition'] == t['endPosition']:
                    flag = False
            if flag:
                has_only_cate_entities = True
                break
        length = len(rst)
        for i in range(length)[::-1]:
            for j in range(i):
                if rst[i]['startPosition'] != rst[j]['startPosition'] or rst[i]['endPosition'] != rst[j]['endPosition']:
                    continue
                # if (rst[i]['type'] != '品类' and rst[j]['type'] != '品类'):  # 时尚美妆
                if rst[i]['type'] != '食品' and rst[j]['type'] != '食品':  # 美食
                    rst.pop(i)
                    break
                if has_only_cate_entities:
                    rst[j]['type'] = rst[i]['type']
                rst.pop(i)
                break
        return rst

    def predict(self, text):
        rst = []
        words = jieba.lcut(text)
        margin = []
        last = -1
        for w in words:
            margin.append(last + 1)
            margin.append(last + len(w))
            last = last + len(w)
        margin.append(len(text))
        for (s, e) in self.trie.match(text):
            entity = text[s:e]
            if s not in margin or e not in margin:
                continue
            if entity.encode('utf-8').isalpha() and ((s > 0 and text[s - 1].encode('utf-8').isalpha()) or (
                    e < len(text) - 2 and text[e - 1].encode('utf-8').isalpha())):
                continue
            for key in self.reverse_index[entity]:
                tmp = {'startPosition': s, 'endPosition': e, 'type': key, 'text': entity, 'score': 100}
                rst.append(tmp)
        return self.postproecss(rst)


if __name__ == '__main__':
    properties = {'品类': ['裤子'],
                  '菜名': ['灯芯绒']}
    extractor = EntityRuleExtractor(properties)
    text = '很闪的一套 裤子裤子属于灯芯绒的面料吧。很有光泽而且很柔软。背带阔腿裤的设计很遮肉啊腿粗也不怕。裤子是九分裤 矮个子也不怕太长。背带有三个扣子可以调节长度，不怕会扯档。是均码的我穿腰围有些松但不影响。所以胖一点姑娘也没问题。上衣是针织的 夹杂了小亮片。'
    print(extractor.predict(text))
