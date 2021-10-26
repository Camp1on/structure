import json
import re
import sys
import time

sys.path.append('..')

from utils.s3_utils import S3Util
from utils.logger_helper import logger
from conf.config import settings
from structure.knowledge import Knowledge
from structure.entity_extractor.rule_extractor import EntityRuleExtractor
from structure.relation_extract.relation_extract_handler import RelationExtractHandler
from structure.entity_extractor.entity_extract_handler import EntityExtractHandler


class NoteStructureServing:

    def __init__(self):
        knowledge_file = S3Util.Instance().get_latest_file(settings['note_structure']['knowledge_file'])
        self.knowledge = Knowledge(knowledge_file)
        properties = self.knowledge.get_all_property()
        # 处理嵌套问题是要删除 "食品"
        self.food_names = set()
        if '食品' in properties:
            self.food_names = set(properties['食品'])
            properties.pop('食品')
        self.rule_extractor = EntityRuleExtractor(properties)

        self.entityExtractHandler = EntityExtractHandler()
        self.relationExtractHandler = RelationExtractHandler()
        self.support_domain = ['美食']
        self.__warmup()

    def __warmup(self):
        cate = '美食'
        text = '点了份麻辣小龙虾烤鱼，里面的小龙虾很多'
        rst = self.predict(cate, text)
        logger.info('NoteStructure results of cate :{} text:{} : {}'.format(cate, text, rst))

    def ner_process(self, ner_rst, domain):
        rst = []
        tmp = set()
        for one in ner_rst:
            key = one['type']
            if key == '否定修饰':
                continue
            value = one['text']
            score = one['score']
            if key + '-' + value not in tmp:
                rst.append({'key': key, 'value': value, 'score': score, 'domain': domain})
                tmp.add(key + '-' + value)
        return rst

    def relation_process(self, relation_rst, domain):
        rst = []
        tmp = set()
        for one in relation_rst:
            score = one['score']
            key = one['s']['type'] + '\t\t' + one['o']['type']
            value = one['s']['text'] + '\t\t' + one['o']['text']
            if key + '-' + value not in tmp:
                tmp.add(key + '-' + value)
                rst.append({'key': key, 'value': value, 'score': score, 'domain': domain})
        return rst

    def nest_process(self, ner_rst):
        entities = []
        for one in ner_rst:
            if one['type'] != '食品' or len(one['text']) <= 2:
                continue
            food_name = one['text']
            ext = self.rule_extractor.predict(food_name)
            if len(ext) == 0 or (len(ext) == 1 and ext[0]['text'] == one['text']):
                continue
            reg = [e['text'] for e in ext]
            reg = '|'.join(reg)
            splits = re.split(reg, food_name)
            for s in splits:
                if s == '' or s not in self.food_names:
                    continue
                start = food_name.find(s)
                if start == -1:
                    continue
                ext.append({'text': s, 'type': '食品', 'startPosition': start, 'endPosition': start + len(s) + 1, 'score': 100})
            for e in ext:
                e['startPosition'] = e['startPosition'] + one['startPosition']
                e['endPosition'] = e['endPosition'] + one['startPosition']
                entities.append(e)
            # print('nest ner:\t' + food_name + ' ——> ' + '；'.join([e['text'] + '-' + e['type'] for e in ext]))
        return entities

    def neg_word_process(self, entities, relation):
        del_entity = []
        del_relation = []
        for i, r in enumerate(relation):
            s = r['s']
            o = r['o']
            if s['type'] != '否定修饰':
                continue
            index_s = entities.index(s)
            index_o = entities.index(o)
            if index_s == -1 or index_o == -1:
                continue
            entities[index_o]['text'] = entities[index_s]['text'] + entities[index_o]['text']
            entities[index_o]['startPosition'] = min(entities[index_o]['startPosition'], entities[index_s]['startPosition'])
            entities[index_o]['endPosition'] = max(entities[index_o]['endPosition'], entities[index_s]['endPosition'])
            del_entity.append(index_s)
            del_relation.append(i)
        for i in del_relation[::-1]:
            relation.pop(i)
        for i in del_entity[::-1]:
            entities.pop(i)

    def predict(self, domain, text):
        if domain not in self.support_domain:
            return []
        ner_ext = self.entityExtractHandler.predict(text)
        entities = ner_ext['entities']
        relation_rst = self.relationExtractHandler.predict(domain, text, entities)
        self.neg_word_process(entities, relation_rst)
        nest_entities = self.nest_process(entities)
        ner_ext['entities'].extend(nest_entities)
        final_ner = self.ner_process(entities, domain)
        final_relation = self.relation_process(relation_rst, domain)
        final_ner.extend(final_relation)
        logger.info('NoteStructure results of cate :{} text:{} : {}'.format(domain, text, final_ner))
        return final_ner

    def predict_debug(self, domain, text):
        if domain not in self.support_domain:
            return []
        ner_ext = self.entityExtractHandler.predict(text)
        entities = ner_ext['entities']
        relation_rst = self.relationExtractHandler.predict(domain, text, entities)
        self.neg_word_process(entities, relation_rst)
        nest_entities = self.nest_process(entities)
        ner_ext['entities'].extend(nest_entities)
        ner = [one['text'] + '-' + one['type'] + '-' + str(one['startPosition']) + '-' + str(one['endPosition']) for one
               in ner_ext['entities']]
        relation = []
        for one in relation_rst:
            s_str = one['s']['text'] + '-' + one['s']['type'] + '-' + str(one['s']['startPosition']) + '-' + str(one['s']['endPosition'])
            o_str = one['o']['text'] + '-' + one['o']['type'] + '-' + str(one['o']['startPosition']) + '-' + str(one['o']['endPosition'])
            relation.append(s_str + '->' + one['p'] + '->' + o_str)
        return ner, relation


if __name__ == '__main__':
    structure = NoteStructureServing()
    cate = '美食'
    fin = open('/Users/user/my-project/note_strcuture/relation_extract/data/gpu_test.txt', 'r', encoding='utf-8')
    for line in fin:
        text = re.sub('/t', '\n', line)
        text = re.sub('##SEP##', '', text)
        text = re.sub('##sep##', '', text)
        rst = structure.predict(cate, text)
