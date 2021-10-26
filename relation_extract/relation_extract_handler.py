import sys
import re
import os.path as osp
sys.path.append('../..')
print(sys.path)
from conf.config import settings
from utils.s3_utils import S3Util
from utils.logger_helper import logger
from structure.relation_extract.rule_extractor import RelationRuleExtractor
from structure.relation_extract.food_rule_extractor import FoodRuleExtractor
from structure.relation_extract.predict import RelationPredictor


def clear_duplicate(src):
    rst = []
    for s in src:
        if s not in rst:
            rst.append(s)
    return rst


class RelationExtractHandler:

    def __init__(self):
        self.relationRuleExtractor = RelationRuleExtractor()
        self.ruleExtractMap = {'美食': FoodRuleExtractor()}

        model_path = S3Util.Instance().get_latest_model_path(settings['note_structure']['relation_extract_model_path'])
        model_file = osp.join(model_path, 'model.pt')
        bert_config = osp.join(model_path, 'bert_base.json')
        vocab_file = osp.join(model_path, 'relation_vocab.txt')
        model_file = 'model_steps_9000.pt'
        self.relationPredictor = RelationPredictor(bert_config, vocab_file, model_file)
        self.MAX_DIS = 150
        self.batch_size = 64
        self.score_threshold = 0.6
        self.constrain = ['食材-描述-食品', '品牌-描述-食品', '口味-描述-食品', '美食概念词-描述-食品', '工艺-描述-食品', '功效-描述-食品', '适宜人群-描述-食品',
                          '口味-描述-食材', '品牌-描述-食材', '功效-描述-食材', '品牌-描述-美食概念词', '口味-描述-美食概念词', '美食概念词-描述-美食概念词',
                          '适宜人群-描述-美食概念词', '功效-描述-美食概念词', '食材-描述-美食概念词', '否定修饰-描述-口味', '否定修饰-描述-功效']
        self.__warm_up()

    def __warm_up(self):
        text = '点了份[E1][口味]麻辣味[/口味][/E1]的[E2][食品]小龙虾烤鱼[/食品][/E2]，里面的小龙虾很多'
        rst = self.relationPredictor.predict(text)
        logger.info('RelationExtractHandler relation results of model extract :{} text:{} '.format(rst, text))

    def clear_text(self, text):
        text = re.sub('##SEP##', '', text)
        text = re.sub('##sep##', '', text)
        splits = re.split('。|！|？|\?|!|\n|/t', text)
        i = 0
        while i < len(splits):
            sent = splits[i]
            if sent.find('[E1]') != -1 or sent.find('[E2]') != -1:
                break
            i += 1
        j = len(splits) - 1
        while j > 0:
            sent = splits[j]
            if sent.find('[E1]') != -1 or sent.find('[E2]') != -1:
                break
            j -= 1
        sents = splits[i: j+1]
        text = '。'.join(sents)
        return text

    def add_markers(self, text, s, o):
        s_start = s['startPosition']
        s_end = s['endPosition']
        o_start = o['startPosition']
        o_end = o['endPosition']
        prefix = '[E1]' + '[' + s['type'] + ']'
        suffix = '[/' + s['type'] + ']' + '[/E1]'
        cur_text = text[:s_start] + prefix + text[s_start: s_end] + suffix + text[s_end:]
        if o_start >= s_end:
            o_start += len(prefix) + len(suffix)
            o_end += len(prefix) + len(suffix)
        prefix = '[E2]' + '[' + o['type'] + ']'
        suffix = '[/' + o['type'] + ']' + '[/E2]'
        cur_text = cur_text[:o_start] + prefix + cur_text[o_start: o_end] + suffix + cur_text[o_end:]
        return cur_text

    def get_candidate(self, entities):
        candidate = []
        length = len(entities)
        for i in range(length):
            for j in range(length):
                if i == j or abs(entities[i]['startPosition'] - entities[j]['startPosition']) > self.MAX_DIS:
                    continue
                key = entities[i]['type'] + '-' + '描述' + '-' + entities[j]['type']
                if key in self.constrain:
                    candidate.append({'s': entities[i], 'p': '描述', 'o': entities[j]})
        return candidate

    def prune_candidate(self, candidate):
        tmp = sorted(candidate, key=lambda d: abs(d['s']['startPosition']-d['o']['startPosition']))
        num = min(self.batch_size, len(candidate))
        return tmp[: num]

    def text_verify(self, text):
        index1 = text.find('[E1]')
        index2 = text.find('[/E1]')
        index3 = text.find('[E2]')
        index4 = text.find('[/E2]')
        if index1 == -1 or index2 == -1 or index3 == -1 or index4 == -1:
            return False
        return True

    def get_batch_text(self, text, cand_relation):
        rst = []
        useless = []
        for i, spo in enumerate(cand_relation):
            cur_text = self.add_markers(text, spo['s'], spo['o'])
            cur_text = self.clear_text(cur_text)
            if len(cur_text) > 180 or not self.text_verify(cur_text):
                useless.append(i)
            rst.append(cur_text)
        for i in useless[::-1]:
            rst.pop(i)
            cand_relation.pop(i)
        return rst

    def predict(self, domain, text, entity_rst):
        entity_rst = sorted(entity_rst, key=lambda d: d['startPosition'])
        rst = []
        cand_relation = self.get_candidate(entity_rst)
        cand_relation = self.prune_candidate(cand_relation)
        batch_text = self.get_batch_text(text, cand_relation)
        if len(batch_text) == 0:
            return []
        score, label_pred = self.relationPredictor.predict_batch(batch_text)
        if len(score) != len(cand_relation) or len(label_pred) != len(cand_relation):
            logger.info('RelationExtractHandler the length of predict results is error, text:{} '.format(text))
            return []
        for i, label in enumerate(label_pred):
            if label != 1 or score[i] < self.score_threshold:
                continue
            cur = cand_relation[i]
            cur['score'] = score[i]
            rst.append(cur)
        # rule_extractor = self.ruleExtractMap[domain]
        # rule_rst = rule_extractor.predict_affixes(text, entity_rst)
        # rst.extend(rule_rst)
        return clear_duplicate(rst)


if __name__ == '__main__':
    pass

