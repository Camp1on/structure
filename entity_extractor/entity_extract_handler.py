import os.path as osp
import sys
import jieba

sys.path.append('../..')

from structure.entity_extractor.rule_extractor import EntityRuleExtractor
from structure.entity_extractor.predict import Predictor
from structure.knowledge import Knowledge
from conf.config import settings
from utils.s3_utils import S3Util
from utils.logger_helper import logger


class EntityExtractHandler:

    def __init__(self):
        """ rule extractor """
        # knowledge_file = S3Util.Instance().get_latest_file(settings['note_structure']['knowledge_file'])
        # self.knowledge = Knowledge(knowledge_file)
        # properties = self.knowledge.get_all_property()
        # self.rule_extractor = EntityRuleExtractor(properties)
        """ model extractor """
        model_path = S3Util.Instance().get_latest_model_path(settings['note_structure']['entity_extract_model_path'])
        model_file = osp.join(model_path, 'model.pt')
        bert_config = osp.join(model_path, 'bert_base.json')
        vocab_file = osp.join(model_path, 'vocab.txt')
        label_index = osp.join(model_path, 'label_index_food.json')
        self.ner_predictor = Predictor(bert_config, vocab_file, model_file, label_index)
        # 属于NER类型
        # self.single_case_type = ['功效', '品牌', '工具', '美食概念词', '适宜人群', '食品']
        self.__warm_up()

    def __warm_up(self):
        # text = '黑色连衣裙'
        text = '当你想吃广东菜的时候/t /t 没有办法,只有自己做,在杭州吃过几家茶餐厅,手撕鸡都很一言难尽,吃货的人生注定是:自己动手丰衣足食。/t 连姜葱汁蘸料都要做到相似, ##SEP## 可惜我们全家就我和我老公买账这个菜,其他人完全get不到这个精华和美味。/t 过两天等我回去还要做五柳炸蛋和猪脚姜,五柳菜和添丁甜醋已经在快递的路上了。'
        jieba.lcut(text)
        # rule_rst = self.rule_extractor.predict(text)
        ner_rst = self.ner_predictor.predict(text)
        # logger.info('EntityExtractHandler rule results of rule_extractor :{} text:{} '.format(rule_rst, text))
        logger.info('EntityExtractHandler ner results of ner_predictor :{} text:{} '.format(ner_rst, text))

    def intersect_extract(self, rule, ner):
        rst = []
        for one in rule:
            for cur in ner:
                if one['type'] == cur['type'] and \
                        one['text'] == cur['text'] and \
                        one['startPosition'] == cur['startPosition']:
                    rst.append(cur)
        return rst

    def postprocess(self, text, entities):
        words = jieba.lcut(text)
        margin = []
        last = -1
        for w in words:
            margin.append(last + 1)
            margin.append(last + len(w) + 1)
            last = last + len(w)
        margin.append(len(text))
        rst = []
        for e in entities:
            if len(e['text']) != 1:
                rst.append(e)
                continue
            start = e['startPosition']
            end = e['endPosition']
            if start in margin and end in margin:
                rst.append(e)
        return rst

    def predict(self, text):
        # sents = re.split('。|！|？|\?|!|\n', text.strip())
        # extract_rst = []
        # for i, sent in enumerate(sents):
        #     rule_ext = self.rule_extractor.predict(sent)
        #     ner_ext = self.ner_predictor.predict(sent)
        #     ner_ext = ner_ext['entities']
        #     cur = self.intersect_extract(rule_ext, ner_ext)
        #     cur = self.select_extract(cate, cur)
        #     extract_rst.append(rule_ext)
        # if len(extract_rst) == 0:
        #     return []
        # extract_rst = [self.clear_extract(one) for one in extract_rst]
        ner_rst = self.ner_predictor.predict(text)
        entities = self.postprocess(text, ner_rst['entities'])
        ner_rst['entities'] = entities
        return ner_rst


if __name__ == '__main__':
    extractor = EntityExtractHandler()
    text = '酸甜油柑 采用本地农户种植的油柑\n \n 挑果洗净放入老式砂缸 加入盐巴搓去涩水\n .\n 再加梅汁白糖手搓入味 搓过后的油柑 入口微涩 回味甘甜 孕妇吃油柑能有效缓解孕吐/便秘'
    print(extractor.predict(text))
