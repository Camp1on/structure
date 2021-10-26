import torch
import re
import sys

sys.path.append('..')

from structure.util import *
from structure.entity_extractor.data import NerDataset
from structure.entity_extractor.model import BertCrfForNer
from transformers import BertTokenizer, BertConfig
from structure.entity_extractor.data_scripts.segsentence import SegSentence
from utils.s3_utils import S3Util
import os.path as osp
from conf.config import settings


class Predictor:

    def __init__(self, bert_cfg, vocab_file, model_file, label_index):
        self.max_len = 150
        self.ner_dataset = NerDataset(label_index, self.max_len)
        self.device = get_device()
        # self.device = "cpu"
        bert_config = BertConfig.from_json_file(bert_cfg)
        bert_config.num_labels = len(self.ner_dataset.labels)
        bert_config.hello_world = 18
        self.model = BertCrfForNer(bert_config)
        self.model.load_state_dict(torch.load(model_file, map_location=torch.device(self.device)))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(vocab_file)
        special_tokens_dict = {'additional_special_tokens': [' ', "\t"]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.softmax = torch.nn.Softmax(dim=-1)

        model_path = S3Util.Instance().get_latest_model_path(settings['note_structure']['entity_extract_model_path'])
        blacklist_file = osp.join(model_path, 'red_tree_food_concept_v26.json')
        # blacklist_file = "/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/red_tree_food_concept_v28.json"
        self.blacklist = json.load(open(blacklist_file, 'r', encoding='utf-8'))
        self.split_aigo = SegSentence()

    def get_tagging_rst(self, sent, pred_index, input_ids, score_matrix):
        pred_label = [self.ner_dataset.labels_reverse[p] for p in pred_index]
        type = [''] * len(pred_label)
        tagging = ['O'] * len(pred_label)
        for i, p in enumerate(pred_label):
            if p == 'O':
                continue
            splits = p.split('-')
            type[i] = self.ner_dataset.reverse_index[splits[0]]
            tagging[i] = splits[1]
        matchs = re.finditer('BI*', ''.join(tagging))
        if not matchs:
            return {"text": sent,
                    "text_predict": "INFO: none entity found",
                    "entities": []}
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        tokens = [' ' if t == '[UNK]' else t for t in tokens]  # bottleneck: 不支持多条数据同时处理
        tokens = [w.replace("##", "") if "##" in w else w for w in tokens]
        one = []
        textPredict = ""
        pre_end = 0
        score_matrix = torch.nn.functional.softmax(score_matrix, dim=0)
        for match in matchs:
            s, e = match.span()
            sent_start = len(''.join(tokens[:s]))
            text = ''.join(tokens[s:e])
            # score = 1
            # for score_index in range(s, e):
            #     score *= score_matrix[score_index][pred_index[score_index]]
            score = 0
            for score_index in range(s, e):
                score += score_matrix[score_index][pred_index[score_index]]
            score /= e - s + 1
            token_type = type[s:e][0]
            # if token_type in ["品牌", "工具", "适宜人群"] and text not in self.blacklist[token_type]:
            #     continue
            if token_type in ["食品", "食材", "美食概念词"] and text not in self.blacklist[token_type] and len(text) <= 2:
                continue
            # if token_type in ["食品", "食材", "品牌"]:
            #     continue
            textPredict += ''.join(tokens[pre_end:s]) + "[" + text + "]_" + token_type + "_"
            pre_end = e
            one.append({'type': type[s],
                        'text': text,
                        'score': int(score.item() * 100),
                        'startPosition': sent_start,
                        'endPosition': sent_start + len(text)})
        textPredict += ''.join(tokens[pre_end:])
        return {"text": sent,
                "textPredict": textPredict.replace("[UNK]", " "),
                "entities": one}
        # if len(sent) == len(tokens):
        #     return {"text": sent,
        #             "textPredict": textPredict.replace("[UNK]", " "),
        #             "entities": one}
        # else:
        #     return {"text": sent,
        #             "textPredict": "ERROR: ori_sent not equal with bert_tokens",
        #             "entities": []}

    def predict(self, document):
        SEP_TAG = "\t"
        document = document.replace("\r", " ")
        document = document.replace("\t", " ")
        document = document.replace("\n", "\t")
        text_split = document.split("\t")

        # SEP_TAG = " ##SEP## "
        # document = document.replace(" ##sep## ", "\t")
        # document = document.replace(" ##SEP## ", "\t")
        # document = document.split("***")[0]
        # text_split = document.split("\t")

        # SEP_TAG = ""
        # document = document.replace("/t", "\t")
        # document = document.replace(" ##sep## ", " ")
        # document = document.replace(" ##SEP## ", " ")
        # text_split = self.split_aigo.cut(document)
        # text_split = [w.replace("\t", " ") for w in text_split]

        rst = []
        for text in text_split:
            input_ids, segment_ids, input_mask, lattice_label = \
                self.ner_dataset.inf_process_text(text, self.tokenizer, self.max_len)
            input_ids = input_ids.to(self.device)
            segment_ids = segment_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            if lattice_label is not None:
                lattice_label = lattice_label.to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask, None,
                                    lattice_label)  # logits[0]: torch.Size([1, 14, 11])
                tags = self.model.crf.decode(logits[0],
                                             input_mask.byte())  # tags: 14 [[0, 0, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 0]] #
            preds = tags[0][1:-1]  # 去除首位标志位: [CLS], [SEP]
            input_ids = input_ids[0].cpu().numpy()[1:-1]  # 去除首位标志位: [CLS], [SEP]
            score_matrix = logits[0].squeeze(0)[1:-1, :]
            rst.append(self.get_tagging_rst(text, preds, input_ids, score_matrix))
        text_concat = ""
        textPredict_concat = ""
        entities_concat = []
        pre_len = 0
        for idx, rs in enumerate(rst):
            text_concat += rs.get("text") + SEP_TAG if idx != len(rst) - 1 else rs.get("text")
            textPredict_concat += rs.get("textPredict") + SEP_TAG if idx != len(rst) - 1 else rs.get("textPredict")
            for entity in rs.get("entities"):
                entities_concat.append({
                    'type': entity.get("type"),
                    'text': entity.get("text"),
                    'score': entity.get("score"),
                    'startPosition': entity.get("startPosition") + pre_len,
                    'endPosition': entity.get("endPosition") + pre_len
                })
            pre_len += len(rs.get("text")) + len(SEP_TAG)
        return {"text": text_concat,
                "textPredict": textPredict_concat,
                "entities": entities_concat}


if __name__ == '__main__':
    # model_file = '/data/model/note_structure/models/20210803/model.pt'
    # bert_config = '/data/model/note_structure/models/20210803/bert_base.json'
    # vocab_file = '/data/model/note_structure/models/20210803/vocab.txt'
    # label_index = '/data/model/note_structure/models/20210803/label_index.json'
    model_file = '/Users/apple/XHSworkspace/data/structure/food/models/20210915221540/model_steps_500.pt'
    bert_config = '/Users/apple/XHSworkspace/data/structure/food/models/bert_base.json'
    vocab_file = '/Users/apple/XHSworkspace/data/structure/food/models/vocab.txt'
    label_index = '/Users/apple/XHSworkspace/data/structure/food/models/label_index_food.json'
    max_len = 150
    p = Predictor(bert_config, vocab_file, model_file, label_index)
    sents = [
        "1\n手撕鸡都很一言难尽\n手撕鸡都很一言难尽"
    ]
    for sent in sents:
        print(sent)
        print(p.predict(sent))
        print()
