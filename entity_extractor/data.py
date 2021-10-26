import copy

import torch
import json

from torch.utils.data import TensorDataset
from utils.logger_helper import logger
import re
from utils.s3_utils import S3Util
import os.path as osp
from conf.config import settings

BIO_PATTERN_ALL = re.compile(r'(?=\[(.+?)\](_(.+?)_))')  # 带有捕获组的正向超前断言 且 最短匹配
BIO_PATTERN = re.compile(r'\[(?!UNK)(.+?)\](_[A-Z]{3}_)')
BIO_BEGIN = re.compile(r'_[A-Z]{3}_\[')
BIO_BEGIN_ALL = re.compile(r'_(.+?)_\[')
BIO_END = re.compile(r'\]_[A-Z]{3}_')
BIO_END_ALL = re.compile(r'\]_(.+?)_')


class NerDataset:

    def __init__(self, label_index_file, max_length=150):
        self.max_length = max_length
        self.labels_map = json.load(open(label_index_file, 'r', encoding='utf-8'))
        self.labels, self.reverse_index = NerDataset.get_labels(label_index_file)
        self.labels_reverse = {value: key for (key, value) in self.labels.items()}
        self.labels_map_reverse = {value: key for (key, value) in self.labels_map.items()}

        model_path = S3Util.Instance().get_latest_model_path(settings['note_structure']['entity_extract_model_path'])
        lattice_file = osp.join(model_path, 'word_lattice_key_food_20211017_clean.json')
        self.lattice = json.load(open(lattice_file, 'r', encoding='utf-8'))
        # self.lattice = json.load(open("/Users/apple/XHSworkspace/data/structure/food/models/word_lattice_key_food_20211017_clean.json", 'r', encoding='utf-8'))
        # self.lattice = None

    @staticmethod
    def get_labels(file):
        labels = {'O': 0}
        index = json.load(open(file, 'r', encoding='utf-8'))
        reverse_index = {value: key for (key, value) in index.items()}
        i = 1
        for key in index:
            label = index[key]
            labels[label + '-B'] = i
            i += 1
            labels[label + '-I'] = i
            i += 1
        return labels, reverse_index

    def _parse_data(self, file_name):
        """ line format: 采用[拼接]_流行元素_设计，更具时尚[户外感]_适用场景_
        """
        logger.info(f'pre process file {file_name}')
        char_text = []
        char_label = []
        text = []
        with open(file_name, 'r', encoding='utf-8') as reader:
            tmp = reader.read()
            lines = tmp.split("\n")
            for line in lines:
                data_pair = NerDataset.pre_process(line.strip(), self.labels_map)
                char_str = [w[0] for w in data_pair]
                char_text.append(char_str)
                char_label.append([w[1] for w in data_pair])
                text.append("".join(char_str))
        return char_text, char_label, text

    def _parse_data_from_json(self, file_name):
        """ line format:
                "user": "黄佳妮",
                "text": "打卡北京最火烤鸭店:四季民福/t /t 运气爆棚正巧轮到二楼靠窗景观位/t 看着落日斜阳下的东华门/t 吃着最有范儿的京味儿/t 惬意......",
                "ner": [
                    {
                        "startPosition": 10,
                        "endPosition": 14,
                        "type": "品牌",
                        "text": "四季民福"
                    },
                    {
                        "startPosition": 59,
                        "endPosition": 62,
                        "type": "美食概念词",
                        "text": "京味儿"
                    }
                ]
        """
        logger.info(f'pre process file {file_name}')
        char_text = []
        char_label = []
        text = []
        with open(file_name, 'r', encoding='utf-8') as reader:
            tmp = reader.read()
            lines = tmp.split("\n")
            for line in lines:
                data_pair = NerDataset.pre_process(line.strip(), self.labels_map)
                char_str = [w[0] for w in data_pair]
                char_text.append(char_str)
                char_label.append([w[1] for w in data_pair])
                text.append("".join(char_str))
        return char_text, char_label, text

    @staticmethod
    def pre_process(text, labels_map):
        tag_res = BIO_PATTERN_ALL.findall(text)
        tag_res_filter = []  # 解决 UNK 问题
        for idx, val in enumerate(tag_res):
            if "[" in val[0] or "]" in val[0]:
                idx += 1
            else:
                tag_res_filter.append(val)
        for i in set([w[1] for w in tag_res_filter]):
            text = text.replace(i, "_TAG_")
        tag_idx = 0
        text = BIO_PATTERN.sub(r'\2[\1]\2', text)
        i = 0
        res = []
        label = ''
        is_begin = False
        is_inner = False
        length = len(text)
        while i < length:
            c = text[i]
            if '_' == c and i + 5 < length and BIO_BEGIN.fullmatch(text, i, i + 6):
                label = labels_map.get(tag_res_filter[tag_idx][2], -1)
                i += 6
                is_begin = True
                is_inner = True
                continue
            if ']' == c and i + 5 < length and BIO_END.fullmatch(text, i, i + 6) and is_inner:
                i += 6
                is_inner = False
                tag_idx += 1
                continue
            if is_begin:
                res.append([c, f'{label}-B'])
                is_begin = False
            elif is_inner:
                res.append([c, f'{label}-I'])
            else:
                res.append([c, 'O'])
            i += 1
        return res

    def char_label_to_bert(self, bert_token, char_text, char_label):
        label = []
        lattice_label = []
        for bt, ct, cl in zip(bert_token, char_text, char_label):
            cur_label = [self.labels['O']]  # 补 O 对齐 [CLS]
            bt = bt[1:]  # 跳过 [CLS]
            if self.lattice is not None:
                cur_lattice = [[0] * len(self.lattice["0"])]
            # 双指针
            i, j = 0, 0
            while i < len(bt) and j < len(ct):
                if self.lattice is not None:
                    cur_lattice.append(self.lattice.get(bt[i], [0] * len(self.lattice["0"])))
                if bt[i] == ct[j] or bt[i] == "[UNK]":
                    cur_label.append(self.labels[cl[j]])
                    i += 1
                    j += 1
                else:
                    tmp = ""
                    j_copy = copy.deepcopy(j)
                    while bt[i].lstrip("#") != tmp and j < len(ct):
                        try:
                            tmp += ct[j]
                        except:
                            print(char_text)
                        j += 1
                    if len(cl[j_copy:j]) > 0:
                        cur_label.append(self.labels[cl[j_copy:j][0]])
                    else:
                        logger.info(f'wrong parse char_label_to_bert: {bert_token}')
                    i += 1
            cur_label += [0] * self.max_length
            cur_label = cur_label[:self.max_length]  # 截断至 max_length
            label.append(cur_label)
            if self.lattice is not None:
                cur_lattice.extend([[0] * len(self.lattice["0"])] * self.max_length)
                cur_lattice = cur_lattice[:self.max_length]
                lattice_label.append(cur_lattice)
            else:
                lattice_label = None
        return label, lattice_label

    def test(self, file, tokenizer):
        """ line format: {"text": "一举一动都透露出一丝浪漫风情", "label": "O O O O O O O O O O 13-B 13-I 13-I O"}
        """
        res = []
        res_label = []
        with open(file, 'r') as reader:
            tmp = reader.read()
            lines = tmp.split("\n")
            for line in lines:
                one = json.loads(line.strip())
                text_tmp = one['text']
                if "[UNK]" not in text_tmp:
                    label_tmp = one['label'].split(" ")
                    input_ids, _, _ = NerDataset.process_text(text_tmp, tokenizer, self.max_length)
                    bert_token = tokenizer.convert_ids_to_tokens(input_ids[0])
                    bert_token = bert_token[1:-1]
                    bert_token = [w.replace("[UNK]", "牛") for w in bert_token]
                    bert_token = [w.replace("#", "") for w in bert_token]
                    res.append(self.post_process(bert_token, label_tmp))

                    cur_label = [self.labels['O']]
                    for l in label_tmp:
                        cur_label.append(self.labels[l])
                    cur_label += [0] * self.max_length
                    cur_label = cur_label[:self.max_length]
                    res_label.append(cur_label)

        return res, res_label

    def post_process(self, text, predict):
        res = []
        end_idx = len(text) - 1
        for i in range(end_idx):
            token = text[i]
            label1 = predict[i]
            label2 = predict[i + 1]
            if 'O' == label1:
                res.append(token)
            else:
                if label1.endswith('-B'):
                    res.append('[')
                res.append(token)
                if not label2.endswith('-I'):
                    res.append(f']_{self.labels_map_reverse[label1[:-2]]}_')
        token = text[end_idx]
        label1 = predict[end_idx]
        if label1.endswith('-B'):
            res.append('[')
        res.append(token)
        if 'O' != label1:
            res.append(f']_{self.labels_map_reverse[label1[:-2]]}_')
        return ''.join(res)

    def load_data(self, file):
        text = []
        label = []
        with open(file, 'r') as fin:
            i = 0
            for line in fin:
                i += 2
                if i == 1000:
                    break
                one = json.loads(line.strip())
                text.append(one['text'])
                cur_label = [
                    self.labels['O']]  # 补 O 原因：'tommy潮流品牌' => ['[CLS]', 'tom', '##my', '潮', '流', '品', '牌', '[SEP]']
                for l in one['label'].split(' '):
                    cur_label.append(self.labels[l])
                cur_label += [0] * self.max_length
                cur_label = cur_label[:self.max_length]  # 截断至 max_length
                label.append(cur_label)
        return text, label

    @staticmethod
    def process_text(text, tokenizer, max_len):
        tokenize = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors='pt'  # 返回的类型为pytorch tensor
        )
        return tokenize['input_ids'], tokenize['token_type_ids'], tokenize['attention_mask']

    def inf_process_text(self, text, tokenizer, max_len):
        tokenize = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors='pt'  # 返回的类型为pytorch tensor
        )
        if self.lattice is None:
            return tokenize['input_ids'], tokenize['token_type_ids'], tokenize['attention_mask'], None
        bert_token = [tokenizer.convert_ids_to_tokens(w) for w in tokenize['input_ids']]
        lattice_label = self.inf_lattice_label(bert_token)
        return tokenize['input_ids'], tokenize['token_type_ids'], tokenize['attention_mask'], torch.tensor(
            lattice_label)

    def inf_lattice_label(self, bert_token):
        lattice_label = []
        for bt in bert_token:
            # max_length = len(bt)
            bt = bt[1:]  # 跳过 [CLS]
            cur_lattice = [[0] * len(self.lattice["0"])]
            # 双指针
            i = 0
            while i < len(bt):
                cur_lattice.append(self.lattice.get(bt[i], [0] * len(self.lattice["0"])))
                i += 1
            # cur_lattice.extend([[0, 0]] * max_length)
            # cur_lattice = cur_lattice[:max_length]
            lattice_label.append(cur_lattice)
        return lattice_label

    def checking_label(self, labels, bert_token):
        labels = labels.numpy()
        for label, bt in zip(labels, bert_token):
            i, j = -1, 0
            while j < len(label):
                if i < 0:
                    i += 1
                    j += 1
                if label[i] == 0 and label[j] != 0 and label[j] % 2 == 0:
                    print(label[i:])
                    print(bt[i:])
                    break
                else:
                    i += 1
                    j += 1

    def get_dataset(self, file, tokenizer):
        # test_text, test_label = self.test("/Users/apple/XHSworkspace/data/structure/models/dataset/test.txt", tokenizer)
        # text, label = self.load_data(file)
        char_text, char_label, text = self._parse_data(file)
        special_tokens_dict = {'additional_special_tokens': [' ', '\t']}
        tokenizer.add_special_tokens(special_tokens_dict)
        input_ids, token_type_ids, attention_mask = NerDataset.process_text(text, tokenizer, self.max_length)
        bert_token = [tokenizer.convert_ids_to_tokens(w) for w in input_ids]
        """ 融入词表信息 """
        label, lattice_label = self.char_label_to_bert(bert_token, char_text, char_label)
        print(type(attention_mask))
        labels = torch.tensor(label)  # labels: torch.Size([499, 150])
        self.checking_label(labels, bert_token)
        if self.lattice is not None:
            lattice_labels = torch.tensor(lattice_label)
            return TensorDataset(input_ids, token_type_ids, attention_mask, labels,
                                 lattice_labels)  # input_ids: torch.Size([499, 82])
        else:
            return TensorDataset(input_ids, token_type_ids, attention_mask, labels)


if __name__ == '__main__':
    d = NerDataset('dataset/label_index.json')
    d.load_data('dataset/train.txt')
