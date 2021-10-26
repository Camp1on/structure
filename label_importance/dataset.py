import torch
import re
import csv
import os
import random

from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from transformers import BertTokenizer
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import s3fs


def preprocess(line):
    text = re.sub(r'\[(.*?)\]_[\u4e00-\u9fa5]+_', r'[E]\1[/E]', line[1])
    return text


class Dataset(IterableDataset):
    def __init__(self, training, tokenizer, max_len, data_dir, num_examples, batch_size, mode):
        self.training = training
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_dir = data_dir
        self.num_examples = num_examples
        self.batch_size = batch_size

    def __len__(self):
        return self.num_examples

    def process(self, line):
        text = preprocess(line)
        tokenize = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len, return_tensors='pt')
        label = torch.randint_like(torch.where(tokenize['input_ids'] == self.tokenizer.convert_tokens_to_ids('[E]'))[0], 2)
        return tokenize, label
        
    def __iter__(self):
        fs = s3fs.S3FileSystem(False, key='AKIAWFJCI76NJA3NEDEP', secret='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', use_ssl=False)
        return map(self.process, csv.reader(fs.open(self.data_dir, 'r'), delimiter=','))

    @staticmethod
    def collate_fn(batch):
        return (pad_sequence([i[0]['input_ids'][0] for i in batch], True), \
               pad_sequence([i[0]['token_type_ids'][0] for i in batch], True), \
               pad_sequence([i[0]['attention_mask'][0] for i in batch], True)), \
               torch.cat([i[1] for i in batch])

    def input_fn(self):
        train_sampler = None if torch.cuda.device_count() == 0 or issubclass(self.__class__, IterableDataset) else DistributedSampler(self)
        return DataLoader(self, batch_size=self.batch_size, num_workers=1, pin_memory=True, drop_last=False,
                collate_fn=self.collate_fn, sampler=train_sampler)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    tokenizer = BertTokenizer.from_pretrained('/data/dataset/bert_pretrain/bert-base-chinese-torch/vocab.txt')
    special_tokens = ['[E]', '[/E]']
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    dt = Dataset(
            training=True,
            tokenizer=tokenizer,
            max_len=300,
            data_dir='s3://xhs.bravo/user/togo/structure/data/label_importance/train/实体抽取_top15类目_200条_对齐标准_chenzhan1_全部_20210814142511.csv',
            num_examples=200,
            batch_size=2, mode='train').input_fn()

    for i in dt:
        print('input_ids', i[0][0].shape) # 64
        print('token_type_ids', i[0][1].shape) # 64
        print('attention_mask', i[0][2].shape) # 64
        print('label', i[1]) # 64
        break

