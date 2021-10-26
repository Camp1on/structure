import sys

sys.path.append("..")

import structure.util as util
import torch
from utils.time_util import date_time

from structure.entity_extractor.train import Trainer
from structure.entity_extractor.data import NerDataset
from structure.entity_extractor.model import BertCrfForNer
from transformers import BertTokenizer, AdamW, BertConfig, BertModel
from torch.utils.data import DataLoader
import os.path as osp
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main(train_cfg='config/finetune.json',
         bert_config='config/bert_base.json',
         pretrain_file='bert-base-chinese.bin',
         vocab_file='vocab.txt',
         train_file='dataset/train.txt',
         test_file='dataset/test.txt',
         save_dir='result',
         label_index="label_index.json",
         max_len=150):
    util.set_seeds(42)
    train_config = util.Config.from_json(train_cfg)

    # 构建数据集
    ner_dataset = NerDataset(label_index, max_len)
    tokenizer = BertTokenizer.from_pretrained(vocab_file)
    train_set = ner_dataset.get_dataset(train_file, tokenizer)
    train_data = DataLoader(train_set, batch_size=train_config.batch_size, shuffle=False)
    test_set = ner_dataset.get_dataset(test_file, tokenizer)
    test_data = DataLoader(test_set, batch_size=train_config.batch_size, shuffle=False)

    # 构建模型
    bert_config = BertConfig.from_json_file(bert_config)
    bert_config.num_labels = len(ner_dataset.labels)
    if len(train_set.tensors) == 5:
        bert_config.hello_world = train_set.tensors[4].shape[2]
    else:
        bert_config.hello_world = 0
    model = BertCrfForNer(bert_config)
    model.bert = BertModel.from_pretrained(pretrain_file, config=bert_config)
    # 使用trainer训练
    trainer = Trainer(train_config,
                      util.get_device(),
                      train_data,
                      test_data,
                      AdamW,
                      model,
                      save_dir)
    trainer.train()


if __name__ == '__main__':
    """ 线上配置 """
    train_cfg = "./structure/entity_extractor/config/finetune.json"
    bert_config = "./structure/entity_extractor/config/bert_base.json"
    pretrain_file = "/data/ut/jiale/pretrainedModel/bert_base_chinese_torch_userdefine/bert-base-chinese.bin"
    vocab_file = "/data/ut/jiale/pretrainedModel/bert_base_chinese_torch_userdefine/vocab.txt"
    train_file = "/data/ut/jiale/data4Train/structure/food/20211003/20211003_ner_nodup_dup3038_520_train.txt"
    test_file = "/data/ut/jiale/data4Train/structure/food/20211003/20211003_ner_nodup_dup3038_520_val.txt"
    label_index = "/data/ut/jiale/pretrainedModel/bert_base_chinese_torch_userdefine/label_index_food.json"
    """ 本地配置 """
    # train_cfg = "config/finetune.json"
    # bert_config = "config/bert_base.json"
    # pretrain_file = "/Users/apple/XHSworkspace/data/structure/bert_base_chinese_torch_userdefine/bert-base-chinese.bin"
    # vocab_file = "/Users/apple/XHSworkspace/data/structure/bert_base_chinese_torch_userdefine/vocab.txt"
    # train_file = '/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_rel4_pkl/20211003/20211003_ner_val_no_priority.txt'
    # test_file = '/Users/apple/XHSworkspace/data/structure/food/dataset/train_data/final_rel4_pkl/20211003/20211003_ner_val_no_priority.txt'
    # label_index = "/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/label_index_food.json"
    save_dir = '/data/ut/jiale/models/structure/food'
    out_path = osp.join(save_dir, date_time()) + "/"
    main(train_cfg=train_cfg, bert_config=bert_config,
         pretrain_file=pretrain_file, vocab_file=vocab_file,
         train_file=train_file, test_file=test_file,
         save_dir=out_path, label_index=label_index)
