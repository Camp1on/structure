import util
import torch

from train import Trainer
from model import RelationClassifier
from dataset import RelationDataset
from transformers import BertTokenizer, BertConfig, AdamW, BertModel
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(train_cfg='config/finetune.json',
         bert_cfg='config/bert_base.json',
         train_file='dataset/train.txt',
         test_file='dataset/test.txt',
         pretrain_file='chinese_base/bert-base-chinese.bin',
         vocab_file='chinese_base/vocab.txt',
         save_dir='result',
         max_len=200):
    util.set_seeds(42)
    train_config = util.Config.from_json(train_cfg)

    # 构建数据集
    tokenizer = BertTokenizer.from_pretrained(vocab_file)
    special_tokens_dict = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]', ' ', '[/工艺]', '[工艺]',
                                                         '[/功效]', '[功效]', '[/食品]', '[食品]', '[/食材]', '[食材]', '[/口味]',
                                                         '[口味]', '[/工具]', '[工具]', '[/适宜人群]', '[适宜人群]', '[/品牌]', '[品牌]',
                                                         '[/美食概念词]', '[美食概念词]', '[/否定修饰]', '[否定修饰]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    train_set = RelationDataset.get_dataset(train_file, tokenizer, max_len)
    train_data = DataLoader(train_set, batch_size=train_config.batch_size, shuffle=True)
    test_set = RelationDataset.get_dataset(test_file, tokenizer, max_len)
    test_data = DataLoader(test_set, batch_size=train_config.batch_size, shuffle=False)

    # 构建模型
    bert_config = BertConfig.from_json_file(bert_cfg)
    model = RelationClassifier(bert_config, len(RelationDataset.labels), tokenizer)
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
    # import os.path as osp
    # import sys
    # sys.path.append('../..')
    # from conf.config import settings
    # from utils.s3_utils import S3Util
    #
    # data_sir = S3Util.Instance().get_latest_model_path(settings['note_structure']['relation_extract_dataset'])
    # train_file = osp.join(data_sir, 'train.txt')
    # test_file = osp.join(data_sir, 'test.txt')
    # pretrain_file = S3Util.Instance().get_latest_file(settings['note_structure']['pretrain_bert'])
    #
    # model_path = S3Util.Instance().get_latest_model_path(settings['note_structure']['relation_extract_model_path'])
    # bert_cfg = osp.join(model_path, 'bert_base.json')
    # vocab_file = osp.join(model_path, 'relation_vocab.txt')
    # save_dir = '/data/ut/chenzhan1/relation_extract/result/'
    bert_cfg = 'config/bert_base.json'
    train_file = '/Users/user/my-project/note_strcuture/relation_extract/label_data_new/test_v1.txt'
    test_file = '/Users/user/my-project/note_strcuture/relation_extract/label_data_new/test_v1.txt'
    pretrain_file = '/data/model/bert_pretrain/bert-base-chinese-torch/bert-base-chinese.bin'
    vocab_file = '/Users/user/my-project/note_strcuture/relation_extract/data/relation_vocab.txt'
    save_dir = '/data/model_train/relation/'
    main(bert_cfg=bert_cfg,
         train_file=train_file,
         test_file=test_file,
         pretrain_file=pretrain_file,
         vocab_file=vocab_file,
         save_dir=save_dir)
