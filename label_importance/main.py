import argparse
import os

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from online_triplet_loss.losses import *

from commons.custom_layers.custom_metrics_torch import metric_output_selector
from commons.model_utils.model_builder_torch import ModelConfig, ModelBuilder
from dataset import Dataset
from model import LabelImportanceClassifier
from transformers import BertTokenizer, BertConfig


class MyBuilder(ModelBuilder):

    def build_train_data_iterator(self):
        train_ds = Dataset(
            training=True,
            tokenizer=self.tokenizer,
            max_len=self.config.max_len,
            data_dir=self.config.train_data_dir,
            num_examples=self.config.num_examples,
            batch_size=self.config.batch_size, mode='train').input_fn()
        return train_ds

    def build_val_data_iterator(self):
        validation_ds = Dataset(
            training=False,
            tokenizer=self.tokenizer,
            max_len=self.config.max_len,
            data_dir=self.config.val_data_dir,
            num_examples=self.config.num_examples,
            batch_size=self.config.batch_size, mode='val').input_fn()
        return validation_ds

    def build_infer_data_iterator(self):
        infer_ds = Dataset(
            training=False,
            tokenizer=self.tokenizer,
            max_len=self.config.max_len,
            data_dir=self.config.infer_data_dir,
            num_examples=self.config.num_examples,
            batch_size=self.config.batch_size, mode=self.config.infer_dataset.split('#')[0]).input_fn()
        return infer_ds

    def build_model_fn(self):
        bert_config = BertConfig.from_json_file('config/bert_base.json')
        bert_config.vocab_size += len(self.special_tokens)
        return LabelImportanceClassifier(bert_config, 2, self.tokenizer)

    def build_learning_rate_fn(self, optimizer):
        return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.config.epochs or 100)

    def build_loss_fn(self):
        def loss(y_true, y_pred):
            loss = torch.nn.CrossEntropyLoss()(y_pred, y_true)
            return {'ces': loss}

        return loss

    def build_metrics_fn(self):
        #return {'acc': metric_output_selector(torchmetrics.Accuracy(), f_y_true=None, f_y_pred=lambda x: F.softmax(x))}
        #return {'top1acc': Top1Accuracy(), 'top5acc': Top5Accuracy()}
        return {}

    def reserved_fuction(self):
        os.system('aws s3 cp %s %s'%(self.config.vocab_file, self.config.local_file))
        self.tokenizer = BertTokenizer.from_pretrained(self.config.local_file)
        self.special_tokens = ['[E]']
        self.tokenizer.add_special_tokens({'additional_special_tokens': self.special_tokens})



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='mode')
    FLAGS, unparsed = parser.parse_known_args()

    config = ModelConfig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'), FLAGS.mode)
    world_size = max(1, len(str(config.gpu_list).split(',')))

    if FLAGS.mode == 'train':
        mp.spawn(MyBuilder(config).train, args=(config.init_ckpt), nprocs=world_size, join=True)
        #MyBuilder(config).train(0, config.init_ckpt)
    if FLAGS.mode == 'val':
        mp.spawn(MyBuilder(config).eval, args=(config.init_ckpt), nprocs=world_size, join=True)
    if FLAGS.mode == 'save':
        MyBuilder(config).export(config.best_ckpt)
    if FLAGS.mode == 'infer':
        mp.spawn(MyBuilder(config).savedmodel_predict, 
                args=(config.savedmodel_dir, 
                    config.infer_dataset.split('#')[1],
                    config.infer_dataset.split('#')[2],
                    config.infer_dataset.split('#')[3]), nprocs=world_size, join=True)
        os.system(f'cat {config.result_dir}.* > {config.result_dir} && rm {config.result_dir}.*')
