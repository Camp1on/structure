import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from utils.logger_helper import logger
from utils.file_util import delete_dir, ensure_dir
from utils import send_text_to_user
import functools

send_text = functools.partial(send_text_to_user, user='jialeyang@xiaohongshu.com')


class Trainer(object):

    def __init__(self, config, device, dataset_train, dataset_test, optimizer, model, save_dir):
        self.config = config
        self.model = model
        self.save_dir = save_dir
        ensure_dir(self.save_dir)
        self.device = device
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        # self.optimizer = optimizer(self.model.parameters(), lr=self.config.lr)
        bert_params = list(map(id, model.bert.parameters()))  # 返回的是parameters的 内存地址
        top_params = filter(lambda p: id(p) not in bert_params, model.parameters())
        self.optimizer = optimizer([
            {'params': model.bert.parameters(), 'lr': self.config.lr},
            {'params': top_params, 'lr': self.config.lr * 15}])

    def train(self, data_parallel=True):
        self.model.train()
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)
        total_steps = len(self.dataset_train) * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0.1 * total_steps,
                                                    num_training_steps=total_steps)  # https://cloud.tencent.com/developer/article/1833108
        global_step = 0
        best_acc_score = 0.0
        best_loss = float("inf")
        for e in range(self.config.epochs):
            loss_sum = 0.
            iter_bar = tqdm(self.dataset_train, desc='Iter (loss=X.XXX)')  # TODO： dataset size=1000 ？
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]
                if len(batch) == 5:
                    loss, logits = model(input_ids=batch[0],
                                         token_type_ids=batch[1],
                                         attention_mask=batch[2],
                                         labels=batch[3],
                                         lattice=batch[4])
                else:
                    loss, logits = model(input_ids=batch[0],
                                         token_type_ids=batch[1],
                                         attention_mask=batch[2],
                                         labels=batch[3])
                print(loss)
                loss_sum += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()
                model.zero_grad()
                iter_bar.set_description('Iter (loss=%5.3f)' % (loss.item() / len(batch)))  # 当前 batch 的损失
                global_step += 1
                if global_step % self.config.save_steps == 0:
                    test_accuracy, test_loss = self.eval(self.dataset_test)
                    msg = 'Epoch: {}/{}, Train Loss: {}, Test Loss: {}, Test Acc: {}, ' \
                        .format(e, i, loss.item() / len(batch), test_loss, test_accuracy)
                    logger.info(msg)
                    send_text(msg)
                    if test_accuracy > best_acc_score and test_loss < best_loss:
                        best_acc_score = test_accuracy
                        best_loss = test_loss
                        msg = 'At epoch {}/{} Model with best test acc score {} and test loss {}, saved at {}'.format(
                            e, i, best_acc_score, test_loss, self.save_dir)
                        logger.info(msg)
                        send_text(msg)
                        delete_dir(osp.dirname(self.save_dir))
                        ensure_dir(self.save_dir)
                        self.save(global_step)
            msg = 'Epoch {} : Average Train Loss {}'.format(e, loss_sum / len(self.dataset_train))
            logger.info(msg)  # 所有已跑 batch 的平均损失
            send_text(msg)

    def save(self, i):
        """ save current model """
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_steps_' + str(i) + '.pt'))

    def load(self, model_file):
        self.model.load_state_dict(torch.load(model_file, map_location=legent(self.device)))

    def eval(self, test_set, data_parallel=True):
        """ Evaluation Loop
        """
        self.model.eval()  # 不再动态更新参数，该用训练好的值
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)
        res_acc = []
        res_loss = []
        iter_bar = tqdm(test_set, desc='Iter (loss=X.XXX) (loss=X.XXX)')
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad():  # 禁用梯度计算的上下文管理器
                batch = [t.to(self.device) for t in batch]
                if len(batch) == 5:
                    loss, logits = model(input_ids=batch[0],
                                         token_type_ids=batch[1],
                                         attention_mask=batch[2],
                                         labels=batch[3],
                                         lattice=batch[4])
                else:
                    loss, logits = model(input_ids=batch[0],
                                         token_type_ids=batch[1],
                                         attention_mask=batch[2],
                                         labels=batch[3])
                tags = self.model.crf.decode(logits, batch[2].byte())  # logits：[32, 81, 11] ；attention_mask：[32, 81]
                label_pred = [w[1:-1] for w in tags]
                acc_sum = []
                acc_count = []
                for pred, true in zip(label_pred, batch[3].tolist()):
                    true_id = true[1:len(pred) + 1]
                    acc = sum(np.array(pred) == np.array(true_id))
                    acc_sum.append(acc)
                    acc_count.append(len(pred))
                res_acc.append(sum(acc_sum) / sum(acc_count) if sum(acc_count) > 0 else 0)
                res_loss.append(loss.item() / len(batch[0]) if len(batch[0]) > 0 else 0)
        return sum(res_acc) / len(res_acc), sum(res_loss) / len(res_loss)
