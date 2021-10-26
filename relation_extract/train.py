import os
import torch
import torch.nn as nn

from tqdm import tqdm
from loss import FocalLoss
from sklearn.metrics import confusion_matrix
from transformers import get_linear_schedule_with_warmup


class Trainer(object):

    def __init__(self, config, device, dataset, test_data, optimizer, model, save_dir):
        self.config = config
        self.model = model
        self.save_dir = save_dir
        self.device = device
        self.dataset = dataset
        self.test_data = test_data
        self.optimizer = optimizer(model.parameters(), lr=self.config.lr)

    def train(self, data_parallel=False):
        self.model.train()
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)
        total_steps = len(self.dataset) * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        global_step = 0
        # criterion = nn.CrossEntropyLoss()
        criterion = FocalLoss(2.0, 0.5)
        for e in range(self.config.epochs):
            loss_sum = 0.
            iter_bar = tqdm(self.dataset, desc='Iter (loss=X.XXX)')
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]
                logits = model(input_ids=batch[0],
                               token_type_ids=batch[1],
                               attention_mask=batch[2])
                loss = criterion(logits, batch[3])
                loss_sum += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()

                iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())
                global_step += 1
                if global_step % self.config.save_steps == 0:
                    self.save(global_step)
                    p, r, f1 = self.eval(self.test_data)
                    print(p, r, f1)
                if total_steps and total_steps < global_step:
                    print('Epoch %d/%d : Average Loss %5.3f' % (e + 1, self.config.epochs, loss_sum / (i + 1)))
                    print('The Total Steps have been reached.')
                    self.save(global_step)
                    return

            print('Epoch %d/%d : Average Loss %5.3f' % (e + 1, self.config.epochs, loss_sum / (i + 1)))
        self.save(global_step)

    def save(self, i):
        """ save current model """
        torch.save(self.model.state_dict(),
                   os.path.join(self.save_dir, 'model_steps_' + str(i) + '.pt'))

    def load(self, model_file):
        self.model.load_state_dict(torch.load(model_file, map_location=torch.device(self.device)))

    def eval(self, test_set, data_parallel=True):
        """ Evaluation Loop """
        self.model.eval()
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)
        results = []
        label_preds = []
        label_ids = []
        iter_bar = tqdm(test_set, desc='Iter (loss=X.XXX)')
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad():
                input_ids, segment_ids, input_mask, label_id = batch
                logits = model(input_ids, segment_ids, input_mask)
                _, label_pred = logits.max(1)
                result = (label_pred == label_id).float()
                accuracy = result.mean()
                label_preds.append(label_pred)
                label_ids.append(label_id)
            results.append(result)
            iter_bar.set_description('Iter(acc=%5.3f)' % accuracy)
        pred = torch.cat(label_preds).cpu().numpy()
        label = torch.cat(label_ids).cpu().numpy()
        matrix = confusion_matrix(label, pred)
        p = matrix[1][1]*1.0 / (matrix[1][1] + matrix[0][1])
        r = matrix[1][1]*1.0 / (matrix[1][1] + matrix[1][0])
        f1 = 2.0 * p*r / (p + r)
        return p, r, f1
