import torch
import sys
sys.path.append('../..')

from utils.logger_helper import logger
from structure.util import *
from structure.relation_extract.dataset import RelationDataset
from structure.relation_extract.model import RelationClassifier
from sklearn.metrics import confusion_matrix
from transformers import BertConfig, BertTokenizer


class RelationPredictor:

    def __init__(self, bert_cfg, vocab_file, model_file):
        self.max_len = 200
        self.device = get_device()
        bert_config = BertConfig.from_json_file(bert_cfg)
        self.tokenizer = BertTokenizer.from_pretrained(vocab_file)
        special_tokens_dict = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]', ' ', '[/工艺]', '[工艺]',
                                                             '[/功效]', '[功效]', '[/食品]', '[食品]', '[/食材]', '[食材]', '[/口味]',
                                                             '[口味]', '[/工具]', '[工具]', '[/适宜人群]', '[适宜人群]', '[/品牌]',
                                                             '[品牌]', '[/美食概念词]', '[美食概念词]', '[/否定修饰]', '[否定修饰]']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model = RelationClassifier(bert_config, len(RelationDataset.labels), self.tokenizer)
        self.model.load_state_dict(torch.load(model_file, map_location=torch.device(self.device)))
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        score, label_pred = self.predict_batch([text])
        return score[0], label_pred[0]

    def predict_batch(self, batch):
        input_ids, segment_ids, input_mask = RelationDataset.process_text(batch, self.tokenizer, self.max_len)
        input_ids = input_ids.to(self.device)
        segment_ids = segment_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask)
            logits = torch.softmax(logits, dim=1)
            score, label_pred = logits.max(1)
            score, label_pred = score.cpu().numpy(), label_pred.cpu().numpy()
            return score, label_pred


if __name__ == '__main__':
    model_path = '/data/model/note_structure/models/relation_extract/20210925/'
    model_file = model_path + 'model.pt'
    bert_config = model_path + 'bert_base.json'
    vocab_file = model_path + 'relation_vocab.txt'
    relationPredictor = RelationPredictor(bert_config, vocab_file, model_file)
    file = '/Users/user/my-project/note_strcuture/relation_extract/label_data_new/test_v0.txt'
    fin = open(file, 'r', encoding='utf-8')
    reals = []
    preds = []
    cnt = 0
    lines = fin.readlines()
    batch_size = 32
    total_steps = len(lines) // batch_size
    score_cnt = [0, 0, 0, 0, 0]
    margin = [0.6, 0.7, 0.8, 0.9, 1.1]
    fout = open('pred_rst.txt', 'w', encoding='utf-8')
    for i in range(total_steps-1):
        batch_real = []
        batch_text = []
        for line in lines[batch_size*i: batch_size*(i+1)]:
            splits = line.strip().split('\t\t', 1)
            if len(splits) != 2:
                continue
            batch_real.append(int(splits[0]))
            batch_text.append(splits[1])
        batch_score, batch_pred = relationPredictor.predict_batch(batch_text)
        for j in range(len(batch_score)):
            if batch_score[j] < 0.7 and batch_pred[j] == 1:
                batch_pred[j] = 0
            for k, m in enumerate(margin):
                # if batch_score[j] < m and batch_pred[j] == 1 and batch_real[j] == 0:
                if batch_score[j] < m and batch_pred[j] != batch_real[j]:
                    score_cnt[k] += 1
                    break
        reals.extend(batch_real)
        preds.extend(batch_pred)
        for j in range(len(batch_score)):
            fout.write(str(batch_real[j]) + '\t' + str(batch_pred[j]) + '\t' + str(batch_score[j]) + '\t' + batch_text[j] + '\n')
            # if batch_score[j] > 0.75 and batch_real[j] != batch_pred[j]:
            #     cnt += 1
            #     print(str(batch_real[j]) + '\t' + str(batch_pred[j]) + '\t' + str(round(batch_score[j], 3)) + '\t' + batch_text[j])
    matrix = confusion_matrix(reals, preds)
    p = matrix[1][1]*1.0 / (matrix[1][1] + matrix[0][1])
    r = matrix[1][1]*1.0 / (matrix[1][1] + matrix[1][0])
    f1 = 2.0*p*r / (p + r)
    print(p)
    print(r)
    print(f1)
    print(score_cnt)


