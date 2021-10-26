import torch.nn as nn
# from torchcrf import CRF
from structure.entity_extractor.CRF_layer import CRF
from transformers import BertModel
import torch


class BertCrfForNer(nn.Module):
    def __init__(self, config):
        super(BertCrfForNer, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size + config.hello_world, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, lattice=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]  # sequence_output: torch.Size([32, 150, 768])
        sequence_output = self.dropout(sequence_output)
        if lattice is not None:  # torch.Size([32, 150, 2])
            """ todo: 加 dense 层稀疏化 lattice """
            lattice = lattice[:, :sequence_output.shape[1], :]
            sequence_output = torch.cat([sequence_output, lattice], 2)  # torch.Size([32, 150, 770])
        logits = self.classifier(sequence_output)  # logits: torch.Size([32, 81, 11])
        outputs = (logits,)
        if labels is not None:  # labels: torch.Size([32, 150])
            labels = labels[:, :logits.shape[1]]
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask.byte())
            outputs = (-1 * loss,) + outputs
        return outputs
