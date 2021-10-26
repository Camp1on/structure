import torch
import torch.nn as nn

from transformers import BertModel


class RelationClassifier(nn.Module):

    def __init__(self, config, num_labels, tokenizer):
        super().__init__()
        self.e1_id = tokenizer.convert_tokens_to_ids('[E1]')
        self.e2_id = tokenizer.convert_tokens_to_ids('[E2]')
        self.bert = BertModel(config)
        self.fc = nn.Linear(config.dim*2, config.dim*2)
        self.activate = nn.Tanh()
        self.drop = nn.Dropout(config.p_drop_hidden)
        self.classifier = nn.Linear(config.dim*2, num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        hidden_states = outputs['last_hidden_state']
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[2]).squeeze()
        flat_input = input_ids.reshape(-1, input_ids.shape[0]*input_ids.shape[1]).squeeze()
        indices = (flat_input == self.e1_id).nonzero(as_tuple=True)[0]
        e1 = torch.index_select(hidden_states, dim=0, index=indices)
        indices = (flat_input == self.e2_id).nonzero(as_tuple=True)[0]
        e2 = torch.index_select(hidden_states, dim=0, index=indices)
        assert e1.shape[0] == e2.shape[0] == input_ids.shape[0]

        pooled_h = torch.cat([e1, e2], dim=-1)
        pooled_h = self.activate(self.fc(pooled_h))
        logits = self.classifier(self.drop(pooled_h))
        return logits


if __name__ == '__main__':
    import torch
    t = torch.FloatTensor([[1, 2, 3, 4], [5, 6, 1, 2]])
    x = torch.randn(22, 4, 10)
    tt = torch.eq(t, 1)
    print(tt)
