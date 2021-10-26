import torch
import torch.nn as nn

from transformers import BertModel


class LabelImportanceClassifier(nn.Module):

    def __init__(self, config, num_labels, tokenizer):
        super().__init__()
        self.endpoints = {}
        self.e_id = tokenizer.convert_tokens_to_ids('[E]')
        #self.e2_id = tokenizer.convert_tokens_to_ids('[E2]')
        self.bert = BertModel(config)
        self.fc = nn.Linear(config.dim*2, config.dim*2)
        self.activate = nn.Tanh()
        self.drop = nn.Dropout(config.p_drop_hidden)
        self.classifier = nn.Linear(config.dim*2, num_labels)

    def forward(self, features):
        input_ids, token_type_ids, attention_mask = features[0], features[1], features[2]
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        hidden_states = outputs['last_hidden_state']
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[2]).squeeze()

        flat_input = input_ids.reshape(-1, input_ids.shape[0]*input_ids.shape[1]).squeeze()
        indices = (flat_input == self.e_id).nonzero(as_tuple=True)[0]
        e = torch.index_select(hidden_states, dim=0, index=indices)

        pooled_h = torch.cat([e, e], dim=-1)
        pooled_h = self.activate(self.fc(pooled_h))
        logits = self.classifier(self.drop(pooled_h))
        self.endpoints['logits'] = logits
        return logits, self.endpoints


if __name__ == '__main__1':
    import torch
    from transformers import BertTokenizer, BertConfig
    vocab_file = '/data/dataset/bert_pretrain/bert-base-chinese-torch/vocab.txt'
    bert_cfg = 'config/bert_base.json'
    bert_config = BertConfig.from_json_file(bert_cfg)
    special_tokens = ['[E]']
    bert_config.vocab_size += len(special_tokens)
    tokenizer = BertTokenizer.from_pretrained(vocab_file)
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    model = LabelImportanceClassifier(bert_config, 5, tokenizer)

    inputs = tokenizer(
            ['clarks[E]其乐[/E]魔术贴[E]休闲[/E][E]凉鞋[/E]。巨舒服，踩 脚感，很好搭，配[E]长裙[/E]都好看，在家一直穿着走一点也不磨，可能是今[E]夏[/E]最爱的[E]凉鞋[/E]了。bloomer 巨[E]褪色[/E]，我的[E]白裤子[/E]。 [E]西西[/E]和 自打流行百慕大，再也买不 到这种长度的[E]西装[/E] 了', 'clarks[E]其乐[/E]魔术贴[E]休闲[/E][E]凉鞋[/E]。巨舒服，踩 脚感，很好搭，配[E]长裙[/E]都好看，在家一直穿着走一点也不磨，可能是今[E]夏[/E]最爱的[E]凉鞋[/E]了。bloomer 巨[E]褪色[/E]，我的[E]白裤子[/E]。 [E]西西[/E]和 自打流行百慕大，再也买不 到这种长度的[E]西装[/E] 了'],
            padding=True,
            truncation=True,
            max_length=300,
            return_tensors='pt'  # 返回的类型为pytorch tensor
            )
    input_ids, token_type_ids, attention_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']
    print('input_ids.shape', input_ids.shape)
    result = model(input_ids, token_type_ids, attention_mask)
