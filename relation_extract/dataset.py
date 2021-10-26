import torch

from torch.utils.data import TensorDataset


class RelationDataset:
    labels = {0: '无关系', 1: '描述', 2: '否定', 3: '相同', 4: '对比'}
    @staticmethod
    def load_data(file):
        text = []
        label = []
        with open(file, 'r', encoding='utf-8') as fin:
            for line in fin:
                splits = line.strip().split('\t\t')
                if len(splits) != 2:
                    continue
                text.append(splits[1])
                label.append(int(splits[0]))
        return text, label

    @staticmethod
    def process_text(text, tokenizer, max_len):
        tokenize = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors='pt'  # 返回的类型为pytorch tensor
        )
        return tokenize['input_ids'], tokenize['token_type_ids'], tokenize['attention_mask']

    @staticmethod
    def get_dataset(file, tokenizer, max_len):
        text, label = RelationDataset.load_data(file)
        input_ids, token_type_ids, attention_mask = RelationDataset.process_text(text, tokenizer, max_len)
        labels = torch.tensor(label)
        return TensorDataset(input_ids, token_type_ids, attention_mask, labels)


if __name__ == '__main__':
    from transformers import BertTokenizer
    file = '/Users/user/my-project/note_strcuture/relation_extract/label_data/57_test.txt'
    vocab_file = 'config/relation_vocab.txt'
    # vocab_file = '/data/dataset/bert_pretrain/bert-base-chinese-torch/vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(vocab_file)
    special_tokens_dict = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    ttt = tokenizer('[E1]冰川[/E1]蓝拆箱。2020最火[E2]冰蓝色[/E2]。一只不可错的颜色', return_tensors='pt')
    test = RelationDataset.get_dataset(file, tokenizer, 300)
    print(test)