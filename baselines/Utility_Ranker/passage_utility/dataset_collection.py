from torch.utils.data import Dataset
from transformers import BertTokenizer, RobertaTokenizer
import torch
from torch.utils.data import TensorDataset


class SEPairwiseDataset(Dataset):
    def __init__(self, log_data, pretrained_model):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        # BertTokenizer.from_pretrained('bert-base-cased')
        # log_data:[[question, an1, an2, label],...]
        self.data = log_data
        self.max_len = 512

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # first item in the pair
        encoding1 = self.tokenizer.encode_plus(
            # self.data[i][0] + ' [SEP] ' + self.data[i][1],
            self.data[i][0], 
            self.data[i][1],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        # second item in the pair
        encoding2 = self.tokenizer.encode_plus(
            # self.data[i][0] + ' [SEP] ' + self.data[i][2],
            self.data[i][0],
            self.data[i][2],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'question': self.data[i][0],
            'candidate1': self.data[i][1],
            'candidate2': self.data[i][2],
            'input_ids1': encoding1['input_ids'].flatten(),
            'input_ids2': encoding2['input_ids'].flatten(),
            'attention_mask1': encoding1['attention_mask'].flatten(),
            'attention_mask2': encoding2['attention_mask'].flatten(),
            'token_type_ids1': encoding1['token_type_ids'].flatten(),
            'token_type_ids2': encoding2['token_type_ids'].flatten(),
            'targets': torch.tensor(self.data[i][3], dtype=torch.float)
        }


class PosNegDataset(Dataset):
    def __init__(self, qa_pairs, pretrained_model):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        # BertTokenizer.from_pretrained('bert-base-cased')
        # log_data:[[question, an1, an2, label],...]
        self.data = qa_pairs
        self.max_len = 512

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # first item in the pair
        encoding1 = self.tokenizer.encode_plus(
            self.data[i][0][0],
            self.data[i][0][1],
            # self.data[i][0][0] + ' [SEP] ' + self.data[i][0][1],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        # second item in the pair
        encoding2 = self.tokenizer.encode_plus(
            self.data[i][1][0],
            self.data[i][1][1],
            # self.data[i][1][0] + ' [SEP]' + self.data[i][1][1],
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text1': self.data[i][0],
            'text2': self.data[i][1],
            'input_ids1': encoding1['input_ids'].flatten(),
            'input_ids2': encoding2['input_ids'].flatten(),
            'attention_mask1': encoding1['attention_mask'].flatten(),
            'attention_mask2': encoding2['attention_mask'].flatten(),
            'token_type_ids1': encoding1['token_type_ids'].flatten(),
            'token_type_ids2': encoding2['token_type_ids'].flatten(),
            'targets': torch.tensor(1, dtype=torch.float),
            'score1': torch.tensor(self.data[i][2], dtype=torch.float),
            'score2': torch.tensor(self.data[i][3], dtype=torch.float),
            'acc1':  torch.tensor(self.data[i][4], dtype=torch.float),
            'acc2':  torch.tensor(self.data[i][5], dtype=torch.float)
        }


class PosNegSingleDataset(Dataset):
    def __init__(self, log_data, question, pretrained_model):
        # def __init__(self, log_data, pretrained_model, qids: list, aids: list, goldids: dict):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        # self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        # BertTokenizer.from_pretrained('bert-base-cased')
        #log_data [answer1, answer2,...]
        self.data = log_data
        self.question = question
        # self.qids = qids
        # self.aids = aids
        # self.goldids = goldids
        self.max_len = 512

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # first item in the pair
        encoding1 = self.tokenizer.encode_plus(
            self.question[i],
            self.data[i],
            # self.question[i] + ' [SEP] ' + self.data[i],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'question': self.question[i],
            'candidate1': self.data[i],
            'input_ids1': encoding1['input_ids'].flatten(),
            'attention_mask1': encoding1['attention_mask'].flatten(),
            'token_type_ids1': encoding1['token_type_ids'].flatten()
            # 'targets': torch.tensor(self.data[i][3], dtype=torch.float)
        }


class SESingleDataset(Dataset):
    def __init__(self, log_data, question, pretrained_model):
        # def __init__(self, log_data, pretrained_model, qids: list, aids: list, goldids: dict):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        # self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        # BertTokenizer.from_pretrained('bert-base-cased')
        #log_data [answer1, answer2,...]
        self.data = log_data
        self.question = question
        # self.qids = qids
        # self.aids = aids
        # self.goldids = goldids
        self.max_len = 512

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # first item in the pair
        encoding1 = self.tokenizer.encode_plus(
            # self.question[i] + ' [SEP] ' + self.data[i],
            self.question[i],
            self.data[i],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'question': self.question[i],
            'candidate1': self.data[i],
            'input_ids1': encoding1['input_ids'].flatten(),
            'attention_mask1': encoding1['attention_mask'].flatten(),
            'token_type_ids1': encoding1['token_type_ids'].flatten()
            # 'targets': torch.tensor(self.data[i][3], dtype=torch.float)
        }


def create_dataset(tokenizer, data, max_seq_len):
    input_ids,  attention_masks, input_labels = [], [], []
    for text, label in data:
        encode_dict = tokenizer.encode_plus(text[0], text[1],
                                            add_special_tokens=True,
                                            max_length=max_seq_len,
                                            pad_to_max_length=True,
                                            return_attention_mask=True,
                                            return_token_type_ids=False,
                                            return_tensor='pt')
        input_ids.append(encode_dict['input_ids'])
        attention_masks.append(encode_dict['attention_mask'])
        # segment_ids.append(encode_dict['token_type_ids'])
        input_labels.append(label)
    return TensorDataset(input_ids, attention_masks, input_labels)
