import torch
from torch import nn
from transformers import BertModel, RobertaModel, AutoConfig
from torch.nn import ReLU, Sigmoid


class BertRanker(nn.Module):
    def __init__(self, pretrained_model):
        super(BertRanker, self).__init__()
        
        #https://stackoverflow.com/questions/63285197/measuring-uncertainty-using-mc-dropout-on-pytorch
        #configuration = AutoConfig.from_pretrained(pretrained_model)
        #configuration.hidden_dropout_prob = 0.5
        #configuration.attention_probs_dropout_prob = 0.5

        bert = BertModel.from_pretrained(pretrained_model) #, config = configuration)
        

        # bert = BertModel.from_pretrained("roberta-base")
        self.embedding_size = bert.config.hidden_size

        # self.bert = DistilBertModel.from_pretrained("distilbert-base-cased")
        # self.bert = nn.DataParallel(bert, device_ids=[7], output_device=7)
        self.bert = nn.DataParallel(bert)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.pooling = nn.DataParallel(self.pooling)

        # self.out = nn.Linear(bert.config.hidden_size, 1)
        self.W1 = nn.Linear(self.embedding_size, 100)
        self.W1 = nn.DataParallel(self.W1)

        self.W2 = nn.Linear(100, 10)
        self.W2 = nn.DataParallel(self.W2)

        self.out = nn.Linear(10, 1)  # only need one output because we just want a rank score

        self.relu = ReLU()


    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2,
    token_type_ids1,token_type_ids2):
        # persume output [batch, seq, embedding]
        sequence_emb = self.bert(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
            token_type_ids = token_type_ids1
        )[0]

        sequence_emb = sequence_emb.transpose(1, 2)
        pooled_output_1 = self.pooling(sequence_emb)
        pooled_output_1 = pooled_output_1.transpose(2, 1)

        h1_1 = self.relu(self.W1(pooled_output_1))
        h2_1 = self.relu(self.W2(h1_1))
        scores_1 = self.out(h2_1)
        scores_1 = torch.squeeze(scores_1).reshape(1, -1)

        sequence_emb = self.bert(
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            token_type_ids = token_type_ids2
        )[0]
        sequence_emb = sequence_emb.transpose(1, 2)
        pooled_output_2 = self.pooling(sequence_emb)
        pooled_output_2 = pooled_output_2.transpose(2, 1)

        h1_2 = self.relu(self.W1(pooled_output_2))
        h2_2 = self.relu(self.W2(h1_2))
        scores_2 = self.out(h2_2)
        scores_2 = torch.squeeze(scores_2).reshape(1, -1)

        return scores_1, scores_2

    def forward_single_item(self, input_ids, attention_mask, token_type_ids):
        sequence_emb = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids = token_type_ids
        )[0]
        sequence_emb = sequence_emb.transpose(1, 2)
        pooled_output = self.pooling(sequence_emb)
        pooled_output = pooled_output.transpose(2, 1)

        h1 = self.relu(self.W1(pooled_output))
        h2 = self.relu(self.W2(h1))
        scores = self.out(h2)

        return scores, torch.squeeze(pooled_output).detach()