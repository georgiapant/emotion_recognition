import torch
import torch.nn as nn
from src.config import BATCH_SIZE, MAX_LEN, BERT_MODEL, num_labels, bidirectional
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F


class bertNrcVadClassifier(nn.Module):
    """
    Bert Model for Classification Tasks.
    """

    def __init__(self, freeze_bert=False):
        """
        @param    robert: a RobertaModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(bertNrcVadClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H_lstm, H, D_out = 768, 200, 384, num_labels

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(BERT_MODEL, problem_type="multi_label_classification")#, torchscript=True)

        if bidirectional == True:
            self.LSTM = nn.LSTM(D_in, H_lstm, num_layers=1, bidirectional=True, batch_first=True)
            self.lstm_fc = nn.Linear(2 * H_lstm, D_in)
        else:
            self.LSTM = nn.LSTM(D_in, H_lstm, num_layers=1, bidirectional=False, batch_first=True)
            self.lstm_fc = nn.Linear(H_lstm, D_in)

        self.emo_freq = nn.Linear(10, D_in)
        self.vad_linear = nn.Linear(3, D_in)
        self.fc = nn.Linear(D_in, H)
        self.label = nn.Linear(H, D_out)
        self.dropout = nn.Dropout(0.3)

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def attention_net(self, input_matrix, final_output_cls):
        hidden = final_output_cls
        attn_weights = torch.bmm(input_matrix, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(input_matrix.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def forward(self, input_ids, attention_mask, nrc_feats, vad_vec):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states = True).hidden_states[-1]
        last_hidden_state_cls = bert_out[:, 0, :]
        # bert_out, _ = self.LSTM(bert_out)
        # bert_out = self.lstm_fc(bert_out)

        nrc = F.relu(self.emo_freq(nrc_feats))
        vad = F.relu(self.vad_linear(vad_vec))
        combine = torch.cat((bert_out,nrc,vad), dim=1)

        # combine = torch.cat((bert_out,nrc), dim=1)
        output= self.attention_net(combine, last_hidden_state_cls)
        # output, _ = self.LSTM(output)
        output = F.relu(self.fc(output))
        output = self.dropout(output)
        logits = self.label(output)

        return logits
