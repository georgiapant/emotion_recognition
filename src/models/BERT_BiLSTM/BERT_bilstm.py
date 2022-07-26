import torch.nn as nn
from src.config import BERT_MODEL, num_labels
from transformers import BertModel
import torch


class BertClassifier(nn.Module):
    """
    Bert Model for Classification Tasks.
    """

    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, num_labels

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(BERT_MODEL, problem_type="multi_label_classification", output_hidden_states=True)

        self.LSTM = nn.LSTM(4*D_in, 512, num_layers=1, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2 * 512, D_out)


        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(2*512, 512),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, D_out)
        )

        # self.fc = nn.Linear(512, 128)
        # self.fc2 = nn.Linear(128, 32)
        # self.fc3 = nn.Linear(32, self.n_classes)
        # self.bert_drop = nn.Dropout(0.3)
        # self.classifier = nn.Sequential(nn.Linear(self.bert.config.hidden_size, D_out))

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        # outputs = self.bert(input_ids=input_ids,
        #                     attention_mask=attention_mask)
        # print(outputs)

        # Extract the last hidden state of the token `[CLS]` for classification task
        # last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        # logits = self.classifier(last_hidden_state_cls)
        # logits = outputs.logits

        # outputs = self.bert(input_ids=input_ids,
        #                     attention_mask=attention_mask)["pooler_output"]

        hidden_states = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)["hidden_states"]

        embedding_output = hidden_states[0]
        attention_hidden_states = hidden_states[1:]

        pooled_output = torch.cat(tuple([attention_hidden_states[i] for i in [-1, -2, -3, -4]]), dim=-1) # concat the last four hidden layers
        # print(pooled_output.shape)
        lstm, _ = self.LSTM(pooled_output)
        # print(lstm.shape)
        fc_input = torch.mean(lstm, 1) # reduce the dimension of the tokens by calculating their average
        # print(fc_input.shape)
        logits = self.classifier(fc_input)

        return logits