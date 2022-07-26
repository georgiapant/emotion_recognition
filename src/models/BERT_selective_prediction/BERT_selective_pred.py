import torch.nn as nn
from src.config import BERT_MODEL, num_labels, basic_dropout_rate #, mc_dropout_rate
from transformers import BertModel
import torch
import torch.nn.functional as F


class EmoEkman(nn.Module):
    """
    Bert Model for Classification Tasks.
    """

    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(EmoEkman, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 384, num_labels
        self.mc_dropout_rate = 0

        self.bert = BertModel.from_pretrained(BERT_MODEL, problem_type="multi_label_classification")
        # self.sigmoid = nn.Sigmoid()
        # Instantiate BERT model

        self.net_init= nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(basic_dropout_rate),
            LambdaLayer(lambda x: F.dropout(x, p=self.mc_dropout_rate))
        )

        self.classification_head = nn.Sequential(
            nn.Linear(H, D_out),
            # nn.Softmax()
        )

        self.selection_head = nn.Sequential(
            nn.Linear(H, 2),
            # nn.ReLU(),
            # nn.Dropout(basic_dropout_rate - 0.1),
            # nn.Linear(96, 2),
            # nn.ReLU(),
            # nn.Dropout(basic_dropout_rate - 0.1),
            # nn.Linear(24, 2), # and THIS not the above
            # nn.Sigmoid()
        )

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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # print(outputs)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        out = self.net_init(last_hidden_state_cls)

        #Classifier head

        out_class_logits = self.classification_head(out)

        selection_out = self.selection_head(out) # output is 2 (emo or no emo?) and THIS


        # class_select_out = selection_out[:, -1:] #index 1 (the last) (no emo??) and THIS
        class_select_out = selection_out[:, :-1] #index 1 (the first) (emo??) and THIS

        # # class_select_out = selection_out[:, 1:] #index 0 (the first) (emo ??)
        #
        #
        # # class_select_out = torch.mean(class_select_out, 1, keepdim=True)
        # #class_select_out = torch.sum(class_select_out, 1, keepdim=True)
        # #select_out_bin = torch.cat((class_select_out, selection_out[:,-1:]), dim=1)
        # #select_out_bin = self.sigmoid(select_out_bin)
        # # select_out_bin_logits = -torch.log((1 / (select_out_bin + 1e-8)) - 1)
        #
        selection_out_logits = torch.cat((out_class_logits, class_select_out), dim=1) # THIS

        return selection_out_logits, out_class_logits


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)