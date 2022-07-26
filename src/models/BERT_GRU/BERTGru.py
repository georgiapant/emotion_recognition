import torch
import torch.nn as nn
from src.config import BATCH_SIZE, MAX_LEN, BERT_MODEL, num_labels
from transformers import BertModel, BertTokenizer
from capsule_layer import CapsuleLinear


class bertGruClassifier(nn.Module):
    """
    Bert Model for Classification Tasks.
    """

    def __init__(self, freeze_bert=False): #, BERT_MODEL, num_labels, BATCH_SIZE):
        """
        @param    robert: a RobertaModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(bertGruClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels

        self.n_classes = num_labels

        # GRU
        self.hidden = 512
        self.n_layers = 2
        self.batch_size = BATCH_SIZE

        # CapsNet
        self.in_length = 128
        self.out_length = 8
        self.share_weight = True
        self.routing_type = "dynamic"
        self.num_iterations = 5

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(BERT_MODEL, problem_type="multi_label_classification")#, torchscript=True)
        self.embedding_dim = self.bert.config.to_dict()["hidden_size"]  # 768

        self.gru1 = nn.GRU(
            self.embedding_dim * 4,
            self.hidden,
            self.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0 if self.n_layers == 1 else 0.25,
        )

        self.gru2 = nn.GRU(
            self.embedding_dim * 4,
            self.hidden,
            self.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0 if self.n_layers == 1 else 0.25,
        )

        self.gru3 = nn.GRU(
            self.embedding_dim * 4,
            self.hidden,
            self.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0 if self.n_layers == 1 else 0.25,
        )

        self.dropout = nn.Dropout(0.5)
        self.caps_net = CapsuleLinear(
            out_capsules=self.n_classes,
            in_length=self.in_length,
            out_length=self.out_length,
            share_weight=self.share_weight,
            routing_type=self.routing_type,
            num_iterations=self.num_iterations,
        )
        self.fc = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, self.n_classes)

        self.caps_fc = nn.Linear(self.hidden, self.in_length)

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False


    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """

        sequence_output, pooled_output, hidden_states = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=False
        )

        embedding_output = hidden_states[0]
        attention_hidden_states = hidden_states[1:]

        # Concat batches of 4 hidden layers of BERT
        pooled_output_1 = torch.cat(
            tuple([attention_hidden_states[i] for i in [-12, -11, -10, -9]]), dim=-1
        )  # (64, 210, 3072)
        pooled_output_2 = torch.cat(
            tuple([attention_hidden_states[i] for i in [-8, -7, -6, -5]]), dim=-1
        )  # (64, 210, 3072)
        pooled_output_3 = torch.cat(
            tuple([attention_hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1
        )  # (64, 210, 3072)

        out1, hidden1 = self.gru1(pooled_output_1)  # (4, 64, 512)
        out2, hidden2 = self.gru2(pooled_output_2)  # (4, 64, 512)
        out3, hidden3 = self.gru3(pooled_output_3)  # (4, 64, 512)

        hidden1 = hidden1.view(self.n_layers, 2, self.batch_size, self.hidden)
        hidden1 = hidden1[-1]  # (2, 64, 512)

        hidden2 = hidden2.view(self.n_layers, 2, self.batch_size, self.hidden)
        hidden2 = hidden2[-1]  # (2, 64, 512)

        hidden3 = hidden3.view(self.n_layers, 2, self.batch_size, self.hidden)
        hidden3 = hidden3[-1]  # (2, 64, 512)

        hidden_concat = torch.cat((hidden1, hidden2, hidden3), dim=0)  # (6, 64, 512)

        ############################### CapsNet ###############################

        # Average of hidden layers
        caps_input = torch.mean(hidden_concat, 0)  # (64, 512)

        # Dropout layer
        caps_input = self.dropout(caps_input)

        # Fully connected
        caps_input = self.caps_fc(caps_input)  # (64, 128)

        # We need an extra dimension for CapsNet
        caps_input = caps_input.unsqueeze(1)  # (64, 1 , 128)

        # Dropout layer
        caps_input = self.dropout(caps_input)

        # Capsule classifier
        caps_input = caps_input.to(torch.float32).contiguous()
        caps_output, caps_prob = self.caps_net(caps_input)  # (64, 3, 8)

        # Classification
        caps_output = caps_output.norm(dim=-1)  # (64, 3)

        ########################### Fully connected ###########################

        # Average of hidden layers
        fc_input = torch.mean(hidden_concat, 0)  # (64, 512)

        # Dropout layer
        fc_input = self.dropout(fc_input)

        # Fully connected layer
        fc_input = self.fc(fc_input)  # (64, 128)

        # Dropout layer
        fc_input = self.dropout(fc_input)

        # Fully connected layer
        fc_input = self.fc2(fc_input)  # (64, 32)

        # Dropout layer
        fc_input = self.dropout(fc_input)

        # Fully connected layer
        fc_output = self.fc3(fc_input)  # (64, 3)

        ############################# Soft Voting #############################

        # Soft voting
        ensemble_logits = (caps_output + fc_output) / 2.0

        return ensemble_logits