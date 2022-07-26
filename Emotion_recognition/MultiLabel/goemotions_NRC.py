import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import time
from sklearn import metrics
from sklearn.metrics import f1_score, classification_report, jaccard_score, precision_recall_curve
import pandas as pd
import numpy as np
import re
import gc
import datetime
import emoji
import contractions
from src.pytorchtools import EarlyStopping
import torch.nn.functional as F
import json
import nrclex
from sklearn.utils import compute_class_weight

gc.collect()

device = torch.device('cuda')
print('GPU:', torch.cuda.get_device_name(0))

BERT_MODEL = 'bert-base-uncased'
# BERT_MODEL = 'bert-base-multilingual-cased'
RANDOM_SEED = 42
MAX_LEN = 126

# For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
BATCH_SIZE = 16  # TODO
EPOCHS = 100
patience = 3
bert_simple = True
bidirectional = True
np.seterr(divide='ignore', invalid='ignore')

# loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.BCEWithLogitsLoss()

emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
            'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude',
            'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse',
            'sadness', 'surprise', 'neutral']
num_labels = len(emotions)

"""
admiration        5122
amusement         2895
anger             1960
annoyance         3093
approval          3687
caring            1375
confusion         1673
curiosity         2723
desire             801
disappointment    1583
disapproval       2581
disgust           1013
embarrassment      375
excitement        1052
fear               764
gratitude         3372
grief               96
joy               1785
love              2576
nervousness        208
optimism          1976
pride              142
realization       1382
relief             182
remorse            669
sadness           1625
surprise          1330
Total             46040
"""

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)


def text_preprocessing(x):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    :param text:
    """
    # Remove '@name'
    x = re.sub(r'(@.*?)[\s]', ' ', x)

    # Replace '&amp;' with '&'
    x = re.sub(r'&amp;', '&', x)

    # Remove trailing whitespace
    x = re.sub(r'\s+', ' ', x).strip()

    x = re.sub(r"https?://[A-Za-z0-9./]+", ' ', x)
    x = re.sub(r"[^a-zA-z.!?'0-9]", ' ', x)
    x = re.sub('\t', ' ', x)
    x = re.sub(r" +", ' ', x)

    # from the other program
    x = re.sub(r'([a-zA-Z\[\]])([,;.!?])', r'\1 \2', x)
    x = re.sub(r'([,;.!?])([a-zA-Z\[\]])', r'\1 \2', x)

    # Demojize
    x = emoji.demojize(x)

    # Expand contraction
    x = contractions.fix(x)

    # Lower
    x = x.lower()

    # correct some acronyms/typos/abbreviations
    x = re.sub(r"lmao", "laughing my ass off", x)
    x = re.sub(r"amirite", "am i right", x)
    x = re.sub(r"\b(tho)\b", "though", x)
    x = re.sub(r"\b(ikr)\b", "i know right", x)
    x = re.sub(r"\b(ya|u)\b", "you", x)
    x = re.sub(r"\b(eu)\b", "europe", x)
    x = re.sub(r"\b(da)\b", "the", x)
    x = re.sub(r"\b(dat)\b", "that", x)
    x = re.sub(r"\b(dats)\b", "that is", x)
    x = re.sub(r"\b(cuz)\b", "because", x)
    x = re.sub(r"\b(fkn)\b", "fucking", x)
    x = re.sub(r"\b(tbh)\b", "to be honest", x)
    x = re.sub(r"\b(tbf)\b", "to be fair", x)
    x = re.sub(r"faux pas", "mistake", x)
    x = re.sub(r"\b(btw)\b", "by the way", x)
    x = re.sub(r"\b(bs)\b", "bullshit", x)
    x = re.sub(r"\b(kinda)\b", "kind of", x)
    x = re.sub(r"\b(bruh)\b", "bro", x)
    x = re.sub(r"\b(w/e)\b", "whatever", x)
    x = re.sub(r"\b(w/)\b", "with", x)
    x = re.sub(r"\b(w/o)\b", "without", x)
    x = re.sub(r"\b(doj)\b", "department of justice", x)

    # replace some words with multiple occurences of a letter, example "coooool" turns into --> cool
    x = re.sub(r"\b(j+e{2,}z+e*)\b", "jeez", x)
    x = re.sub(r"\b(co+l+)\b", "cool", x)
    x = re.sub(r"\b(g+o+a+l+)\b", "goal", x)
    x = re.sub(r"\b(s+h+i+t+)\b", "shit", x)
    x = re.sub(r"\b(o+m+g+)\b", "omg", x)
    x = re.sub(r"\b(w+t+f+)\b", "wtf", x)
    x = re.sub(r"\b(w+h+a+t+)\b", "what", x)
    x = re.sub(r"\b(y+e+y+|y+a+y+|y+e+a+h+)\b", "yeah", x)
    x = re.sub(r"\b(w+o+w+)\b", "wow", x)
    x = re.sub(r"\b(w+h+y+)\b", "why", x)
    x = re.sub(r"\b(s+o+)\b", "so", x)
    x = re.sub(r"\b(f)\b", "fuck", x)
    x = re.sub(r"\b(w+h+o+p+s+)\b", "whoops", x)
    x = re.sub(r"\b(ofc)\b", "of course", x)
    x = re.sub(r"\b(the us)\b", "usa", x)
    x = re.sub(r"\b(gf)\b", "girlfriend", x)
    x = re.sub(r"\b(hr)\b", "human ressources", x)
    x = re.sub(r"\b(mh)\b", "mental health", x)
    x = re.sub(r"\b(idk)\b", "i do not know", x)
    x = re.sub(r"\b(gotcha)\b", "i got you", x)
    x = re.sub(r"\b(y+e+p+)\b", "yes", x)
    x = re.sub(r"\b(a*ha+h[ha]*|a*ha +h[ha]*)\b", "haha", x)
    x = re.sub(r"\b(o?l+o+l+[ol]*)\b", "lol", x)
    x = re.sub(r"\b(o*ho+h[ho]*|o*ho +h[ho]*)\b", "ohoh", x)
    x = re.sub(r"\b(o+h+)\b", "oh", x)
    x = re.sub(r"\b(a+h+)\b", "ah", x)
    x = re.sub(r"\b(u+h+)\b", "uh", x)

    # Handling emojis
    x = re.sub(r"<3", " love ", x)
    x = re.sub(r"xd", " smiling_face_with_open_mouth_and_tightly_closed_eyes ", x)
    x = re.sub(r":\)", " smiling_face ", x)
    x = re.sub(r"^_^", " smiling_face ", x)
    x = re.sub(r"\*_\*", " star_struck ", x)
    x = re.sub(r":\(", " frowning_face ", x)
    x = re.sub(r":\^\(", " frowning_face ", x)
    x = re.sub(r";\(", " frowning_face ", x)
    x = re.sub(r":\/", " confused_face", x)
    x = re.sub(r";\)", " wink", x)
    x = re.sub(r">__<", " unamused ", x)
    x = re.sub(r"\b([xo]+x*)\b", " xoxo ", x)
    x = re.sub(r"\b(n+a+h+)\b", "no", x)

    # Remove special characters and numbers replace by space + remove double space
    x = re.sub(r"\b([.]{3,})", " dots ", x)
    x = re.sub(r"[^A-Za-z!?_]+", " ", x)
    x = re.sub(r"\b([s])\b *", "", x)
    x = re.sub(r" +", " ", x)
    x = x.strip()

    return x


def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,  # Max length to truncate/pad
            # pad_to_max_length=True,
            padding='max_length',  # Pad sentence to max length
            truncation='longest_first',
            return_tensors='pt',  # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    # input_ids = torch.tensor(input_ids)
    # attention_masks = torch.tensor(attention_masks)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def datasets():
    # Load data and set labels
    stats = pd.DataFrame()

    train = pd.read_csv(
        r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\GoEmotions\train.tsv",
        encoding="utf8", low_memory=False, sep='\t', names=['Text', 'sentiment_id', 'id'])

    validation = pd.read_csv(
        r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\GoEmotions\dev.tsv",
        encoding="utf8", low_memory=False, sep='\t', names=['Text', 'sentiment_id', 'id'])

    test = pd.read_csv(
        r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\GoEmotions\test.tsv",
        encoding="utf8", low_memory=False, sep='\t', names=['Text', 'sentiment_id', 'id'])

    # Loading emotion labels for GoEmotions taxonomy
    with open(r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\GoEmotions\emotions.txt", "r") as file:
        GE_taxonomy = file.read().split("\n")

    size_train = train.shape[0]
    size_val = validation.shape[0]
    size_test = test.shape[0]

    df_all = pd.concat([train, validation, test], axis=0).reset_index(drop=True).drop(['id'], axis=1)
    df_all['sentiment_id'] = df_all['sentiment_id'].apply(lambda x: x.split(','))

    def idx2class(idx_list):
        arr = []
        for i in idx_list:
            arr.append(GE_taxonomy[int(i)])
        return arr

    df_all['sentiment'] = df_all['sentiment_id'].apply(idx2class)

    # OneHot encoding for multi-label classification
    for emo in GE_taxonomy:
        df_all[emo] = np.zeros((len(df_all), 1))
        df_all[emo] = df_all['sentiment'].apply(lambda x: 1 if emo in x else 0)

    df_all = df_all.drop(['sentiment_id', 'sentiment'], axis=1)
    print(df_all)
    weights = calculating_class_weights(df_all.drop(columns=['Text']))
    print(f'The weights of the classes are:\n {weights}')

    X_train = df_all.iloc[:size_train, :]['Text']
    X_val = df_all.iloc[size_train:size_train + size_val, :]['Text']
    X_test = df_all.iloc[size_train + size_val:size_train + size_val + size_test, :]['Text']

    y_train = df_all.iloc[:size_train, :].drop(columns=['Text'])
    y_val = df_all.iloc[size_train:size_train + size_val, :].drop(columns=['Text'])
    y_test = df_all.iloc[size_train + size_val:size_train + size_val + size_test, :].drop(columns=['Text'])

    stats['Train'] = y_train.sum(axis=0)
    stats['Val'] = y_val.sum(axis=0)
    stats['Test'] = y_test.sum(axis=0)
    print(f'The statistics of the dataset are:\n {stats}')

    return X_train, y_train, X_val, y_val, X_test, y_test, weights


def calculating_class_weights(y_true):
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(len(emotions)):
        weights[i] = compute_class_weight('balanced', classes=[0, 1], y=y_true[emotions[i]])
    return weights


def get_weighted_loss(logits, labels_true, weights):
    loss_fn = nn.BCEWithLogitsLoss()
    weights = torch.from_numpy(weights).to(device)

    zero_cls = weights[:, 0] ** (1 - labels_true)
    one_cls = weights[:, 1] ** labels_true
    loss = loss_fn(logits, labels_true)
    weighted_loss = torch.mean((zero_cls * one_cls) * loss)

    return weighted_loss


def nrc_feats(input_ids):
    vals_corpus = []

    for sentence in input_ids:
        vals_sentence = []
        sentence = tokenizer.convert_ids_to_tokens(sentence)
        for word in sentence:
            emos = nrclex.NRCLex(word)
            frqs = emos.affect_frequencies.values()
            vals_sentence.append(list(frqs))
        vals_corpus.append(vals_sentence)
    feat = torch.tensor(vals_corpus)
    return feat


def vad_feats(input_ids):

    arousal_dict = json.load(open(r"C:\Users\georgiapant\PycharmProjects\REBECCA\Resources\vad\arousal_dict.json"))
    valence_dict = json.load(open(r"C:\Users\georgiapant\PycharmProjects\REBECCA\Resources\vad\valence_dict.json"))
    dom_dict = json.load(open(r"C:\Users\georgiapant\PycharmProjects\REBECCA\Resources\vad\dom_dict.json"))

    vad_corpus = []

    for sentence in input_ids:
        vad_sentence = []
        sentence = tokenizer.batch_decode(sentence, clean_up_tokenization_spaces=False)
        arousal_vec = [arousal_dict.get(i) for i in sentence]
        arousal_vec = [float(0.5) if v is None else float(v) for v in arousal_vec]
        vad_sentence.extend(arousal_vec)

        valence_vec = [valence_dict.get(i) for i in sentence]
        valence_vec = [float(0.5) if v is None else float(v) for v in valence_vec]
        vad_sentence.extend(valence_vec)

        dom_vec = [dom_dict.get(i) for i in sentence]
        dom_vec = [float(0.5) if v is None else float(v) for v in dom_vec]
        vad_sentence.extend(dom_vec)

        vad_sentence = np.reshape(vad_sentence, (3, MAX_LEN)).T

        vad_corpus.append(vad_sentence)

    vad = torch.tensor(vad_corpus).float()
    return vad


class BertClassifier(nn.Module):
    """
    Bert Model for Classification Tasks.
    """

    def __init__(self, freeze_bert=False):
        """
        @param    robert: a RobertaModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H_lstm, H, D_out = 768, 200, 384, num_labels

        # Instantiate BERT model
        self.bert = BertForSequenceClassification.from_pretrained(BERT_MODEL, problem_type="multi_label_classification")

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

        bert_out = \
            self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states = True).hidden_states[-1]
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


def initialize_model(train_dataloader, epochs=EPOCHS):
    """
    Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=3e-5,  # Default learning rate
                      eps=1e-8  # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)

    return bert_classifier, optimizer, scheduler


def monitor_metrics(logits, labels_true):
    if labels_true is None:
        return {}

    preds = torch.sigmoid(logits)
    preds = preds.cpu().detach().numpy()
    labels_true = labels_true.cpu().detach().numpy()

    fpr_micro, tpr_micro, thresholds = metrics.roc_curve(labels_true.ravel(), preds.ravel())
    auc_micro = metrics.auc(fpr_micro, tpr_micro)
    thresholds = []

    for i in range(len(emotions)):
        p, r, th = precision_recall_curve(labels_true[:, i], preds[:, i])
        f1 = np.nan_to_num(2 * p * r / (p + r), 0)
        f1_max = f1.argmax()
        thresholds.append(th[f1_max])
        # f1_scores.append(f1[f1_max])

    y_pred = preds > np.asarray(thresholds)

    # opt_idx = np.argmax(tpr_micro - fpr_micro)
    # threshold = thresholds[opt_idx]
    # y_pred = logits_to_labels(preds, threshold=thresholds[opt_idx])

    accuracy = jaccard_score(labels_true, y_pred, average='samples')

    return auc_micro, accuracy, thresholds


def logits_to_labels(logits, threshold=0.1):
    y_pred_labels = np.zeros_like(logits)

    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            if logits[i][j] > threshold:
                y_pred_labels[i][j] = 1
            else:
                y_pred_labels[i][j] = 0

    return y_pred_labels


def evaluate(logits, y_true, thresholds):
    logits = torch.sigmoid(logits)
    logits = logits.cpu().detach().numpy()
    y_true = y_true.to_numpy()

    roc_metrics = []

    for i in range(len(emotions)):
        roc = metrics.roc_auc_score(y_true[:,i], logits[:, i])
        roc_metrics.append(roc)

    s = pd.DataFrame(roc_metrics, index=emotions)
    print(f'AUC metrics - F1 - threshold for all classes:\n {s}', flush=True)

    y_pred = logits > np.asarray(thresholds) # Each class different threshold
    # y_pred = logits_to_labels(logits, threshold)

    print(f'2. Classification report:\n {classification_report(y_true, y_pred, target_names=emotions)}', flush=True)

    # jaccard score for multilabel classification
    accuracy = jaccard_score(y_true, y_pred, average='samples')
    fscore_micro = f1_score(y_true, y_pred, average='micro')
    fscore_macro = f1_score(y_true, y_pred, average='macro')
    print('Accuracy: %f F1-score micro: %f F1-score macro: %f' % (accuracy, fscore_micro, fscore_macro))


def set_seed(seed_value=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def validation(model, val_dataloader, weights= None):
    """
    After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    val_auc_micro = []
    val_thresholds = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        lex_feats = nrc_feats(b_input_ids).to(device)
        vad = vad_feats(b_input_ids).to(device)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask, lex_feats, vad)

        # Compute loss
        b_labels = b_labels.float().to(device)
        # loss = loss_fn(logits, b_labels)
        loss = get_weighted_loss(logits, b_labels, weights)
        val_loss.append(loss.item())

        auc_micro, accuracy, thresholds = monitor_metrics(logits, b_labels)
        val_auc_micro.append(auc_micro)
        val_accuracy.append(accuracy)
        val_thresholds.append(thresholds)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    val_auc = np.mean(val_auc_micro)
    val_thresholds = np.mean(val_thresholds, axis=0)

    return val_loss, val_accuracy, val_auc, val_thresholds


def train(model, train_dataloader, val_dataloader=None, optimizer=None, scheduler=None, epochs=EPOCHS,
          evaluation=False, weights = None):
    """
    Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n", flush=True)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    t0 = time.time()
    threshold_list = []
    thresholds_fin = []
    for epoch_i in range(epochs):
        gc.collect()
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(
            f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val Auc': ^9}| {'Elapsed':^12} | {'Elapsed Total':^12}",
            flush=True)
        print("-" * 100, flush=True)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            lex_feats = nrc_feats(b_input_ids).to(device)
            vad = vad_feats(b_input_ids).to(device)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask, lex_feats, vad)

            # Compute loss and accumulate the loss values
            b_labels = b_labels.float().to(device)
            # loss = loss_fn(logits, b_labels)
            loss = get_weighted_loss(logits, b_labels, weights)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed_batch = format_time(time.time() - t0_batch)
                time_elapsed_total = format_time(time.time() - t0_epoch)
                # Print training results
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {'-':^9}|{time_elapsed_batch:^12} | {time_elapsed_total:^12}",
                    flush=True)

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 100)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy, val_auc, thresholds = validation(model, val_dataloader, weights=weights)

            # Print performance over the entire training data
            time_elapsed = format_time(time.time() - t0_epoch)
            threshold_list.append(thresholds)

            print(
                f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val Auc': ^9}| {'Elapsed':^12} | {'Elapsed Total':^12}",
                flush=True)
            print("-" * 100, flush=True)
            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {val_auc:^9.6f}| | {'-':^12}| {time_elapsed:^12}",
                flush=True)
            print("-" * 100)

            early_stopping(val_loss, model)
            print(f"All thresholds list: {threshold_list}", flush=True)

            if early_stopping.early_stop:
                print("Early stopping")
                thresholds_fin = np.mean(threshold_list[0:epoch_i-patience+1], axis=0)
                break
            else:
                thresholds_fin = np.mean(threshold_list, axis=0)

        model.load_state_dict(torch.load('checkpoint.pt'))
        # print(threshold_list)

        print("\n")
    print(f"Thresholds are {thresholds_fin}", flush=True)
    print(f"Total training time: {format_time(time.time() - t0)}", flush=True)

    print("Training complete!", flush=True)
    return thresholds_fin


def bert_predict(model, test_dataloader):
    """
    Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []
    t0 = time.time()

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
        lex_feats = nrc_feats(b_input_ids).to(device)
        vad = vad_feats(b_input_ids).to(device)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask, lex_feats, vad)
        all_logits.append(logits)

    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    # probs = F.softmax(all_logits, dim=1).cpu().numpy()

    print(f"Total prediction time: {format_time(time.time() - t0)}", flush=True)

    return all_logits


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, weights = datasets()

    print('Tokenizing data...', flush=True)
    train_inputs, train_masks = preprocessing_for_bert(X_train)
    val_inputs, val_masks = preprocessing_for_bert(X_val)
    print('Done', flush=True)

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train.values)
    val_labels = torch.tensor(y_val.values)

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

    set_seed(42)  # Set seed for reproducibility
    bert_classifier, optimizer, scheduler = initialize_model(train_dataloader, epochs=EPOCHS)
    thresholds = train(bert_classifier, train_dataloader, val_dataloader, optimizer, scheduler, epochs=EPOCHS,
                      evaluation=True, weights=weights)

    # bert_classifier.load_state_dict(torch.load('checkpoint.pt'))

    print("Validation set", flush=True)
    probs_val = bert_predict(bert_classifier, val_dataloader)
    evaluate(probs_val, y_val, thresholds)

    print("Test set", flush=True)
    print('Tokenizing data...', flush=True)
    test_inputs, test_masks = preprocessing_for_bert(X_test)

    # Create the DataLoader for our test set
    test_dataset = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

    probs_test = bert_predict(bert_classifier, test_dataloader)
    evaluate(probs_test, y_test, thresholds)


if __name__ == '__main__':
    main()
