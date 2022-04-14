from sklearn.model_selection import train_test_split
import torch
from transformers import BertModel, BertTokenizer,  XLMRobertaTokenizer, XLMRobertaModel  # ,BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import numpy as np
import re
import gc
from collections import Counter
import datetime
import nrclex
import emoji
import contractions
from pytorchtools import EarlyStopping
import json

gc.collect()

device = torch.device('cuda')
print('GPU:', torch.cuda.get_device_name(0))

BERT_MODEL = 'bert-base-uncased'
RANDOM_SEED = 42
MAX_LEN = 126

# For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
BATCH_SIZE = 16  # TODO
EPOCHS = 100
patience = 3
bidirectional = True

loss_fn = nn.CrossEntropyLoss()
emotions = ['sadness', 'worry', 'surprise', 'love', 'happiness', 'anger']
num_labels = len(emotions)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)


def text_preprocessing(text, tokenize=False):


    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub(r" +", ' ', text)

    text = text.lower()
    # Demojize
    text = emoji.demojize(text)

    # Expand contraction
    text = contractions.fix(text)


    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    # correct some acronyms/typos/abbreviations
    x = re.sub(r"lmao", "laughing my ass off", text)
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
    text = re.sub(r"\b(doj)\b", "department of justice", x)

    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)
    # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)
    text = str(re.sub("\S*\d\S*", "", text).strip())

    return text


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


def read_file(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    label = []
    text = []
    for l in lines:
        label.append(int(l[0]))
        text.append(l[2:])

    label = np.array(label)
    return text, label


def datasets():
    # Load data and set labels
    # data = pd.read_csv(
    #     r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\emotions_merged.csv",
    #     encoding="utf8", low_memory=False)

    data = pd.read_csv(
        r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\emotions_merged_original.csv",
        encoding="utf8", low_memory=False)

    data = data[['sentiment', 'content']]

    print('Original dataset shape %s' % Counter(data['sentiment']), flush=True)

    # Create statistics of dataset
    stats = pd.DataFrame()
    stats['count'] = data.groupby('sentiment').count()['content']
    stats['percent'] = 100 * stats['count'] / len(data)
    stats['sentiment'] = stats.index
    stats = stats[['count', 'percent', 'sentiment']]
    # stats.plot.pie(y='percent')
    stats = stats.reset_index(drop=True)
    print(stats)
    #
    # orig_dataset = data.reset_index(drop=True)
    # dataset = pd.DataFrame(columns=['content', 'sentiment'])
    # trim = round(stats['count'].mean())
    #
    # for i in range(len(stats)):
    #     temp = orig_dataset[orig_dataset['sentiment'] == stats['sentiment'][i]]
    #     dataset = dataset.append(temp[0:trim], ignore_index=True)
    #
    # print('Trimmed dataset shape %s' % Counter(dataset['sentiment']), flush=True)

    # Transform text labels to numbers
    d = dict(zip(emotions, range(0, 13)))
    data['label'] = data['sentiment'].map(d, na_action='ignore').astype('int64')
    data.drop(['sentiment'], inplace=True, axis=1)

    # Resample with  to deal with dataset imbalance

    # sampler = RandomOverSampler(sampling_strategy='all') #TODO
    # sampler = RandomOverSampler(sampling_strategy='all')
    #
    # X_res, y_res = sampler.fit_resample(data['content'].values.reshape(-1, 1), data['label'].values.reshape(-1, 1))
    #
    # y = pd.DataFrame(y_res)
    # y.columns = ['label']
    # x = pd.DataFrame(X_res)
    # x.columns = ['content']
    #
    # print('Resampled dataset shape %s' % Counter(y.squeeze()),  flush=True)
    # tw_y = y.squeeze()
    # tw_x = x.squeeze()
    # tw = pd.concat([tw_x, tw_y], axis=1)

    X = data['content'].values
    y = data['label'].values

    # load train, test and validation data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_SEED,
                                                    stratify=y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


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


class RobertaClassifier(nn.Module):
    """
    Bert Model for Classification Tasks.
    """

    def __init__(self, freeze_bert=False):
        """
        @param    robert: a RobertaModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(RobertaClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H_lstm, H, D_out = 768, 200, 384, num_labels

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(BERT_MODEL, problem_type="multi_label_classification")

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


def initialize_model(train_dataloader, epochs=EPOCHS):
    """
    Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = RobertaClassifier(freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,  # Default learning rate
                      eps=1e-8  # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)

    return bert_classifier, optimizer, scheduler


def validation(model, val_dataloader):
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

        b_labels = b_labels.type(torch.LongTensor).to(device)
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


def evaluate(probs, y_true):
    """
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    y_pred = []

    for i in range(len(probs)):
        pred = np.where(probs[i] == np.amax(probs[i]))[0][0]
        y_pred.append(pred)

    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%', flush=True)
    print(f'F1-score macro: {f1_score(y_true, y_pred, average="macro")}', flush=True)
    print(f'Classification report: {classification_report(y_true, y_pred)}', flush=True)


def set_seed(seed_value=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train(model, train_dataloader, val_dataloader=None, optimizer=None, scheduler=None, epochs=4, evaluation=False):
    """
    Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n", flush=True)
    t0 = time.time()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch_i in range(epochs):
        gc.collect()
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}",
              flush=True)
        print("-" * 70, flush=True)

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
            b_labels = b_labels.type(torch.LongTensor).to(device)

            loss = loss_fn(logits, b_labels)
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
            if (step % 100 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed_batch = format_time(time.time() - t0_batch)
                time_elapsed_total = format_time(time.time() - t0_epoch)
                # Print training results
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed_batch} | {time_elapsed_total}",
                    flush=True)

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = validation(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = format_time(time.time() - t0_epoch)

            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed}",
                flush=True)
            print("-" * 70)

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        model.load_state_dict(torch.load('checkpoint.pt'))
        print("\n")
    print(f"Total training time: {format_time(time.time() - t0)}", flush=True)

    print("Training complete!", flush=True)


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
            logits = model(b_input_ids, b_attn_mask,lex_feats, vad)
        all_logits.append(logits)

    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    print(f"Total prediction time: {format_time(time.time() - t0)}", flush=True)

    return probs


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = datasets()

    print('Tokenizing data...', flush=True)
    train_inputs, train_masks = preprocessing_for_bert(X_train)
    val_inputs, val_masks = preprocessing_for_bert(X_val)
    print('Done', flush=True)

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

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
    train(bert_classifier, train_dataloader, val_dataloader, optimizer, scheduler, epochs=EPOCHS, evaluation=True)

    print("Validation set", flush=True)
    probs_val = bert_predict(bert_classifier, val_dataloader)
    evaluate(probs_val, y_val)

    print("Test set", flush=True)
    print('Tokenizing data...', flush=True)
    test_inputs, test_masks = preprocessing_for_bert(X_test)

    # Create the DataLoader for our test set
    test_dataset = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

    probs_test = bert_predict(bert_classifier, test_dataloader)
    evaluate(probs_test, y_test)


if __name__ == '__main__':
    main()
