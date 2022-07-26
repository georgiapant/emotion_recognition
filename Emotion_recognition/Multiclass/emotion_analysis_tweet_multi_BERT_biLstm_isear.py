from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertModel #,BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import numpy as np
import emoji
import contractions
import re
import gc
from collections import Counter
import datetime
from pytorchtools import EarlyStopping

gc.collect()

device = torch.device('cuda')
print('GPU:', torch.cuda.get_device_name(0))

BERT_MODEL = 'bert-base-uncased' #"roberta-base"
RANDOM_SEED = 42
MAX_LEN = 126

# For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
BATCH_SIZE = 16  # TODO

EPOCHS = 100
bidirectional = False

loss_fn = nn.CrossEntropyLoss()
emotions = ['sadness', 'fear', 'guilt', 'joy', 'shame', 'anger', 'disgust']
num_labels = len(emotions)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
# tokenizer = RobertaTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)


def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    :param text:
    """
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

    # Demojize
    x = emoji.demojize(text)

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

    # Remove special characters and numbers replace by space + remove double space
    text = re.sub(r"\b([.]{3,})", " dots ", x)
    text = re.sub(r"[^A-Za-z!?_]+", " ", text)
    text = re.sub(r"\b([s])\b *", "", text)
    text = re.sub(r" +", " ", text)
    text = text.strip()

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


def datasets():
    # Load data and set labels

    data = pd.read_csv(
        r"C:\Users\georgiapant\PycharmProjects\REBECCA\Datasets\Emotion\ISEAR_DATA.csv",
        encoding="utf8", low_memory=False)

    data = data[['SIT', 'Field1']]
    data.rename(columns={'SIT':'Text', 'Field1':'emotion'}, inplace=True)

    X = data['Text'].values
    y = data.drop(columns=['Text'])
    # y['emotion'] = (y.iloc[:,1:] ==1).idxmax(1)
    y = y['emotion']

    stats = pd.DataFrame()
    stats['count'] = y.value_counts()
    stats['percent'] = 100 * stats['count'] / len(y)
    stats['emotion'] = stats.index
    stats = stats[['count', 'percent', 'emotion']]
    # stats.plot.pie(y='percent')
    stats = stats.reset_index(drop=True)
    print(stats)
    # Transform text labels to numbers
    d = dict(zip(emotions, range(0, num_labels)))
    print(d)
    y['label'] = y.map(d, na_action='ignore').astype('int64')
    y = y['label']

    # stats = pd.DataFrame()
    # # Create statistics of dataset
    # stats['Complete'] = y.sum(axis = 0, skipna=True)

    # load train, test and validation data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=RANDOM_SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_SEED, stratify=y_test)

    # stats['Train'] = y_train.sum(axis=0)
    # stats['Val'] = y_val.sum(axis=0)
    # stats['Test'] = y_test.sum(axis=0)
    # print(f'The statistics of the dataset are:\n {stats}')
    # print(f"Total: {stats['complete'].sum(axis=0)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


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
        D_in, H, D_out = 768, 200, num_labels

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(BERT_MODEL, problem_type="multi_label_classification")  # , config=config)
        self.bert_drop = nn.Dropout(0.3)

        if bidirectional:
            self.LSTM = nn.LSTM(D_in, H, num_layers=1, bidirectional=True, batch_first=True)
            self.linear = nn.Linear(2 * H, D_out)
        else:
            self.LSTM = nn.LSTM(D_in, H, num_layers=1, bidirectional=False, batch_first=True)
            self.linear = nn.Linear(H, D_out)

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            # nn.ReLU(),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(H, D_out)
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

        # Original code
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        lstm, (h,c) = self.LSTM(output)
        lstm_output = lstm[:, 0, :]
        drop = self.bert_drop(lstm_output)
        logits = self.linear(drop)
        # LSTM v1
        # xx = lstm[-1]
        # hidden_last_cell = h[-1]



        # Simple Classifier
        # last_hidden_state_cls = output[:, 0, :]
        # # Feed input to classifier to compute logits
        # logits = self.classifier(last_hidden_state_cls)

        #return logits#, hidden
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
                      lr=3e-5,  # Default learning rate 3e-5
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

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

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
    print(f'Accuracy: {accuracy * 100:.2f}%',  flush=True)
    print(f'F1-score macro: {f1_score(y_true, y_pred, average="macro")}',  flush=True)
    print(f'Classification report: {classification_report(y_true, y_pred)}',  flush=True)

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
    early_stopping = EarlyStopping(patience=3, verbose=True)
    t0 = time.time()
    for epoch_i in range(epochs):
        gc.collect()
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}",  flush=True)
        print("-" * 70,  flush=True)

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
            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

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
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed_batch} | {time_elapsed_total}",  flush=True)

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
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed}", flush=True)
            print("-" * 70)

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        model.load_state_dict(torch.load('checkpoint.pt'))
        print("\n")
    print(f"Total training time: {format_time(time.time() - t0)}", flush=True)

    print("Training complete!",  flush=True)


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

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
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
    train(bert_classifier, train_dataloader, val_dataloader, optimizer, scheduler, epochs=EPOCHS, evaluation=True)

    print("Validation set",  flush=True)
    probs_val = bert_predict(bert_classifier, val_dataloader)
    evaluate(probs_val, y_val)

    print("Test set",  flush=True)
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
