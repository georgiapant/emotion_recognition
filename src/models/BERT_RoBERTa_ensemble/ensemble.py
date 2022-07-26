import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import random
import sys

project_root_path = "./src"
# project_root_path = "C:/Users/georgiapant/PycharmProjects/GitHub/rebecca"
sys.path.append(project_root_path)

from collections import Counter

import torch
# from data_prep import machine_translation
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AdamW,
    AutoTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    logging,
)
# from utils import language_model_preprocessing, translated_preprocessing
from config import BATCH_SIZE, MAX_LEN, EPOCHS, patience, BERT_MODEL, RANDOM_SEED, project_root_path
from src.data.create_dataset import ekman_dataset
from src.features.preprocess_feature_creation import create_dataloaders_BERT_w_token_type
import torch.nn as nn
# from src.helpers import format_time, set_seed
import gc
from src.pytorchtools import EarlyStopping

emotions = ['sadness', 'fear', 'surprise', 'joy', 'disgust', 'anger']
num_labels = len(emotions)
tokenizer_bert = AutoTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
roBERTa_MODEL = "xlm-roberta-base"
tokenizer_roberta = AutoTokenizer.from_pretrained(roBERTa_MODEL, do_lower_case=True)

gc.collect()

device = torch.device('cuda')
print('GPU:', torch.cuda.get_device_name(0))
loss_fn = nn.CrossEntropyLoss()
D_in, H, D_out = 768, 50, num_labels
classifier = nn.Sequential(
    nn.Linear(D_in, H),
    nn.Sigmoid(),
    # nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(H, D_out))
classifier.to(device)


def train_model(bert_model, dataloader_train, optimizer, scheduler, device):
    """The architecture's training routine."""

    bert_model.train()
    loss_train_total = 0


    for batch_idx, batch in enumerate(dataloader_train):
        gc.collect()

        # set gradient to 0
        bert_model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {
            "input_ids": batch[0],
            "token_type_ids": batch[1],
            "attention_mask": batch[2],
            "labels": batch[3],
        }

        output = bert_model(
            inputs["input_ids"],
            token_type_ids=inputs["token_type_ids"],
            attention_mask=inputs["attention_mask"],
            # labels=inputs["labels"],
            return_dict=False,
            output_hidden_states=True
        )
        # print(output)
        output = output[1][-1][:, 0, :]

        logits = classifier(output)
        loss = loss_fn(logits, inputs["labels"])

        # Compute train loss
        loss_train_total += loss.item()

        loss.backward()

        # gradient accumulation
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (batch_idx % 100 == 0 and batch_idx != 0) or (batch_idx == len(dataloader_train) - 1):
            print(batch_idx)

    # torch.save(bert_model.state_dict(), f"models/ BERT_ft_epoch{epoch}.model")
    loss_train_avg = loss_train_total / len(dataloader_train)

    return loss_train_avg


def evaluate_model(dataloader_val, bert_model, device):
    """The architecture's evaluation routine."""

    # evaluation mode
    bert_model.eval()

    # tracking variables
    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        # load into GPU
        batch = tuple(b.to(device) for b in batch)

        # define inputs
        inputs = {
            "input_ids": batch[0],
            "token_type_ids": batch[1],
            "attention_mask": batch[2],
            "labels": batch[3],
        }

        # compute logits
        with torch.no_grad():
            # loss, logits = bert_model(
            #     inputs["input_ids"],
            #     token_type_ids=inputs["token_type_ids"],
            #     attention_mask=inputs["attention_mask"],
            #     labels=inputs["labels"],
            #     return_dict=False,
            # )
            output = bert_model(
                inputs["input_ids"],
                token_type_ids=inputs["token_type_ids"],
                attention_mask=inputs["attention_mask"],
                # labels=inputs["labels"],
                return_dict=False,
                output_hidden_states=True
            )[1][-1][:,0,:]

            logits = classifier(output)
            loss = loss_fn(logits, inputs["labels"])

        # Compute validation loss
        loss_val_total += loss.item()

        # compute accuracy
        logits = logits.detach().cpu().numpy()
        label_ids = inputs["labels"].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    # compute average loss
    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


def test_model(dataloader_test, bert_model, device):
    """The architecture's test routine."""

    bert_model.eval()
    predictions, true_vals = [], []

    for batch in dataloader_test:
        # load into GPU
        batch = tuple(b.to(device) for b in batch)

        # define inputs
        inputs = {
            "input_ids": batch[0],
            "token_type_ids": batch[1],
            "attention_mask": batch[2],
            "labels": batch[3],
        }

        # compute logitsq
        with torch.no_grad():
            # _, logits = bert_model(
            #     inputs["input_ids"],
            #     token_type_ids=inputs["token_type_ids"],
            #     attention_mask=inputs["attention_mask"],
            #     labels=inputs["labels"],
            #     return_dict=False,
            # )
            output = bert_model(
                inputs["input_ids"],
                token_type_ids=inputs["token_type_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                return_dict=False,
                output_hidden_states=True
            )

            logits = classifier(output)
            # loss = loss_fn(logits, inputs["labels"])

        # compute accuracy
        logits = logits.detach().cpu().numpy()
        label_ids = inputs["labels"].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return predictions, true_vals


def f1_score_func(preds, labels):
    """Calculates the macro F1-score."""

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average="macro")


def accuracy_per_class(preds, labels):
    """Calculates the accuracy per class."""

    # make prediction
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    print(confusion_matrix(labels_flat, preds_flat))
    print(classification_report(labels_flat, preds_flat))

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f"Class: {label}")
        print(f"Accuracy:{len(y_preds[y_preds == label])}/{len(y_true)}\n")


def initialize_bert_model(train_dataloader, epochs=EPOCHS):
    """
    Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertForSequenceClassification.from_pretrained(
        BERT_MODEL,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=True,
    )

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,  # Default learning rate
                      eps=1e-8,  # Default epsilon value
                      weight_decay=0.1
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0.1 * total_steps,  # Default value
                                                num_training_steps=total_steps)

    return bert_classifier, optimizer, scheduler


def initialize_roberta_model(train_dataloader, epochs=EPOCHS):
    """
    Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    roberta_classifier = BertForSequenceClassification.from_pretrained(
        roBERTa_MODEL,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=True,
    )

    # Tell PyTorch to run the model on GPU
    roberta_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(roberta_classifier.parameters(),
                      lr=5e-5,  # Default learning rate
                      eps=1e-8,  # Default epsilon value
                      weight_decay=0.1
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0.1 * total_steps,  # Default value
                                                num_training_steps=total_steps)

    return roberta_classifier, optimizer, scheduler


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = ekman_dataset(RANDOM_SEED, oversampling=True)

    # BERTforSeqClass

    train_dataloader_bert = create_dataloaders_BERT_w_token_type(X_train, y_train, tokenizer_bert, MAX_LEN, BATCH_SIZE,
                                                                 sampler='random')
    val_dataloader_bert = create_dataloaders_BERT_w_token_type(X_val, y_val, tokenizer_bert, MAX_LEN, BATCH_SIZE,
                                                               sampler='sequential')
    test_dataloader_bert = create_dataloaders_BERT_w_token_type(X_test, y_test, tokenizer_bert, MAX_LEN, BATCH_SIZE,
                                                                sampler='sequential')
    bert_classifier, bert_optimizer, bert_scheduler = initialize_bert_model(train_dataloader_bert, EPOCHS)

    # XLM Roberta
    train_dataloader_roberta = create_dataloaders_BERT_w_token_type(X_train, y_train, tokenizer_roberta, MAX_LEN,
                                                                    BATCH_SIZE, sampler='random')
    val_dataloader_roberta = create_dataloaders_BERT_w_token_type(X_val, y_val, tokenizer_roberta, MAX_LEN, BATCH_SIZE,
                                                                  sampler='sequential')
    test_dataloader_roberta = create_dataloaders_BERT_w_token_type(X_test, y_test, tokenizer_roberta, MAX_LEN,
                                                                   BATCH_SIZE, sampler='sequential')

    roberta_classifier, roberta_optimizer, roberta_scheduler = initialize_roberta_model(train_dataloader_bert, EPOCHS)

    print(f"Training phase ...")
    early_stopping_bert = EarlyStopping(patience=patience, verbose=True, path='bert_checkpoint.pt')
    early_stopping_roberta = EarlyStopping(patience=patience, verbose=True, path='roberta_checkpoint.pt')
    for epoch in range(1, EPOCHS + 1):
        train_loss_bert = train_model(bert_classifier, train_dataloader_bert, optimizer=bert_optimizer,
                                      scheduler=bert_scheduler, device=device)
        train_loss_roberta = train_model(roberta_classifier, train_dataloader_roberta, optimizer=roberta_optimizer,
                                         scheduler=roberta_scheduler, device=device)

        val_loss_bert, predictions_bert, true_vals = evaluate_model(val_dataloader_bert, bert_classifier, device)
        val_loss_roberta, predictions_roberta, true_vals = evaluate_model(val_dataloader_roberta, roberta_classifier,
                                                                          device)

        ensemble_predictions = (predictions_bert + predictions_roberta) / 2.0
        val_f1 = f1_score_func(ensemble_predictions, true_vals)
        print(
            f"Epoch #{epoch} - bert_t_loss {train_loss_bert} - bert_v_loss {val_loss_bert} - roberta_t_loss {train_loss_roberta} - roberta_v_loss {val_loss_roberta} mean_v_f1 {val_f1}"
        )

        early_stopping_bert(val_loss_bert, bert_classifier)
        early_stopping_roberta(val_loss_roberta, roberta_classifier)

        if early_stopping_bert.early_stop & early_stopping_roberta.early_stop:
            print("Early stopping")
            break

    bert_classifier.load_state_dict(torch.load('bert_checkpoint.pt'))
    roberta_classifier.load_state_dict(torch.load('roberta_checkpoint.pt'))

    predictions_bert, true_vals = test_model(test_dataloader_bert, bert_classifier, device)
    predictions_roberta, true_vals = test_model(test_dataloader_roberta, roberta_classifier, device)

    ensemble_predictions = (predictions_bert + predictions_roberta) / 2.0
    print(accuracy_per_class(ensemble_predictions, true_vals))


if __name__ == '__main__':
    main()
