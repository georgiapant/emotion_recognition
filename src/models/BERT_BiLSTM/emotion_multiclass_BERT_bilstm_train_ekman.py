import sys
sys.path.append("./")
import json
from sklearn.model_selection import train_test_split
from src.config import BATCH_SIZE, MAX_LEN, EPOCHS, patience, BERT_MODEL, RANDOM_SEED, project_root_path
import torch
from transformers import BertModel, BertTokenizer,  XLMRobertaTokenizer, XLMRobertaModel  # ,BertForSequenceClassification
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import numpy as np
import gc
from collections import Counter
from src.features.preprocess_feature_creation import create_dataloaders_BERT
from src.pytorchtools import EarlyStopping
from src.helpers import format_time, set_seed
from src.data.create_dataset import ekman_dataset
from src.models.BERT_BiLSTM import BERT_bilstm

# sys.path.append(project_root_path)
gc.collect()

device = torch.device('cuda')
print('GPU:', torch.cuda.get_device_name(0))

loss_fn = nn.CrossEntropyLoss()
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
tokenizer.save_pretrained("../../models/tokenizer_simple/")


def initialize_model(train_dataloader, epochs=EPOCHS):
    """
    Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    # classifier = model() #freeze_bert=False)
    classifier = BERT_bilstm.BertClassifier(freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(classifier.parameters(),
                      lr=5e-5,  # Default learning rate
                      eps=1e-8  # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)

    return classifier, optimizer, scheduler


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
    print(f'Accuracy: {accuracy * 100:.2f}%', flush=True)
    print(f'F1-score macro: {f1_score(y_true, y_pred, average="macro")}', flush=True)
    print(f'Classification report: {classification_report(y_true, y_pred)}', flush=True)


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
        # os.remove('checkpoint.pt')
        print("\n")
    print(f"Total training time: {format_time(time.time() - t0)}", flush=True)

    # torch.save(model, "../../models/multiclass_emotion.pt")  # save model
    # torch.save(model, project_root_path+"/models/multiclass_emotion.h5")  # save model

    # torch.save(model.state_dict(),  project_root_path+"/models/multiclass_emotion.pt")

    print("Training complete!", flush=True)


def bert_predict(X_test, y_test=None):
    """
    Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    t0 = time.time()
    # model = torch.load(project_root_path+'/models/multiclass_emotion.pt')
    model = torch.jit.load(project_root_path + '/models/model_scripted_multiclass_emotion_simple.pt')
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(project_root_path+"/models/tokenizer_simple/")
    test_dataloader = create_dataloaders_BERT(X_test, y_test, tokenizer, MAX_LEN, BATCH_SIZE, sampler='sequential')
    all_logits = []

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
    X_train, y_train, X_val, y_val, X_test, y_test = ekman_dataset(RANDOM_SEED)

    train_dataloader = create_dataloaders_BERT(X_train, y_train, tokenizer, MAX_LEN, BATCH_SIZE, sampler='random')
    val_dataloader = create_dataloaders_BERT(X_val, y_val, tokenizer, MAX_LEN, BATCH_SIZE, sampler='sequential')

    set_seed(42)  # Set seed for reproducibility

    bert_classifier, optimizer, scheduler = initialize_model(train_dataloader, epochs=EPOCHS)
    train(bert_classifier, train_dataloader, val_dataloader, optimizer, scheduler, epochs=EPOCHS, evaluation=True)

    for batch in train_dataloader:
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
        model_scripted = torch.jit.trace(bert_classifier, (b_input_ids, b_attn_mask))
        torch.jit.save(model_scripted, project_root_path+'/models/model_scripted_multiclass_emotion_simple.pt')
        break

    # model_scripted = torch.jit.script(bert_classifier)  # Export to TorchScript
    # model_scripted.save(project_root_path+'/models/model_scripted_multiclass_emotion.pt')

    print("Validation set", flush=True)
    probs_val = bert_predict(X_val)
    evaluate(probs_val, y_val)

    print("Test set", flush=True)

    probs_test = bert_predict(X_test)
    evaluate(probs_test, y_test)

if __name__ == '__main__':
    main()
