import sys

sys.path.append("./")
import json
from sklearn.model_selection import train_test_split
from src.config import BATCH_SIZE, MAX_LEN, EPOCHS, patience, BERT_MODEL, RANDOM_SEED, project_root_path, lamda, cov, \
    num_labels, alpha, mc_dropout_rate, basic_dropout_rate, weight_decay
import torch
from transformers import BertModel, BertTokenizer, XLMRobertaTokenizer, \
    XLMRobertaModel  # ,BertForSequenceClassification
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
from src.data.create_dataset import ekman_dataset_categorical, ekman_dataset
from src.models.BERT_selective_prediction import BERT_selective_pred

# sys.path.append(project_root_path)
gc.collect()

device = torch.device('cuda')
print('GPU:', torch.cuda.get_device_name(0))

# loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.NLLLoss()
# loss_fn = nn.BCELoss()
loss_fn = nn.BCEWithLogitsLoss()
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
tokenizer.save_pretrained(project_root_path + "/models/tokenizer_simple/")


def rcc_auc(conf, risk):
    # risk-coverage curve's area under curve
    n = len(conf)
    cr_pair = list(zip(conf, risk))
    cr_pair.sort(key=lambda x: x[0], reverse=True)

    cumulative_risk = [cr_pair[0][1]]
    for i in range(1, n):
        cumulative_risk.append(cr_pair[i][1] + cumulative_risk[-1])

    points_x = []
    points_y = []

    auc = 0
    for k in range(n):
        auc += cumulative_risk[k] / (1 + k)
        points_x.append((1 + k) / n)  # coverage
        points_y.append(cumulative_risk[k] / (1 + k))  # current avg. risk

    return auc


def rpp(conf, risk):
    # reverse pair proportion
    # for now only works when risk is binary
    n = len(conf)
    cr_pair = list(zip(conf, risk))
    cr_pair.sort(key=lambda x: x[0], reverse=False)

    pos_count, rp_count = 0, 0
    for i in range(n):
        # print(f'{i}\t{cr_pair[i][1]==0}\t{pos_count}\t{rp_count}')
        if cr_pair[i][1] == 0:  # risk==0
            pos_count += 1
        else:
            rp_count += pos_count

    return rp_count / (n ** 2)


def selective_loss(logits, y_true):
    """
    Two options: one if in the dataset there is a class for "No emotion" and another if there isn't
    """
    # r`(f, g|Sm) = loss_fn(y_pred[:,-1:].repeat(1,num_labels)*y_true[:, :], y_pred[:, :-1]) = rl(f,g|Sm)
    # ˆφ(g|Sm) = y_pred[:, -1].mean() (empirical coverage)
    # print(-y_pred[:, -1].mean())
    # print(max(-y_pred[:, -1].mean() + cov, 0))

    # y_pred[:, -1:].repeat(1, num_labels)
    # y_true = y_true.detach()
    # y_pred = y_pred.detach()
    # true = y_pred[:,-1:].repeat(1,num_labels)*y_true[:, :-1]
    # pred = y_pred[:, :-1]
    # pred2 = torch.softmax(y_pred, dim=1)
    # pred3 = torch.sigmoid(y_pred)
    # loss_f = loss_fn(y_pred[:, :-1], y_true.float())
    # risk = loss_f * torch.gt(y_pred[:, -1], 0.5).bool().int() #torch.le()


    if y_true.shape[1] != 7:
        #if there is no class for "no emotion"
        probs = torch.softmax(logits, dim=1)
        probs_emo = torch.sum(probs[:, :-1], keepdim=True, dim=1)
        # x = probs_emo.repeat(1, num_labels) * y_true

        loss = loss_fn(logits[:, :-1], probs_emo.repeat(1, num_labels) * y_true) \
               + lamda * max(-probs_emo.mean() + cov, 0) ** 2

        # loss = loss_fn(logits[:, :-1], logits[:,-1:].repeat(1,num_labels)*y_true[:, :-1]) \
        #     + lamda * max(-logits[:, -1].mean() + cov, 0) ** 2

    else:

        probs = torch.softmax(logits, dim=1)  # last is no emo
        # y = torch.sum(probs[:,:-1], keepdim=True, dim=1)
        # w = probs[:,-1:]
        y_probs_2 = torch.cat((torch.sum(probs[:, :-1], keepdim=True, dim=1), probs[:, -1:]),
                              dim=1)  # binary (emo or no emo)
        y_true_2 = torch.cat((torch.sum(y_true[:, :-1], keepdim=True, dim=1), y_true[:, -1:]),
                             dim=1)  # binary emo or no emo

        # cv = y_probs_2[:, 0].mean()
        # cv3 = y_probs_2[:, :-1].mean()
        # cv2 = logits[:,-1].mean()

        loss_2 = loss_fn(y_probs_2, y_true_2)
        loss = loss_2 + lamda * max(-y_probs_2[:, 0].mean() + cov, 0) ** 2  # max of emo

    # loss = loss_fn(y_pred[:, :-1], y_true) \
    #        + lamda * max(-y_pred[:, -1].mean() + cov, 0) ** 2

    # loss_s = risk + lamda * max(-y_pred[:, -1].mean() + cov, 0) ** 2
    # loss_s = torch.mean(loss_s)

    # ce_loss = categorical_ce_with_logits(y_pred[:, :-1], y_pred[:,-1:].repeat(1,num_labels)*y_true[:, :]) + lamda * max(-y_pred[:, -1].mean() + cov, 0) ** 2

    return loss


def selective_acc(logits, y_true):
    # same function as calc_selective_risk
    # how many of the selected to predict items are predicted correctly

    # covered_idx = torch.gt(y_pred[:, -1], 0.5).bool().int()
    # preds = torch.argmax(y_pred[:, :-1], dim=1)
    # labels = torch.argmax(y_true, dim=1)
    # temp1 = sum((covered_idx) * torch.eq(labels, preds))#.int()
    # acc_with_select = temp1 / sum(covered_idx)

    probs = torch.softmax(logits, dim=1)
    covered_idx = (probs[:, -1] < 0.5).cpu()  # checks if no emo is below 0.5...

    y_hat = torch.argmax(logits[:, :-1], 1)
    correct_preds_from_selected = y_hat[covered_idx] == torch.argmax(y_true[covered_idx, :], 1)
    acc = sum(correct_preds_from_selected) / sum(covered_idx)
    #
    # print(y_pred)
    # print(y_true)
    # covered_idx2 = (torch.argmax(y_pred,1)!=6).cpu()
    # correct_preds_from_selected2 = y_hat[covered_idx2] == torch.argmax(y_true[covered_idx2, :], 1)
    # acc2 = sum(correct_preds_from_selected2)/sum(covered_idx2)
    # print(acc2)

    return acc


def coverage(y_logits):
    probs = torch.softmax(y_logits, dim=1)
    covered_idx = (probs[:, -1] < 0.5).cpu()
    return covered_idx.type(torch.FloatTensor).mean()


def initialize_model(train_dataloader, epochs=EPOCHS):
    """
    Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    # classifier = model() #freeze_bert=False)
    classifier = BERT_selective_pred.EmoEkman(freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(classifier.parameters(),
                      lr=1e-5,  # Default learning rate
                      eps=1e-8,  # Default epsilon value
                      weight_decay=weight_decay
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)

    """ 
    TODO
    learning_rate = 0.1

    lr_decay = 1e-6

    lr_drop = 25

    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)


    # optimization details
    sgd = optimizers.SGD(learning_rate=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    """

    return classifier, optimizer, scheduler


def validation(model, val_dataloader, weights=None):
    """
    After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_selection_accuracy = []
    val_loss = []
    val_coverage = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits_selection, logits_classification = model(b_input_ids, b_attn_mask)

        # Compute loss and accumulate the loss values
        # b_labels = b_labels.type(torch.LongTensor).to(device)

        # Compute loss

        b_labels = b_labels.type(torch.LongTensor).to(device)
        loss_select = selective_loss(logits_selection, b_labels.float())

        if b_labels.shape[1] == 7:
            loss_class = loss_fn(logits_classification, b_labels[:, :-1].float())
        else:
            loss_class = loss_fn(logits_classification, b_labels.float())

        # loss_class = loss_fn(logits_classification, b_labels[:, :-1].float())
        loss = alpha * loss_select + (1 - alpha) * loss_class
        val_loss.append(loss.item())

        # Get the predictions
        # preds = torch.argmax(logits, dim=1).flatten()
        select_accuracy = selective_acc(logits_selection, b_labels)

        preds = torch.argmax(logits_classification, dim=1).flatten()
        b_labels = torch.argmax(b_labels, dim=1).flatten()
        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100

        covrge = coverage(logits_selection)

        val_accuracy.append(accuracy)
        val_selection_accuracy.append(select_accuracy.item() * 100)
        val_coverage.append(covrge)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_selection_accuracy = np.mean(val_selection_accuracy)
    val_accuracy = np.mean(val_accuracy)
    val_coverage = np.mean(val_coverage)

    return val_loss, val_selection_accuracy, val_accuracy, val_coverage


def calc_selective_risk(logits, y_true, calibrated_coverage=None):
    """
    @param logits: The logits of the selective model
    @param y_true: The real values
    @param calibrated_coverage: Whether or not to predict the calibrated coverage (and how much)
    @return: The metrics to evaluate the selective model (selective_risk, selective_accuracy, coverage (for the non calibrated coverage),
     rcc_auc_value, rpp_value f1 and the classif_report
    """
    y_true = torch.Tensor(y_true.values).cpu() #[:, :-1]  #### TODO
    probs_all = torch.softmax(logits, dim=1)

    if calibrated_coverage is None:
        threshold = 0.5
    else:
        threshold = np.percentile(probs_all[:, -1].cpu(), 100 * calibrated_coverage)
        print("the threshold is: {}".format(threshold))

    covered_idx = (probs_all[:,
                   -1] < threshold).cpu()  # checks no emo below threshold - therefore there is emo and predicts
    preds = torch.argmax(logits[:, :-1], 1).cpu()

    risk = preds[covered_idx] != torch.argmax(y_true[covered_idx, :],
                                              1)  # error rate for selectively predicting the class, error rate is the number of predictions made that were not correct
    coverage = covered_idx.type(torch.FloatTensor).mean()
    selective_risk = risk.sum() / covered_idx.sum()

    probs = torch.softmax(logits, dim=1)[:, :-1]
    conf, _ = torch.max(probs[covered_idx], dim=1, keepdim=True)

    rcc_auc_value = rcc_auc(conf, risk)
    rpp_value = rpp(conf, risk)

    accuracy = preds[covered_idx] == torch.argmax(y_true[covered_idx, :], 1)
    selective_accuracy = accuracy.sum() / covered_idx.sum()

    # print(" ")
    print("The selective risk is: {}, the selective accuracy is: {} and the Coverage is {}".format(selective_risk,
                                                                                                   selective_accuracy,
                                                                                                   coverage))

    accuracy2 = accuracy_score(torch.argmax(y_true[covered_idx, :], 1), preds[covered_idx])

    f1 = f1_score(torch.argmax(y_true[covered_idx, :], 1), preds[covered_idx], average="macro")
    classif_rep = classification_report(torch.argmax(y_true[covered_idx, :], 1), preds[covered_idx], zero_division=0)
    return [selective_risk.item(), selective_accuracy.item(), accuracy2, coverage.item(), rcc_auc_value.item(),
            rpp_value, f1], classif_rep


def mc_dropout(model, x_test, iter=50):
    print("MC dropout predict")
    # K.set_value(mc_dropout_rate, dropout)
    repititions = []
    for i in range(iter):
        _, logits = bert_predict(x_test, y_test=None, mc=True, batch_size=100, model=model)
        pred = torch.softmax(logits, dim=1)
        repititions.append(pred.cpu().numpy())

    # K.set_value(mc_dropout_rate, 0)

    repititions = np.array(repititions)
    mc = np.var(repititions, 0)
    mc = np.mean(mc, -1)
    return -mc


def selective_risk_at_coverage(model, x_test, y_test, coverage, mc=False):
    _, logits = bert_predict(x_test)
    probs = torch.softmax(logits, dim=1).cpu()

    if mc:
        sr = mc_dropout(model, x_test)
        sr = torch.from_numpy(sr)
    else:
        sr, _ = torch.max(probs, dim=1, keepdim=False)

    sr_sorted, _ = torch.sort(sr)
    threshold = sr_sorted[probs.shape[0] - int(coverage * probs.shape[0])]
    print("The threshold is: {}".format(threshold))
    covered_idx = (sr > threshold).cpu()
    labels = torch.tensor(y_test.values).cpu()
    preds_covidx = torch.argmax(probs[covered_idx], 1)

    selective_acc = preds_covidx == torch.argmax(labels[covered_idx], 1)

    conf, _ = torch.max(probs[covered_idx], dim=1, keepdim=True)

    risk = preds_covidx != torch.argmax(labels[covered_idx], 1)
    rcc_auc_value = rcc_auc(conf, risk)
    rpp_value = rpp(conf, risk)

    selective_risk = risk.sum() / covered_idx.sum()
    selective_accuracy = selective_acc.sum() / covered_idx.sum()

    accuracy2 = accuracy_score(torch.argmax(labels[covered_idx], 1), preds_covidx)
    f1 = f1_score(torch.argmax(labels[covered_idx], 1), preds_covidx, average="macro")
    classif_rep = classification_report(torch.argmax(labels[covered_idx], 1), preds_covidx, zero_division=0)

    return [selective_risk.item(), selective_accuracy.item(), accuracy2, coverage, rcc_auc_value.item(), rpp_value,
            f1], classif_rep


def train(model, train_dataloader, val_dataloader=None, optimizer=None, scheduler=None, epochs=4, evaluation=False,
          weights=None):
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
        print(
            f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Select Loss':^10} | {'Val Acc':^9} | {'Val Coverage':^9} | {'Elapsed':^9}",
            flush=True)
        print("-" * 70, flush=True)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        batch_select_loss, batch_class_loss = 0, 0
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
            logits_selection, logits_classification = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            b_labels = b_labels.type(torch.LongTensor).to(device)

            loss_select = selective_loss(logits_selection, b_labels.float())

            if b_labels.shape[1] == 7:

                loss_class = loss_fn(logits_classification, b_labels[:, :-1].float())
            else:
                loss_class = loss_fn(logits_classification, b_labels.float())
            ###
            # batch_select_loss += loss_select.item()
            # batch_class_loss += loss_class.item()
            ###

            loss = alpha * loss_select + (1 - alpha) * loss_class
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
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {'-':^9} | {'-':^9} | {time_elapsed_batch} | {time_elapsed_total}",
                    flush=True)

                # print("batch select loss: {}, batch classification loss: {}".format(batch_select_loss/batch_counts, batch_class_loss/batch_counts))

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
            val_loss, val_selection_accuracy, val_accuracy, covrg = validation(model, val_dataloader, weights)

            # Print performance over the entire training data
            time_elapsed = format_time(time.time() - t0_epoch)

            print(
                f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Select Acc':^10} | {'Val Acc':^9} | {'Val Coverage':^9} | {'Elapsed':^9}",
                flush=True)
            print("-" * 70, flush=True)
            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_selection_accuracy:^9.2f} | {val_accuracy:^9.2f} | {covrg:^9.2f} |  {time_elapsed}",
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


def bert_predict(X_test, y_test=None, mc=False, batch_size=1000, model=None):

    t0 = time.time()

    if mc:
        # If MC dropout, necessary to have dropout, therefore put the model in train mode
        model.train()
        model.mc_dropout_rate = basic_dropout_rate + 0.2
        batch = batch_size

    else:
        model = torch.jit.load(project_root_path + '/models/model_scripted_multiclass_emotion_selective.pt')
        model.eval()
        batch = BATCH_SIZE

    tokenizer = BertTokenizer.from_pretrained(project_root_path + "/models/tokenizer_simple/")
    test_dataloader = create_dataloaders_BERT(X_test, y_test, tokenizer, MAX_LEN, batch, sampler='sequential')
    all_logits_select = []
    all_logits_class = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits_select, logits_class = model(b_input_ids, b_attn_mask)

        all_logits_select.append(logits_select)
        all_logits_class.append(logits_class)

    # Concatenate logits from each batch
    all_logits_select = torch.cat(all_logits_select, dim=0)
    all_logits_class = torch.cat(all_logits_class, dim=0)

    # Apply softmax to calculate probabilities
    # probs = F.softmax(all_logits_select, dim=1).cpu().numpy()

    print(f"Total prediction time: {format_time(time.time() - t0)}", flush=True)

    return all_logits_select, all_logits_class


# def bert_predict_mc(model, X_test, y_test=None, batch_size=1000):
#     """
#     Perform a forward pass on the trained BERT model to predict probabilities
#     on the test set.
#     """
#     # Put the model into the evaluation mode. The dropout layers are disabled during
#     # the test time.
#     t0 = time.time()
#     # model = torch.load(project_root_path+'/models/multiclass_emotion.pt')
#     # model = torch.jit.load(project_root_path + '/models/model_scripted_multiclass_emotion_selective.pt')
#
#     model.train()
#     model.mc_dropout_rate = basic_dropout_rate + 0.2  # set mc dropout rate
#
#     tokenizer = BertTokenizer.from_pretrained(project_root_path + "/models/tokenizer_simple/")
#     test_dataloader = create_dataloaders_BERT(X_test, y_test, tokenizer, MAX_LEN, batch_size, sampler='sequential')
#     all_logits_select = []
#     all_logits_class = []
#
#     # For each batch in our test set...
#     for batch in test_dataloader:
#         # Load batch to GPU
#         b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
#
#         # Compute logits
#         with torch.no_grad():
#             logits_select, logits_class = model(b_input_ids, b_attn_mask)
#
#         all_logits_select.append(logits_select)
#         all_logits_class.append(logits_class)
#
#     # Concatenate logits from each batch
#     all_logits_select = torch.cat(all_logits_select, dim=0)
#     all_logits_class = torch.cat(all_logits_class, dim=0)
#
#     # Apply softmax to calculate probabilities
#     # probs = F.softmax(all_logits_select, dim=1).cpu().numpy()
#
#     model.mc_dropout_rate = 0  # reset mc_dropout rate
#
#     print(f"Total prediction time: {format_time(time.time() - t0)}", flush=True)
#
#     return all_logits_select, all_logits_class


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = ekman_dataset_categorical(RANDOM_SEED)
    # X_train, y_train, X_val, y_val, X_test, y_test, _ = ekman_dataset(RANDOM_SEED)

    train_dataloader = create_dataloaders_BERT(X_train, y_train, tokenizer, MAX_LEN, BATCH_SIZE, sampler='random')
    val_dataloader = create_dataloaders_BERT(X_val, y_val, tokenizer, MAX_LEN, BATCH_SIZE, sampler='sequential')

    set_seed(42)  # Set seed for reproducibility

    bert_classifier, optimizer, scheduler = initialize_model(train_dataloader, epochs=EPOCHS)
    train(bert_classifier, train_dataloader, val_dataloader, optimizer, scheduler, epochs=EPOCHS,
          evaluation=True)  # , weights=weights)

    for batch in train_dataloader:
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
        model_scripted = torch.jit.trace(bert_classifier, (b_input_ids, b_attn_mask))
        torch.jit.save(model_scripted, project_root_path + '/models/model_scripted_multiclass_emotion_selective.pt')
        break

    results_val = {}
    print(" ")
    print("---- Validation set ----", flush=True)
    print(" ")
    logits_select_val, logits_class_val = bert_predict(X_val)
    # print("Validation set selective risk")
    results_val["selective - calib"], class_rep_val_select_calib = calc_selective_risk(logits_select_val, y_val,
                                                                                      calibrated_coverage=cov)
    results_val["selective - not calib"], class_rep_val_select = calc_selective_risk(logits_select_val, y_val,
                                                                                      calibrated_coverage=None)

    results_val["MC"], class_rep_val_mc = selective_risk_at_coverage(bert_classifier, X_val, y_val, cov,
                                                                                  mc=True)

    results_val["SR"], class_rep_val_sr = selective_risk_at_coverage(bert_classifier, X_val, y_val, cov, mc=False)

    print(" ")
    print("---- Test set ----", flush=True)
    print(" ")
    results_test = {}
    logits_select_test, logits_class_test = bert_predict(X_test)
    # print("Test set selective risk")
    results_test["selective - calib"], class_rep_test_select_calib = calc_selective_risk(logits_select_test, y_test,
                                                                                 calibrated_coverage=cov)
    results_test["selective - not calib"], class_rep_test_select = calc_selective_risk(logits_select_test, y_test,
                                                                                 calibrated_coverage=None)

    results_test["MC"], class_rep_test_mc = selective_risk_at_coverage(bert_classifier, X_test, y_test, cov,
                                                                             mc=True)

    results_test["SR"], class_rep_test_sr = selective_risk_at_coverage(bert_classifier, X_test, y_test, cov, mc=False)

    results_val_df = pd.DataFrame.from_dict(results_val, orient='index',
                                        columns=['selective_risk', 'selective_accuracy', 'accuracy2', 'coverage',
                                                 'rcc_auc_value', 'rpp_value', 'f1'])
    results_test_df = pd.DataFrame.from_dict(results_test, orient='index',
                                        columns=['selective_risk', 'selective_accuracy', 'accuracy2', 'coverage',
                                                 'rcc_auc_value', 'rpp_value', 'f1'])
    print(" ---- Metrics ----")
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 5,
                           ):
        print("---- Validation ----")
        print(results_val_df)
        print("---- Test ----")
        print(results_test_df)

    print(" ")
    print(" ---- Classification Reports ----")
    print("---- Validation Classification reports ----")
    print(" ")
    print("---- Validation Selective prediction - calibrated ----")
    print(class_rep_val_select_calib)
    print(" ")
    print("---- Validation Selective prediction - not calibrated ----")
    print(class_rep_val_select)
    print(" ")
    print("---- Validation MC ----")
    print(class_rep_val_mc)
    print(" ")
    print("---- Validation SR ----")
    print(class_rep_val_sr)
    print(" ")
    print("---- Test Classification reports ----")
    print(" ")
    print("---- Test Selective prediction - calibrated ----")
    print(class_rep_test_select_calib)
    print(" ")
    print("---- Test Selective prediction - not calibrated ----")
    print(class_rep_test_select)
    print(" ")
    print("---- Test MC ----")
    print(class_rep_test_mc)
    print(" ")
    print("---- Test SR ----")
    print(class_rep_test_sr)


if __name__ == '__main__':
    main()
