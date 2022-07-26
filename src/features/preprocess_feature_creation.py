
import nrclex
import emoji
import contractions
import re
import torch
import json
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from src.config import project_root_path
from src.helpers import convert_to_unicode, run_strip_accents

import sys
# project_root_path = "C:/Users/georgiapant/PycharmProjects/GitHub/rebecca"
sys.path.append(project_root_path)


def fix_encoding(text):
    text = re.sub("[\x8b]", ' ', text)
    text = re.sub("[\xc2-\xf4][\x80-\xbf]+", lambda m: m.group(0).encode('latin1').decode('utf8'), text)
    text = convert_to_unicode(text.rstrip().lower())
    text = run_strip_accents(text)
    return text


def text_preprocessing(text):
    text = re.sub("[\xc2-\xf4][\x80-\xbf]+", lambda m: m.group(0).encode('latin1').decode('utf8'), text)
    text = convert_to_unicode(text.rstrip().lower())
    text = run_strip_accents(text)
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
    text = re.sub(r"_", " ", text)

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
    text = re.sub(r'https?:/\/.*[\r\n]*', '', text)
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)
    text = str(re.sub("\S*\d\S*", "", text).strip())

    return text


def preprocessing_for_bert(data, tokenizer, MAX_LEN):
    """Perform required preprocessing steps for pretrained BERT.
    @param MAX_LEN:
    @param data:
    @param tokenizer:
    @param    data =(list): Array of texts to be processed.
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

def preprocessing_for_bert_w_token_type(data, tokenizer, MAX_LEN):
    """Perform required preprocessing steps for pretrained BERT.
    @param MAX_LEN:
    @param data:
    @param tokenizer:
    @param    data =(list): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """

    # Create empty lists to store outputs
    input_ids = []
    token_type_ids = []
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
            return_attention_mask=True,  # Return attention mask
            return_token_type_ids=True

        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
        token_type_ids.append(encoded_sent.get('token_type_ids'))

    # Convert lists to tensors
    # input_ids = torch.tensor(input_ids)
    # attention_masks = torch.tensor(attention_masks)

    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, token_type_ids, attention_masks


def nrc_feats(input_ids, tokenizer):
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


def vad_feats(input_ids, tokenizer, MAX_LEN):
    arousal_dict = json.load(open(project_root_path+ "/data/external/vad/arousal_dict.json"))
    valence_dict = json.load(open(project_root_path+ "/data/external/vad/valence_dict.json"))
    dom_dict = json.load(open(project_root_path+ "/data/external/vad/dom_dict.json"))

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


def create_dataloaders_BERT(X, y, tokenizer, MAX_LEN, BATCH_SIZE, sampler='sequential'):
    print('Tokenizing data...', flush=True)
    inputs, masks = preprocessing_for_bert(X, tokenizer, MAX_LEN)
    print('Done', flush=True)

    if y is None:
        data = TensorDataset(inputs, masks)
    else:
        if y.shape[1] == 1:
            labels = torch.tensor(y) #.values)
        else:
            labels = torch.tensor(y.values)
        data = TensorDataset(inputs, masks, labels)

    if sampler == 'sequential':
        sampler = SequentialSampler(data)
    elif sampler == 'random':
        sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=BATCH_SIZE)

    return dataloader


def create_dataloaders_BERT_w_token_type(X, y, tokenizer, MAX_LEN, BATCH_SIZE, sampler='sequential'):
    print('Tokenizing data...', flush=True)
    inputs, token_type_ids, masks = preprocessing_for_bert_w_token_type(X, tokenizer, MAX_LEN)
    print('Done', flush=True)

    if y is None:
        data = TensorDataset(inputs, token_type_ids, masks)
    else:
        labels = torch.tensor(y) #.values)
        data = TensorDataset(inputs, token_type_ids, masks, labels)

    if sampler == 'sequential':
        sampler = SequentialSampler(data)
    elif sampler == 'random':
        sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=BATCH_SIZE)

    return dataloader