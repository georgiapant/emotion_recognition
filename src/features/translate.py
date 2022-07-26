from langdetect import detect, DetectorFactory
import torch
import gc
from easynmt import EasyNMT
from src.features.preprocess_feature_creation import text_preprocessing
import pandas as pd
import statistics
from collections import Counter

device = torch.device('cuda')
DetectorFactory.seed = 0


def translate(sentences, target_lang='en'):
    gc.collect()
    model = EasyNMT('opus-mt')
    sentences = sentences.apply(lambda row: text_preprocessing(row))
    translated_sentences = pd.Series()
    for sentence in sentences:
        try:
            language = detect(sentence)
            if language is not 'en':
                translated_sentence = model.translate_sentences(sentence, target_lang=target_lang, source_lang=language)
            else:
                translated_sentence = sentence

        except:
            translated_sentence=sentence

        translated_sentences = pd.concat([translated_sentences, pd.Series(translated_sentence)])

    return translated_sentences


def oversample_with_back_translation(train_data):
    class_count = Counter(train_data['label'])

    average_instances = statistics.mean(class_count.values())

    new_samples = pd.DataFrame(columns=['content', 'label'])
    for i in class_count.items():
        if i[1]< average_instances:
            original = train_data[train_data["label"]==i[0]]
            translated_to_de = translate(original['content'], target_lang='de')

            translate_to_en = translate(translated_to_de).to_frame().rename(columns={0:'content'})
            translate_to_en['label'] = i[0]
            new_samples = pd.concat([new_samples, translate_to_en])

    train_data = pd.concat([train_data, new_samples])

    # Free some resources from GPU
    del translate_to_en
    del translated_to_de
    del new_samples
    torch.cuda.empty_cache()

    return train_data


# def translate_swedish(sentence):
#     gc.collect()
#     # translate swedish to english
#     model = EasyNMT('opus-mt')
#
#     # tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-sv-en")
#     # model_sw_to_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-sv-en").to(device)
#     #
#     # input_ids = tokenizer(sentence, return_tensors="pt", add_special_tokens=True).input_ids.to(device).long()
#     # output_ids = model_sw_to_en.generate(input_ids)[0]
#     # translated_sentence = tokenizer.decode(output_ids, skip_special_tokens=True)
#     translated_sentence = model.translate_sentences(sentence, target_lang='en', source_lang='sv')
#
#     return translated_sentence