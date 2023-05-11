#!/usr/bin/python

""" 
utilities to aid data handling in the baseline CRF model.
"""

import pandas as pd
from itertools import chain
import nltk

nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import WordPunctTokenizer

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: list(
            zip(
                s['word'].values.tolist(),
                s['pos'].values.tolist(),
                s['tag'].values.tolist(),
            )
        )
        self.grouped = self.data.groupby('sentence_id').apply(agg_func)
        self.sentences = list(self.grouped)

    def get_next(self):
        try:
            s = self.grouped[f'Sentence: {self.n_sent}']
            self.n_sent += 1
            return s
        except Exception:
            return None

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2]
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features |= {
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        }
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features |= {
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        }
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

def flatten(y):
    """
    flattens a list of lists.
    from teamhg-memex/sklearn-crfsuite.
    """
    return list(chain.from_iterable(y))

def convert_prompt_to_df(text):
    """ 
    converts the text to SentenceGetter-parseable dataframe.
    See txt_to_df in preprocessing/src/utils.py for more info. 
    """

    sentence_number = 1
    sentence_number_list = []
    word_list = []
    pos_list = []
    tag_list = [] # this keeps the architecture intact but will not be used
    rolling_sentence_list = []
    previous_line_break = True

    for word in WordPunctTokenizer().tokenize(text):
        if word: # i.e. not line break
            sentence_number_list.append(str(sentence_number))
            word_list.append(word)
            tag_list.append('?')
            rolling_sentence_list.append(word)

            previous_line_break = False

        elif not previous_line_break:
            previous_line_break = True
            pos_list.extend(iter([x[1] for x in nltk.pos_tag(rolling_sentence_list)]))
            rolling_sentence_list = []
            sentence_number += 1
    # and again for the last sentence..
    pos_list.extend(iter([x[1] for x in nltk.pos_tag(rolling_sentence_list)]))
    return pd.DataFrame(
        {
            'sentence_id': sentence_number_list,
            'word': word_list, 
            'pos': pos_list,
            'tag': tag_list
        }
    )

def prepare_prompt(text):
    text_df = convert_prompt_to_df(text)
    getter_prompt = SentenceGetter(text_df)
    sentences_prompt = getter_prompt.sentences
    return [sent2features(s) for s in sentences_prompt]