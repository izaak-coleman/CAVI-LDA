"""Module for parsing and filtering the word count file."""
import pandas as pd
import numpy as np
VOCAB_SIZE = 10473


def filter_docs(wc, sw_threshold = 0.05, top_n_words=1000):
    """Removes stopwords, then takes top_n_words most frequent words."""
    # convert wc matrix to presence absence
    n_docs_containing_word = (wc > 0).sum(axis=0)

    # determine stopwords, stopword assigned 0, otherwise assigned 1
    n_docs = float(wc.shape[0])
    stop_words = (n_docs_containing_word / n_docs) < sw_threshold

    # remove stopwords by multipling stopword vector by
    # the wc. This effectively zeros out stopwords
    wc = wc.mul(stop_words)

    # remove infrequent words
    word_freq = wc.sum(axis=0)  # columnwise sum counts occurence of word over all docs
    top_words = word_freq.argsort(kind='stable')[-top_n_words:]
    wc = wc.mul(wc.columns.isin(top_words))  # zero out all infrequent word columns

    # drop all zero rows. Removes all stopwords and infrequent words
    wc = wc.loc[:, (wc != 0).any(axis=0)]
    return wc


def parse_to_docs(wc_file):
    " Turns the ap.dat into a word count matrix"

    # count the number of docs
    print('Counting number of docs...')
    with open(wc_file) as f:
        n_docs = sum([1 for l in f])

    # load word count data
    print('Loading word counts into df...')
    wc = pd.DataFrame(np.full((n_docs, VOCAB_SIZE), 0)) # docs x vocab size 0 filled array
    with open(wc_file) as f:
        for i, data in enumerate(f):
            data = data.split()
            data.pop(0) # get rid of the number of words
            for record in data:
                word, count = record.split(':')
                wc.iloc[i, int(word)] = int(count)
    return wc

