import os
import csv
import numpy as np
import pickle

import numpy as np
import re
from time import time
from math import sqrt
from dictionary import Dictionary, Corpus



def batch_iter(corpus,
        phase="train",
        batch_size=32,
        n_epochs=30):

    phase_lookup = {
        "train": {
            "sentences": corpus.train_sentences,
            "labels": corpus.train_labels,
            "sequence_lengths": corpus.train_sequence_lengths,
            "batch_size": batch_size
        },
        "dev": {
            "sentences": corpus.dev_sentences,
            "labels": corpus.dev_labels,
            "sequence_lengths": corpus.dev_sequence_lengths,
            "batch_size": batch_size
        },
        "test": {
            "sentences": corpus.test_sentences,
            "labels": corpus.test_labels,
            "sequence_lengths": corpus.test_sequence_lengths,
            "batch_size": batch_size
        },
    }

    n_samples = len(phase_lookup[phase]["sentences"])
    batch_size = phase_lookup[phase]["batch_size"]
    n_batches_per_epoch = int((n_samples-1)/batch_size) + 1
    np_sentences = np.array(phase_lookup[phase]["sentences"])
    np_labels = np.array(phase_lookup[phase]["labels"])
    np_sequence_lengths = np.array(phase_lookup[phase]["sequence_lengths"])

    for _ in range(n_epochs):
        #shuffle
        shuffle_indices = np.random.permutation(np.arange(n_samples))
        np_sentences = np_sentences[shuffle_indices]
        np_labels = np_labels[shuffle_indices]
        np_sequence_lengths = np_sequence_lengths[shuffle_indices]

        for batch_num in range(n_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, n_samples)
            # print(start_index, end_index, n_samples, sep='\t')
            max_sentences_length_in_batch = max([length for length in np_sequence_lengths[start_index:end_index]])
            
            actual_batch_size = min(batch_size, end_index - start_index)
            padded_np_sentences = np.zeros(shape=[actual_batch_size, max_sentences_length_in_batch], dtype=np.int32)
            padded_np_labels = np.zeros(shape=[actual_batch_size, max_sentences_length_in_batch], dtype=np.int32)
            
            # print(start_index, ' ', end_index, ' ', end_index - start_index)
            for idx in range(min(batch_size, end_index - start_index)):
                padded_char_indices = word_indices_to_char_indices(
                    np_sentences[start_index:end_index],
                    max_sentences_length_in_batch,
                    corpus.max_word_len,
                    corpus.char_dict
                )
                for subidx in range(max_sentences_length_in_batch):
                    if subidx < np_sequence_lengths[start_index + idx]:
                        np_word = np_sentences[start_index + idx][subidx]
                        np_word = corpus.word_dictionary.to_id(np_word)
                        padded_np_sentences[idx][subidx] = np_word

                        np_label = np_labels[start_index + idx][subidx]
                        np_label = corpus.label_dictionary.to_id(np_label)
                        padded_np_labels[idx][subidx] = np_label
                        
            yield ( padded_np_sentences, 
                    padded_char_indices, 
                    padded_np_labels, 
                    np_sequence_lengths[start_index:end_index], 
                    max_sentences_length_in_batch )


def word_indices_to_char_indices(
        sentences,
        max_doc_len, 
        max_word_len, 
        char_dict):
    batch_size = sentences.shape[0]
    res = np.zeros(shape=[batch_size, max_doc_len, max_word_len], dtype=np.int32)
    for idx_sent, sentence in enumerate(sentences):
        for idx_word, word in enumerate(sentence):
            char_indices = word_2_indices_per_char(word, max_word_len, char_dict)
            res[idx_sent, idx_word, :] = char_indices
    return res


def word_2_indices_per_char(word, max_word_len, char_dict):
    data = np.zeros(max_word_len, dtype=np.int32)
    rest = max_word_len - len(word)
    for i in range(0, min(len(word), max_word_len)):
        if word[i] in char_dict:
            data[i + rest//2] = char_dict[word[i]]
    return data


if __name__ == '__main__':
    corpus = Corpus()
    print(len(corpus.word_dictionary.vocabs))
    print(corpus.word_dictionary.get_word_embedding().shape)
    print(len(corpus.label_dictionary.vocabs))

    for batch in batch_iter(corpus, n_epochs=1):
        sents, chars, labels, seqlens, maxlen_batch = batch
        


        break
