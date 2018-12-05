import os
import csv
import numpy as np
import shelve

import numpy as np
import re
from time import time
from math import sqrt
import tensorflow as tf
import sys
from lxml import etree
from POS_datahelper import encode_labels, generate_lookup_word_embedding, word_indices_to_char_indices, load_from_file

def encode_sentences_for_testing(sentences, map_word_id):
    encoded_sentences = np.zeros(shape=[sentences.shape[0]], dtype=object)
    for idx, sent in enumerate(sentences):
        tmp = np.zeros(shape=[len(sent)], dtype=np.int32)
        for subidx, word in enumerate(sent):
            w = word.lower()
            idx_of_word = map_word_id[w]
            tmp[subidx] = idx_of_word
        encoded_sentences[idx] = tmp
    return encoded_sentences


def update_lookup_table_for_testing(test_sentences, lookup_table, map_id_word, map_word_id):
    n_out_of_vocabs = 0
    out_of_vocabs = list()
    old_n_vocabs = len(map_word_id.keys())
    oov_map_word_id = dict()
    
    for sent in test_sentences:
        for word in sent:
            
            w = word.lower()
            if w not in map_word_id:
                oov_map_word_id[w] = n_out_of_vocabs
                n_out_of_vocabs += 1
                out_of_vocabs.append(w)

    #update map_id_word and map_word_id
    for idx, w in enumerate(out_of_vocabs):
        map_word_id[w] = idx + old_n_vocabs
        map_id_word[idx + old_n_vocabs] = w

    #update lookup table
    dims = lookup_table.shape[1]
    updated_lookup_table = np.zeros(shape=[lookup_table.shape[0] + n_out_of_vocabs, dims])

    oov_lookup_table = generate_lookup_word_embedding(out_of_vocabs, oov_map_word_id)
    for idx in range(lookup_table.shape[0]):
        updated_lookup_table[idx, :] = lookup_table[idx, :]

    for idx in range(oov_lookup_table.shape[0]):
        updated_lookup_table[idx + old_n_vocabs, :] = oov_lookup_table[idx, :]

    return updated_lookup_table


def get_feed_dict_for_testting(test_sentences, test_labels, test_sequence_lengths, max_word_len, char_dict, labels_template, updated_lookup_table, map_id_word, map_word_id):

    dims = updated_lookup_table.shape[1]

    encoded_test_labels = encode_labels(test_labels, labels_template)
    encoded_test_sentences = encode_sentences_for_testing(test_sentences, map_word_id)

    for idx in range(test_sentences.shape[0]):
        
        sent = encoded_test_sentences[idx].reshape(1, -1)
        label = encoded_test_labels[idx].reshape(1, -1)
        sequence_length = np.array([test_sequence_lengths[idx]])

        max_sentences_length_in_batch = sent.shape[1]

        vectors = np.zeros(shape=(1, max_sentences_length_in_batch, dims), dtype=np.float32)

        for subidx, word in enumerate(test_sentences[idx]):
            w = word.lower()
            id_of_word = map_word_id[w]
            vectors[0, subidx, :] = updated_lookup_table[id_of_word, :]

        chars_indices = word_indices_to_char_indices(sent, sequence_length, max_sentences_length_in_batch, max_word_len, char_dict, map_id_word)
        feed_dict = {
                "labels_placeholder": label,
                "vectors": vectors,
                "sequence_lengths_placeholder": sequence_length,
                "chars_placeholder": chars_indices,
                "max_sentences_length_placeholder": max_sentences_length_in_batch,
                "dropout_prob_placeholder": 1.0
                }

        yield feed_dict



if __name__ == '__main__':
    type_embeddings = sys.argv[1].strip()
    tagging = sys.argv[2].strip()
    type_w_emb = sys.argv[3].strip()
    phrase = sys.argv[4].strip()


    assert phrase in ['dev', 'test', 'debug']
    assert job in ['prepare-data', 'tsv-to-xml', 'stat']
    print()
    print("Type embeddings: %s " % type_embeddings)
    print("Tagging: %s" % tagging)
    print('Variant word embedding: %s' % type_w_emb)
    print('Phrase: %s' % phrase)
    print('Job: %s' % job)

    thong_ke(phrase)


