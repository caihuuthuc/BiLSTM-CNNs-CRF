import numpy as np
import nltk
from math import sqrt
from Task1_datahelper import word_indices_to_char_indices
def get_feed_dict(sentence, max_word_len, char_dict, lookup_table, map_id_word, map_word_id):
    sentence = nltk.word_tokenize(sentence)
    sentence_in_indices = np.zeros(shape=(1, len(sentence)), dtype=np.int32)

    label = [0]*len(sentence)
    label = np.array(label).reshape(1, -1)
    dims = lookup_table.shape[1]

    vectors = np.zeros(shape=(1, len(sentence), dims), dtype=np.float32)
    new_idx = lookup_table.shape[0]
    map_word_id_new = dict()
    map_id_word_new = dict()
    for k in map_id_word:
        map_id_word_new[k] = map_id_word[k]
    for k in map_word_id:
        map_word_id_new[k] = map_word_id[k]

    for idx, w in enumerate(sentence):
        if w in map_word_id:
            vectors[0, idx, :] = lookup_table[map_word_id_new[w], :]
        else:
            vectors[0, idx, :] = np.random.uniform(-sqrt(3.0/dims), sqrt(3.0/dims), dims)
            map_id_word_new[new_idx] = w
            map_word_id_new[w] = new_idx
            new_idx += 1
        sentence_in_indices[0, idx] = map_word_id_new[w]
    sequence_length = np.array([len(sentence)])
    max_sentences_length_in_batch = len(sentence)

    
    chars_indices = word_indices_to_char_indices(sentence_in_indices, sequence_length, max_sentences_length_in_batch, max_word_len, char_dict, map_id_word_new)
    
    feed_dict = {
            "labels_placeholder": label,
            "vectors": vectors,
            "sequence_lengths_placeholder": sequence_length,
            "chars_placeholder": chars_indices,
            "max_sentences_length_placeholder": max_sentences_length_in_batch,
            "dropout_prob_placeholder": 1.0
        }
    return sentence, feed_dict