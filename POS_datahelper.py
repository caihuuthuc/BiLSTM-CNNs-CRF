import os
import csv
import numpy as np
import shelve

import numpy as np
import re
from time import time
from math import sqrt

LIMIT_LENGTH_OF_SENTENCES = 1000

def readFileTSV(filename):
    sentences = []
    labels = []
    sequence_lengths = []
    sent = []
    label = []
    leng = 0

    with open(filename) as f:
        for _, line in enumerate(f.readlines()):
            row = line.strip()
            row = re.split(" ", row)
            if len(row) == 1:
                if leng > 0 and leng <= LIMIT_LENGTH_OF_SENTENCES:
                    sentences.append(sent)
                    labels.append(label)
                    sequence_lengths.append(leng)
                leng = 0
                label = []
                sent = []
            elif row[0] != '-DOCSTART-':
                word = row[0].lower().strip()
                sent.append(word)
                label.append(row[1].strip())
                leng += 1

    return np.array(sentences), np.array(labels), np.array(sequence_lengths)

def get_vocabs(sentences):
    vocabs = []
    for sent in sentences:
        for word in sent:
            if word not in vocabs:
                vocabs.append(word)
    vocabs.append('')
    return vocabs

def get_map_word_id(vocabs):
    map_word_id = dict()
    map_id_word = dict()
    for id_, word in enumerate(vocabs):
        map_word_id[word] = id_
        map_id_word[id_] = word
    return map_word_id, map_id_word

def generate_char_dict():
    char_dict = {}
    alphabet = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{}'
    for i,c in enumerate(alphabet):
        char_dict[c] = i
    return char_dict

def word_2_indices_per_char(word, max_word_len, char_dict):
    data = np.zeros(max_word_len, dtype=np.int32)
    rest = max_word_len - len(word)
    for i in range(0, len(word)):
        if i >= max_word_len:
            break
        elif word[i] in char_dict:
            data[i + rest//2] = char_dict[word[i]]
        else:
            # unknown character set to be 0
            data[i + rest//2] = 0
    return data

def generate_lookup_word_embedding(vocabs, map_word_id):
    def get_pretrained_word2vec():

        filename = 'glove.6B.100d.txt'
        from gensim.scripts.glove2word2vec import glove2word2vec
        from gensim.test.utils import get_tmpfile
        from gensim.models.keyedvectors import KeyedVectors
        tmp_file = get_tmpfile("glove_to_word2vec.txt")
        glove2word2vec(filename, tmp_file)
        pretrained_word2vec = KeyedVectors.load_word2vec_format(tmp_file, binary=False)
        return pretrained_word2vec
    
    pretrained_word2vec = get_pretrained_word2vec()
    dims = 100
    n_vocabs = len(vocabs)

    lookup_table = np.zeros(shape=[n_vocabs, dims])
    for _, word in enumerate(vocabs):
        idx = map_word_id[word]
        if word in pretrained_word2vec:
            lookup_table[idx, :] = pretrained_word2vec[word]
        else:
            lookup_table[idx, :] = np.random.uniform(-sqrt(3.0/dims), sqrt(3.0/dims), dims)

    return lookup_table

def encode_sentences(sentences, map_word_id):
    encoded_sentences = np.zeros(shape=[sentences.shape[0]], dtype=object)
    for idx, sent in enumerate(sentences):
        tmp = np.zeros(shape=[len(sent)], dtype=np.int32)
        for subidx, word in enumerate(sent):
            idx_of_word = map_word_id[word]
            tmp[subidx] = idx_of_word
        encoded_sentences[idx] = tmp
    return encoded_sentences
    
def get_labels_template(labels):
    labels_template = []
    labels_template.append('PAD')
    for label in labels:
        for l in label:
            if l not in labels_template:
                labels_template.append(l)
    print(labels_template)
    return labels_template

def encode_labels(labels, labels_template):
    labels_encoded = np.zeros(shape=[labels.shape[0]], dtype=object)
    for idx, label in enumerate(labels):
        tmp = np.zeros(shape=[len(label)], dtype=np.int32)
        for subidx, sublabel in enumerate(label):
            tmp[subidx] = labels_template.index(sublabel)
        labels_encoded[idx] = tmp
    return labels_encoded

    
def batch_iter(sentences, labels, sequence_lengths, idx_of_word_pad, idx_of_label_pad, batch_size=32, num_epochs=1000, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    n_samples = sentences.shape[0]
    num_batches_per_epoch = int((n_samples-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(n_samples))

            sentences = sentences[shuffle_indices]
            labels = labels[shuffle_indices]
            sequence_lengths = sequence_lengths[shuffle_indices]
        
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, n_samples)

            max_sentences_length_in_batch = max([length for length in sequence_lengths[start_index:end_index]])

            actual_batch_size = min(batch_size, end_index - start_index)
            padded_sentences = np.zeros(shape=[actual_batch_size, max_sentences_length_in_batch], dtype=np.int32)
            padded_labels = np.zeros(shape=[actual_batch_size, max_sentences_length_in_batch], dtype=np.int32)

            # print(start_index, ' ', end_index, ' ', end_index - start_index)
            for idx in range(min(batch_size, end_index - start_index)):
                for subidx in range(max_sentences_length_in_batch):
                    if subidx < sequence_lengths[start_index + idx]:
                        padded_sentences[idx][subidx] = sentences[start_index + idx][subidx]
                        padded_labels[idx][subidx] = labels[start_index + idx][subidx]
                        
                    else:
                        padded_sentences[idx][subidx] = idx_of_word_pad
                        padded_labels[idx][subidx] = idx_of_label_pad
                    

            yield padded_sentences, padded_labels, sequence_lengths[start_index:end_index], max_sentences_length_in_batch


def get_word_from_idx(map_id_word, idx):
    return map_id_word[idx]

def word_indices_to_char_indices(sents, lengths, max_doc_len, max_word_len, char_dict, map_id_word):
    batch_size = sents.shape[0]
    res = np.zeros(shape=[batch_size, max_doc_len, max_word_len], dtype=np.int32)
    for idx_sent, sentence in enumerate(sents):
        for idx_word, word_indices in enumerate(sentence):
            if idx_word < lengths[idx_sent]:
                word = get_word_from_idx(map_id_word, word_indices)
                char_indices = word_2_indices_per_char(word, max_word_len, char_dict)
                res[idx_sent, idx_word, :] = char_indices
            else:
                break
    return res

def write_to_file(data, filename):
    with shelve.open(filename) as f:
        for k in data:
            f[k] = data[k]
    
def load_from_file(filename):
    data = dict()
    with shelve.open(filename) as f:
        for k in f:
            data[k] = f[k]
    return data


def next_lr(lr, p=0.05, t=100):
    return 1.0*lr/(1.0 + p*t)


if __name__ == '__main__':
    def preprocess():
        sentences, labels, sequence_lengths = readFileTSV('data/eng.train')
        # print(labels)
        vocabs = get_vocabs(sentences)
        map_word_id, map_id_word  = get_map_word_id(vocabs)
        for w in vocabs:    
            assert w in map_word_id
        char_dict = generate_char_dict()
   
        lookup_table = generate_lookup_word_embedding(vocabs, map_word_id)
        
        labels_template = get_labels_template(labels)
        encoded_labels = encode_labels(labels, labels_template) #chứa labels dạng số 
        encoded_sentences = encode_sentences(sentences, map_word_id)

        data = dict()
        
        data['labels_template'] = labels_template
        data['lookup_table'] = lookup_table
        data['train_sentences'] = encoded_sentences
        data['train_labels'] = encoded_labels
        data['train_sequence_lengths'] = sequence_lengths
        data['vocabs'] = vocabs
        data['map_word_id']= map_word_id
        data['map_id_word'] = map_id_word

        data['char_dict'] = char_dict
        data['max_word_len'] = 30

        write_to_file(data, "./data_feed_model_POS.shlv")


    preprocess()

