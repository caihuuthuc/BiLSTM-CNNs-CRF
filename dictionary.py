import numpy as np
import re
from math import sqrt
from update_tag_scheme import update_tag_scheme
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

LIMIT_LENGTH_OF_SENTENCES = 1000
GLOVE_FILENAME = "/home/huuthuc/Downloads/glove.6B/glove.6B.100d.txt"

class Dictionary():
    def __init__(self):
        self.vocabs = list()
        self.word2idx = dict()
        
    
    
    def add_word(self, word):
        if word not in self.vocabs:
            self.vocabs.append(word)
            self.word2idx[word] = len(self.vocabs) - 1
    

    def to_id(self, word):
        if word in self.vocabs:
            return self.word2idx[word]
        else:
            return 0


    def to_word(self, idx):
        if idx < len(self.vocabs):
            # print(idx, '\t', len(self.vocabs))
            return self.vocabs[idx]
        else:
            return self.vocabs[0]


    def get_word_embedding(self, embedd_size=300):
        n_words = len(self.vocabs)
        return np.random.uniform(-sqrt(3.0/embedd_size), sqrt(3.0/embedd_size), size=(n_words, embedd_size))


    def get_word_embedding_from_glove(self, glove_fn=GLOVE_FILENAME):
        glove_file = datapath(glove_fn)
        glove_in_w2v_format_fn = "glove_in_word2vec_format.txt"
        tmp_file = get_tmpfile(glove_in_w2v_format_fn)
        _ = glove2word2vec(glove_file, tmp_file)
        pretrained_w2v = KeyedVectors.load_word2vec_format(tmp_file)
        
        n_words = len(self.vocabs)
        embedd_size = pretrained_w2v['hello'].shape[0]
        embedd_matrix = np.random.uniform(-sqrt(3.0/embedd_size), sqrt(3.0/embedd_size), size=(n_words, embedd_size))

        for word, idx in self.word2idx.items():
            if word in pretrained_w2v:
                embedd_matrix[idx, :] = pretrained_w2v[word]
        
        return embedd_matrix


    def __len__(self):
        return len(self.vocabs)


class Corpus():
    def __init__(self):
        train_fn = './data/eng.train'
        dev_fn = './data/eng.testa'
        test_fn = './data/eng.testb'

        self.train_sentences, self.train_labels, self.train_sequence_lengths = self.read_tsv(train_fn)
        self.dev_sentences, self.dev_labels, self.dev_sequence_lengths = self.read_tsv(dev_fn)
        self.test_sentences, self.test_labels, self.test_sequence_lengths = self.read_tsv(test_fn)
        
        self.word_dictionary = self.create_dictionary(self.train_sentences, word_dictionary=True)
        self.label_dictionary = self.create_dictionary(self.train_labels)
        self.char_dict = self.generate_char_dict()
        self.max_word_len = 30


    def read_tsv(self, filename):
        sentences = []
        labels = []
        sequence_lengths = []
        sent = []
        label = []
        leng = 0

        with open(filename) as fin:
            for line in fin:

                row = re.split(" ", line.strip())
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
                    label.append(row[-1].strip())
                    leng += 1
        update_tag_scheme(labels, sequence_lengths)
        return sentences, labels, sequence_lengths
    

    def create_dictionary(self, data, word_dictionary=False):
        dictionary = Dictionary()
        if word_dictionary:
            dictionary.add_word('UNK') # for unknowkn words
        for row in data:
            for value in row:
                dictionary.add_word(value)
        return dictionary

    
    def generate_char_dict(self):
        char_dict = {}
        alphabet = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{}'
        for i,c in enumerate(alphabet):
            char_dict[c] = i
        return char_dict

    
if __name__ == '__main__':
    corpus = Corpus()
    a = corpus.word_dictionary.get_word_embedding_from_glove()