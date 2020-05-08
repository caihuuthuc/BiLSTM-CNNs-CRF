import os
import pickle
import numpy as np
import tensorflow as tf
import sys
import tempfile
from time import time
from math import sqrt
from datahelper import batch_iter
from dictionary import Corpus
from evaluator import Evaluator

class Network():
    def __init__(self, corpus):
        self.corpus = corpus
        self.n_train_samples = len(self.corpus.train_sentences)
        self.char_embedding_size = 16
        self.char_representation_size = 30
        self.max_word_len = self.corpus.max_word_len
        self.num_classes = len(self.corpus.label_dictionary)
        self.hidden_size_lstm = 200
        self.dropout_prob = 0.5
        self.n_epochs = 60
        self.batch_size = 256
        self.n_batches = int(self.n_train_samples//self.batch_size) + 1
        self.learning_rate = 0.015
        self.momentum = 0.9
        self.gradient_limit = 5.0

        self.chars_ph= tf.placeholder(
                tf.int32, 
                shape=[None, None, self.max_word_len], 
                name="characters_ph")
        self.sentences_ph = tf.placeholder(
                tf.int32, 
                shape=[None, None], 
                name='sentences_ph')
        self.labels_ph = tf.placeholder(
                tf.int32, 
                shape=[None, None], 
                name='labels_ph')
        self.sequence_lengths_ph = tf.placeholder(
                tf.int32, 
                shape=[None], 
                name='lengths_ph')
        self.dropout_prob_ph = tf.placeholder_with_default(1.0, shape=(), name='dropout_ph')
        self.lr_placeholer = tf.placeholder(tf.float64, name='lr_ph')
        self.max_sentences_length_ph = tf.placeholder(tf.int32, name='max_sentences_length_in_batch_ph')

        with tf.device("/device:gpu:0"), tf.variable_scope('char-embedding', reuse=tf.AUTO_REUSE):
            bound_value = sqrt(3.0/self.char_embedding_size)
            const_init_value = np.random.uniform(-bound_value, bound_value)
            embedd_shape = (len(self.corpus.char_dict.keys()), self.char_embedding_size)
            W_char_embedding = tf.get_variable(name="char-embedding", \
                                        initializer=tf.constant_initializer(const_init_value),\
                                        shape=embedd_shape, \
                                        trainable=True)
            char_embedding = tf.nn.embedding_lookup(W_char_embedding, self.chars_ph)
            self.char_embedding = tf.nn.dropout(char_embedding, self.dropout_prob_ph)

        with tf.device("/device:gpu:0"), tf.variable_scope('char-cnn', reuse=tf.AUTO_REUSE):
            window_size = 3
            bound_value = sqrt(6.0/(self.char_embedding_size + self.char_representation_size))
            filter_shape = (1, window_size, self.char_embedding_size, self.char_representation_size)
            initilizer = tf.constant_initializer(np.random.uniform(-bound_value, bound_value))
            W = tf.get_variable(name='W', shape=filter_shape, initializer=initilizer)
            conv = tf.nn.conv2d(self.char_embedding, W, strides=[1, 1, 1, 1], padding='SAME', name='conv')
            relu = tf.nn.relu(conv, name='relu')
            pool = tf.nn.max_pool(relu, ksize=[1, 1, self.max_word_len, 1], 
                              strides = [1, 1, self.max_word_len, 1], padding='VALID', name='pool')
            self.char_representation = tf.reshape(pool, [-1, self.max_sentences_length_ph, self.char_representation_size])

        with tf.device("/device:gpu:0"), tf.variable_scope('word-embedding'):
            word_embedd_init = self.corpus.word_dictionary.get_word_embedding_from_glove()
            W_embedding = tf.Variable(name='word-embedding', initial_value=word_embedd_init, dtype=tf.float32, trainable=False)
            vectors = tf.nn.embedding_lookup(W_embedding, self.sentences_ph, name='vectors')
            word_embedding_with_char_representation = tf.concat([vectors, self.char_representation], axis=2)
            self.word_embedding_with_char_representation = tf.nn.dropout(
                    word_embedding_with_char_representation, 
                    self.dropout_prob_ph)

        with tf.device("/device:gpu:0"), tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn( \
                                cell_fw, cell_bw, self.word_embedding_with_char_representation, \
                                sequence_length=self.sequence_lengths_ph, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            self.output = tf.nn.dropout(output, self.dropout_prob_ph)

        with tf.device("/device:gpu:0"), tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
            r_plus_c = 2*self.hidden_size_lstm + self.num_classes
            bound_value = sqrt(6.0/r_plus_c)
            const_init_value = np.random.uniform(-bound_value, bound_value)
            W_shape = (2*self.hidden_size_lstm, self.num_classes)
            W = tf.get_variable("W", \
                    dtype=tf.float32, \
                    initializer=tf.constant_initializer(const_init_value),\
                    shape=W_shape)
            b = tf.get_variable("b", \
                    shape=(self.num_classes), \
                    dtype=tf.float32, \
                    initializer=tf.constant_initializer(0.0))
            output = tf.reshape(self.output, (-1, 2*self.hidden_size_lstm))
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, (-1, self.max_sentences_length_ph, self.num_classes))

        with tf.device("/device:gpu:0"), tf.name_scope('crf_encode'):
            log_likelihood, self.trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, 
                    self.labels_ph, 
                    self.sequence_lengths_ph)
            self.loss = tf.reduce_mean(-log_likelihood)

        with tf.device("/device:gpu:0"), tf.name_scope('crf_decode'):
            self.viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
                    self.logits, 
                    self.trans_params, 
                    self.sequence_lengths_ph)

        with tf.device("/device:gpu:0"), tf.name_scope('optimizer'):
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum)
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -self.gradient_limit, self.gradient_limit), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs)

    
    def train(self):
        config = tf.ConfigProto(allow_soft_placement = True)
        with tf.Session(config = config) as self.sess:
            self.sess.run( tf.global_variables_initializer())
            print("Training: Start")
            step = 0
            batches = batch_iter(
                    self.corpus, 
                    batch_size=self.batch_size, 
                    n_epochs=self.n_epochs)
            
            
            saved_model_folder = './saved_model/'
            if os.path.isdir(saved_model_folder) == False:
                os.mkdir(saved_model_folder)
            if os.path.isdir('./saved_model/ner') == False:
                os.mkdir('./saved_model/ner')
            
            saver = tf.train.Saver()
            timer = time()            
            for step, batch in enumerate(batches): 
                sent_batch, char_batch, label_batch, sequence_length_batch, max_sentences_length_in_batch = batch
                feed_dict = {
                        self.dropout_prob_ph: self.dropout_prob,
                        self.sentences_ph: sent_batch, 
                        self.labels_ph: label_batch, 
                        self.sequence_lengths_ph: sequence_length_batch,
                        self.max_sentences_length_ph: max_sentences_length_in_batch,
                        self.chars_ph: char_batch}
                # print(sent_batch.shape)
                # print(char_batch.shape)
                # print(label_batch.shape)
                # print(sequence_length_batch.shape)
                loss_, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
                
                if (step+1) % 100 == 0 or step >= self.n_batches*self.n_epochs - 1:
                    print("Step %d/%d Loss: %f" % (step, self.n_batches*self.n_epochs, loss_), end=' ')
                    print('Took %fs' % (time() - timer))
                    timer = time()
                    saver.save(self.sess, './saved_model/ner/ckpt', global_step=step)
                    
                    if (step+1) % 500 == 0:
                        self.dev()
                        print('Took %fs' % (time() - timer))
                        timer = time()
                    
                # break

            print()    
            print("Training: Done")


    def dev(self):
        f_gold = tempfile.NamedTemporaryFile('w')
        f_guess = tempfile.NamedTemporaryFile('w')
        for batch in batch_iter(self.corpus, phase="test"):
            gold_sents, gold_chars, gold_labels, gold_seqlens, max_seqlens = batch
            feed_dict = {
                    self.dropout_prob_ph: 1.0,
                    self.sentences_ph: gold_sents, 
                    self.labels_ph: gold_labels, 
                    self.sequence_lengths_ph: gold_seqlens,
                    self.max_sentences_length_ph: max_seqlens,
                    self.chars_ph: gold_chars}
            guess_labels = self.sess.run(self.viterbi_sequence, feed_dict=feed_dict)
            for gold_sent, gold_label, gold_seqlen in zip(gold_sents, gold_labels, gold_seqlens):
                np_words = gold_sent[:gold_seqlen]
                np_labels = gold_label[:gold_seqlen]
                for np_word, np_label in zip(np_words, np_labels):
                    word = self.corpus.word_dictionary.to_word(np_word)
                    label = self.corpus.label_dictionary.to_word(np_label)
                
                    f_gold.write('%s\t%s\n' % (word, label))
                f_gold.write('\n')
            
            for gold_sent, guess_label, gold_seqlen in zip(gold_sents, guess_labels, gold_seqlens):
                np_words = gold_sent[:gold_seqlen]
                np_labels = guess_label[:gold_seqlen]
                for np_word, np_label in zip(np_words, np_labels):
                    word = self.corpus.word_dictionary.to_word(np_word)
                    label = self.corpus.label_dictionary.to_word(np_label)
                
                    f_guess.write('%s\t%s\n' % (word, label))
                f_guess.write('\n')

        evaluater = Evaluator(f_gold.name, f_guess.name)
        evaluater.evaluate("strict")
        f1 = evaluater.result[0][-1]
        print("Test evaluate: F1 score: %.2f" % f1)


if __name__ == '__main__':
    corpus = Corpus()
    with open('resources.pickle', 'wb') as fout:
        pickle.dump(corpus, fout)
    network = Network(Corpus())
    network.train()
