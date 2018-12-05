import os
import shelve
import numpy as np
import tensorflow as tf
from time import time
from POS_datahelper import load_from_file, word_2_indices_per_char, batch_iter, get_word_from_idx, word_indices_to_char_indices, next_lr
from math import sqrt
import sys


data = load_from_file("./data_feed_model_POS.shlv")

max_word_len = data['max_word_len']
labels_template = data['labels_template']

train_sentences = data["train_sentences"]
train_labels = data["train_labels"]
train_sequence_lengths = data["train_sequence_lengths"]


word_lookup_table = data['lookup_table']
map_word_id = data['map_word_id']
map_id_word = data['map_id_word']
char_dict = data['char_dict']

char_embedding_size = 16
char_representation_size = 30
num_classes = len(labels_template)
hidden_size_lstm = 200
dropout_prob = 0.5
n_epochs = 50
batch_size = 10
n_batches = int(train_sentences.shape[0]//batch_size) + 1
learning_rate = 0.015
momentum = 0.9
gradient_limit = 5.0

label_train = dict()
label_dev = dict()


chars_placeholder = tf.placeholder(tf.int32, shape=[None, None, max_word_len], name="characters")
sentences_placeholder = tf.placeholder(tf.int32, shape=[None, None], name='sentences')
labels_placeholder = tf.placeholder(tf.int32, shape=[None, None], name='labels')
sequence_lengths_placeholder = tf.placeholder(tf.int32, shape=[None], name='lengths')
dropout_prob_placeholder = tf.placeholder_with_default(1.0, shape=(), name='dropout')
lr_placeholer = tf.placeholder(tf.float64, name='lr')
max_sentences_length_placeholder = tf.placeholder(tf.int32, name='max_sentences_length_in_batch')

with tf.device("/device:gpu:0"), tf.variable_scope('char-embedding', reuse=tf.AUTO_REUSE):
    W_char_embedding = tf.get_variable(name="char-embedding", \
                                        initializer=tf.constant_initializer(np.random.uniform(-sqrt(3.0/char_embedding_size), sqrt(3.0/char_embedding_size))),\
                                        shape=[len(char_dict.keys()), char_embedding_size], trainable=True)
    char_embedding = tf.nn.embedding_lookup(W_char_embedding, chars_placeholder)
    char_embedding = tf.nn.dropout(char_embedding, dropout_prob_placeholder)
with tf.device("/device:gpu:0"), tf.variable_scope('char-cnn', reuse=tf.AUTO_REUSE):
    window_size = 3

    filter_shape = [1, window_size, char_embedding_size, char_representation_size]
    initilizer = tf.constant_initializer(np.random.uniform(-sqrt(6.0/(char_embedding_size + char_representation_size)), sqrt(6.0/(char_embedding_size + char_representation_size))))
    W = tf.get_variable(name='W', shape=filter_shape, initializer=initilizer)
    conv = tf.nn.conv2d(char_embedding, W, strides=[1, 1, 1, 1], padding='SAME', name='conv')

    relu = tf.nn.relu(conv, name='relu')

    pool = tf.nn.max_pool(relu, ksize=[1, 1, max_word_len, 1], 
                              strides = [1, 1, max_word_len, 1], padding='VALID', name='pool')

    char_representation = tf.reshape(pool, [-1, max_sentences_length_placeholder, char_representation_size])

with tf.device("/device:gpu:0"), tf.variable_scope('word-embedding'):
    W_embedding = tf.Variable(name='word-embedding', initial_value=word_lookup_table, dtype=tf.float32, trainable=False)
    vectors = tf.nn.embedding_lookup(W_embedding, sentences_placeholder, name='vectors')
    word_embedding_with_char_representation = tf.concat([vectors, char_representation], axis=2)
    word_embedding_with_char_representation = tf.nn.dropout(word_embedding_with_char_representation, dropout_prob_placeholder)

with tf.device("/device:gpu:0"), tf.variable_scope("bi-lstm"):
    cell_fw = tf.contrib.rnn.LSTMCell(hidden_size_lstm)
    cell_bw = tf.contrib.rnn.LSTMCell(hidden_size_lstm)
    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn( \
                                cell_fw, cell_bw, word_embedding_with_char_representation, \
                                sequence_length=sequence_lengths_placeholder, dtype=tf.float32)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.nn.dropout(output, dropout_prob_placeholder)

with tf.device("/device:gpu:0"), tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
    r_plus_c = 2*hidden_size_lstm + num_classes
    W = tf.get_variable("W", dtype=tf.float32, initializer=tf.constant_initializer(np.random.uniform(-sqrt(6.0/r_plus_c), sqrt(6.0/r_plus_c))),\
                                                    shape=[2*hidden_size_lstm, num_classes])

    b = tf.get_variable("b", shape=[num_classes],dtype=tf.float32, initializer=tf.constant_initializer(0.0))

    output = tf.reshape(output, [-1, 2*hidden_size_lstm])
    pred = tf.matmul(output, W) + b
   
    logits = tf.reshape(pred, [-1, max_sentences_length_placeholder, num_classes])

with tf.device("/device:gpu:0"), tf.name_scope('crf_encode'):

    log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(logits, labels_placeholder, sequence_lengths_placeholder)
    loss = tf.reduce_mean(-log_likelihood)
with tf.device("/device:gpu:0"), tf.name_scope('crf_decode'):
    viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(logits, trans_params, sequence_lengths_placeholder)

with tf.device("/device:gpu:0"), tf.name_scope('optimizer'):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)

    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -gradient_limit, gradient_limit), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

config = tf.ConfigProto(allow_soft_placement = True)

with tf.Session(config = config) as sess:
    sess.run( tf.global_variables_initializer())
    print("Training: Start")

    step = 0
    batches = batch_iter(train_sentences, train_labels, train_sequence_lengths, map_word_id[''], labels_template.index('PAD'), batch_size=batch_size, num_epochs=n_epochs, shuffle=True)
    timer = time()
    for batch in batches: 
        sent_batch, label_batch, sequence_length_batch, max_sentences_length_in_batch = batch
        loss_, _, predicts = sess.run([loss, train_op, viterbi_sequence], feed_dict={
                                                                                dropout_prob_placeholder: dropout_prob,
                                                                                sentences_placeholder: sent_batch, 
                                                                                labels_placeholder: label_batch, 
                                                                                sequence_lengths_placeholder: sequence_length_batch,
                                                                                max_sentences_length_placeholder: max_sentences_length_in_batch,
                                                                                chars_placeholder: word_indices_to_char_indices(sent_batch, \
                                                                                        sequence_length_batch, max_sentences_length_in_batch, max_word_len, char_dict, map_id_word)
                                                                                
                                                                            })
        step += 1
        if step % n_batches == 0 or step >= n_batches*n_epochs - 1:
            print("Step %d/%d Loss: %f" % (step, n_batches*n_epochs, loss_), end=' ')
            print('Took %fs' % (time() - timer), end=' ')
            print('avg each step %fs' % ((time() - timer)/n_batches))
            timer = time()
        # break

    print()

    saver = tf.train.Saver()

    saved_model_folder = './saved_model/'
    if os.path.isdir(saved_model_folder) == False:
        os.mkdir(saved_model_folder)

    if os.path.isdir('./saved_model/pos') == False:
        os.mkdir('./saved_model/pos')
    saver.save(sess, './saved_model/pos/ckpt')
    
    print("Training: Done")
    