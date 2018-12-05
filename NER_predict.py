import os
import shelve
import numpy as np
import tensorflow as tf
from time import time
from Task1_datahelper import readFileTSV, load_from_file, word_2_indices_per_char, get_word_from_idx, word_indices_to_char_indices, next_lr
from testing_datahelper import get_feed_dict_for_testting, update_lookup_table_for_testing

from math import sqrt
import sys
import re



if __name__ == '__main__':
    data = load_from_file("./data_feed_model_NER.shlv")

    labels_template = data['labels_template']

    map_word_id = data['map_word_id']
    map_id_word = data['map_id_word']
    char_dict = data['char_dict']
    max_word_len = data['max_word_len']


    
    config = tf.ConfigProto(allow_soft_placement = True)
    with tf.Session(config = config) as sess:
      
        saver = tf.train.import_meta_graph('./saved_model/ner/ckpt.meta' )
    
        with tf.device("/device:gpu:0"):
            print('Prepare data: Start')
            saver.restore(sess, tf.train.latest_checkpoint('./saved_model/ner/'))
            graph = tf.get_default_graph()
            lookup_table = sess.run(graph.get_tensor_by_name('word-embedding/word-embedding:0'))

            chars_placeholder = graph.get_tensor_by_name("characters:0")
            labels_placeholder = graph.get_tensor_by_name('labels:0')
            sequence_lengths_placeholder = graph.get_tensor_by_name('lengths:0')
            dropout_prob_placeholder = graph.get_tensor_by_name('dropout:0')
            max_sentences_length_in_batch = graph.get_tensor_by_name('max_sentences_length_in_batch:0')
            vectors = graph.get_tensor_by_name('word-embedding/vectors:0')
            viterbi_sequence = graph.get_tensor_by_name("crf_decode/cond/Merge:0")
        
            sentences, _, _ = readFileTSV('data/eng.testb')

            updated_lookup_table = update_lookup_table_for_testing(sentences, lookup_table, map_id_word, map_word_id)
            del lookup_table, sentences
            print('Prepare data: Done - Start predict')
            
            sentences, labels, sequence_lengths = readFileTSV('data/eng.testb')
            tsvfile = open('predict.tsv', 'w')

                
            feed_dicts = get_feed_dict_for_testting(sentences, labels, sequence_lengths, max_word_len, char_dict, labels_template, updated_lookup_table, map_id_word, map_word_id)
            for idx, fd in enumerate(feed_dicts):
                feed_dict = {
                    chars_placeholder: fd['chars_placeholder'],
                    labels_placeholder: fd['labels_placeholder'],
                    sequence_lengths_placeholder: fd['sequence_lengths_placeholder'],
                    dropout_prob_placeholder: fd['dropout_prob_placeholder'],
                    vectors: fd['vectors'],
                    max_sentences_length_in_batch: fd['sequence_lengths_placeholder'][0] 
                }
                
                predict = sess.run(viterbi_sequence, feed_dict=feed_dict)

                for subidx in range(sequence_lengths[idx]):

                    word = sentences[idx][subidx]
                    golden_tag = labels[idx][subidx]
                    predict_tag = labels_template[predict[0][subidx]]

                    tsvfile.write("%s\t%s\t%s\n" % (word, golden_tag, predict_tag))
                tsvfile.write('\n')

            tsvfile.close()
            del updated_lookup_table

    print('Done')
                


    

    

