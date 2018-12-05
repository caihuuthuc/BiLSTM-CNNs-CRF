import grpc
from concurrent import futures
import time
import sys
sys.path.append('../task1/code')
from Task1_datahelper import load_from_file, word_indices_to_char_indices
from server_helper import get_feed_dict
from math import sqrt
# import the generated classes
import calculator_pb2
import calculator_pb2_grpc

import tensorflow as tf
import os

import nltk
nltk.download('punkt')
import numpy as np

# import the original calculator.py
# import calculator

# create a class to define the server functions, derived from
# calculator_pb2_grpc.CalculatorServicer

def square_root(a):
    return a**2

def run_model_NER(sentence):
    config = tf.ConfigProto(allow_soft_placement = True)

    sess_NER = tf.Session(config = config)
    saver_NER = tf.train.import_meta_graph('./saved_model/ner/ckpt.meta')
    global map_word_id_NER
    global map_id_word_NER
    global char_dict_NER
    global labels_template_NER
    global max_word_len_POS
    
    with tf.device("/device:gpu:0"):
        
        saver_NER.restore(sess_NER, tf.train.latest_checkpoint('saved_model/ner'))
        graph = tf.get_default_graph()
    
        with tf.device("/device:gpu:0"):        
            lookup_table = sess_NER.run(graph.get_tensor_by_name('word-embedding/word-embedding:0'))

            chars_placeholder = graph.get_tensor_by_name("characters:0")
            labels_placeholder = graph.get_tensor_by_name('labels:0')
            sequence_lengths_placeholder = graph.get_tensor_by_name('lengths:0')
            dropout_prob_placeholder = graph.get_tensor_by_name('dropout:0')
            max_sentences_length_in_batch = graph.get_tensor_by_name('max_sentences_length_in_batch:0')
            vectors = graph.get_tensor_by_name('word-embedding/vectors:0')
            viterbi_sequence = graph.get_tensor_by_name("crf_decode/cond/Merge:0")

            tokens, fd = get_feed_dict(sentence, max_word_len_NER, char_dict_NER, lookup_table, map_id_word_NER, map_word_id_NER)

            feed_dict = {
                    chars_placeholder: fd['chars_placeholder'],
                    labels_placeholder: fd['labels_placeholder'],
                    sequence_lengths_placeholder: fd['sequence_lengths_placeholder'],
                    dropout_prob_placeholder: fd['dropout_prob_placeholder'],
                    vectors: fd['vectors'],
                    max_sentences_length_in_batch: fd['sequence_lengths_placeholder'][0] 
                    }
    
            predict = sess_NER.run(viterbi_sequence, feed_dict=feed_dict)
    del sess_NER, saver_NER, graph
    return ' '.join([w for w in tokens]),' '.join([labels_template_NER[w] for w in predict[0]])

def run_model_POS(sentence):
    config = tf.ConfigProto(allow_soft_placement = True)

    sess_POS = tf.Session(config = config)
    saver_POS = tf.train.import_meta_graph('./saved_model/pos/ckpt.meta')
    global map_word_id_POS
    global map_id_word_POS
    global char_dict_POS
    global labels_template_POS
    global max_word_len_POS
    with tf.device("/device:gpu:0"):
        
        saver_POS.restore(sess_POS, tf.train.latest_checkpoint('saved_model/pos'))
        graph = tf.get_default_graph()
    
        with tf.device("/device:gpu:0"):        
            lookup_table = sess_POS.run(graph.get_tensor_by_name('word-embedding/word-embedding:0'))

            chars_placeholder = graph.get_tensor_by_name("characters:0")
            labels_placeholder = graph.get_tensor_by_name('labels:0')
            sequence_lengths_placeholder = graph.get_tensor_by_name('lengths:0')
            dropout_prob_placeholder = graph.get_tensor_by_name('dropout:0')
            max_sentences_length_in_batch = graph.get_tensor_by_name('max_sentences_length_in_batch:0')
            vectors = graph.get_tensor_by_name('word-embedding/vectors:0')
            viterbi_sequence = graph.get_tensor_by_name("crf_decode/cond/Merge:0")

            tokens, fd = get_feed_dict(sentence, max_word_len_POS, char_dict_POS, lookup_table, map_id_word_POS, map_word_id_POS)

            feed_dict = {
                    chars_placeholder: fd['chars_placeholder'],
                    labels_placeholder: fd['labels_placeholder'],
                    sequence_lengths_placeholder: fd['sequence_lengths_placeholder'],
                    dropout_prob_placeholder: fd['dropout_prob_placeholder'],
                    vectors: fd['vectors'],
                    max_sentences_length_in_batch: fd['sequence_lengths_placeholder'][0] 
                    }
    
            predict = sess_POS.run(viterbi_sequence, feed_dict=feed_dict)
    del sess_POS, saver_POS, graph
    return ' '.join([w for w in tokens]),' '.join([labels_template_POS[w] for w in predict[0]])




class CalculatorServicer(calculator_pb2_grpc.CalculatorServicer):

    # calculator.square_root is exposed here
    # the request and response are of the data type
    # calculator_pb2.Number
    def SquareRoot(self, request, context):
        response = calculator_pb2.Number()
        response.value = square_root(request.value)
        return response

    def getNER(self, request, context):
        res = calculator_pb2.Res()
        res.token, res.label = run_model_NER(request.sentence)
        return res

    def getPOS(self, request, context):
        res = calculator_pb2.Res()
        res.token, res.label = run_model_POS(request.sentence)
        return res

if __name__ == '__main__':
    type_embeddings = 'glove'
    tagging = 'BIOES'

    data_NER = load_from_file("data_feed_model_NER.shlv")
    data_POS = load_from_file("data_feed_model_POS.shlv")

    labels_template_NER = data_NER['labels_template']
    map_word_id_NER = data_NER['map_word_id']
    map_id_word_NER = data_NER['map_id_word']
    char_dict_NER = data_NER['char_dict']
    max_word_len_NER = data_NER['max_word_len']

    labels_template_POS = data_POS['labels_template']
    map_word_id_POS = data_POS['map_word_id']
    map_id_word_POS = data_POS['map_id_word']
    char_dict_POS = data_POS['char_dict']
    max_word_len_POS = data_POS['max_word_len']

    
    # create a gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # use the generated function `add_CalculatorServicer_to_server`
    # to add the defined class to the server
    calculator_pb2_grpc.add_CalculatorServicer_to_server(
        CalculatorServicer(), server)

    

    # listen on port 50051
    print('Starting server. Listening on port 50051.')
    server.add_insecure_port('0.0.0.0:50051')
    server.start()

    # since server.start() will not block,
    # a sleep-loop is added to keep alive
        
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)