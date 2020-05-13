import os
import pickle
import numpy as np
import tensorflow as tf

from datahelper import batch_iter
from dictionary import Corpus

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

with open('resources.pickle', 'rb') as fin:
    corpus = pickle.load(fin)

if __name__ == '__main__':
    tf_sess = tf.Session()
    saver = tf.train.import_meta_graph('./saved_model/ner/ckpt-112299.meta')
    saver.restore(tf_sess, tf.train.latest_checkpoint('../saved_model/ner/'))
    graph = tf.get_default_graph()


