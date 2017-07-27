#!/usr/bin/env python

import numpy as np
import sys
sys.path.append('../../MLFlow/utils')
sys.path.append('../../TFModels/cnn_classifier')
import dataproc
import tensorflow as tf
from text_cnn import TextCNNClassifier
from operator import itemgetter


NCLASS = 1999
NWORDS = 411721
SEQ_LEN = 30
LABEL_REPR = 'dense'
MAX_ITER = 30000
BATCH_SIZE = 256
MODEL_CKPT_DIR = './model_ckpt/'


def inp_fn(data, word_tb):
    inp_x = []
    inp_y = []
    for inst in data:
        flds = inst.split('\t')
        label = flds[1]
        words = flds[3]
        word_ids = word_tb.lookup(words.split(','))
        feats = [x if x else 0 for x in word_ids]
        inp_x.append(dataproc.zero_padding(feats, SEQ_LEN))
        inp_y.append(label)
    return np.array(inp_x), inp_y


def get_top5(preds, proba=False):
    res = sorted(list(enumerate(preds)), key=itemgetter(1), reverse=True)[0:5]
    if proba:
        return res
    else:
        return [x[0] for x in res]

out_mode = 'prob' if len(sys.argv) > 1 and sys.argv[1] == 'prob' else 'cls'


label_tb = dataproc.DictTable('label_dict')
word_tb = dataproc.DictTable('word_dict')
test_file = '../data_test/data_test.tsv'
test_reader = dataproc.BatchReader(test_file, max_epoch=1)
if out_mode == 'cls':
    test_res = 'res_test/res_test.tsv'
else:
    test_res = 'res_test/res_test_prob.tsv'
test_writer = open(test_res, 'w')

# load model
mdl = TextCNNClassifier(
    seq_len=SEQ_LEN,
    emb_dim=256,
    nclass=NCLASS,
    vocab_size=NWORDS,
    filter_sizes=[3, 4, 5],
    nfilters=128,
    reg_lambda=0.0,
    label_repr=LABEL_REPR,
    lr=1e-3)
sess = tf.Session()
mdl.saver.restore(sess, tf.train.latest_checkpoint(MODEL_CKPT_DIR))

niter = 0
while niter < MAX_ITER:
    niter += 1
    batch_data = test_reader.get_batch(BATCH_SIZE)
    if not batch_data:
        break
    test_x, label = inp_fn(batch_data, word_tb)
    preds = mdl.predict_proba(sess, test_x)
    pred5 = map(get_top5, preds)
    pred5 = map(label_tb.lookup_rev, pred5)
    for t, p in zip(label, pred5):
        print >> test_writer, t + '\t' + ','.join(p)


sess.close()
test_writer.close()
