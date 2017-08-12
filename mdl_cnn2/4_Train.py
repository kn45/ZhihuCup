#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import sys
sys.path.append('../utils')
sys.path.append('../../TFModels/cnn_classifier')
import dataproc
from text_cnn import TextCNNClassifier

NCLASS = 1999
NWORDS = 411721
SEQ_LEN = 30
MAX_ITER = 30000
MODEL_CKPT_DIR = './model_ckpt/model.ckpt'

def inp_fn(data):
    inp_x = []
    inp_y = []
    for inst in data:
        flds = inst.split('\t')
        labels = map(int, flds[0].split(','))
        nlabel = len(labels)
        feats = map(int, flds[1].split(','))
        labels = dataproc.sparse2dense(labels, ndim=NCLASS)
        inp_y.append(labels / float(nlabel))
        inp_x.append(dataproc.zero_padding(feats, SEQ_LEN))
    return np.array(inp_x), np.array(inp_y)

train_file = 'feat_train/trnvld_feature.tsv'
freader = dataproc.BatchReader(train_file, max_epoch=1)

mdl = TextCNNClassifier(
    seq_len=SEQ_LEN,
    emb_dim=256,
    nclass=NCLASS,
    vocab_size=NWORDS,
    filter_sizes=[3, 4, 5],
    nfilters=128,
    reg_lambda=0.0,
    multi_label=True,
    lr=1e-3)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
metrics = ['loss']
niter = 0

while niter < MAX_ITER:
    niter += 1
    batch_data = freader.get_batch(256)
    if not batch_data:
        break
    train_x, train_y = inp_fn(batch_data)
    mdl.train_step(sess, train_x, train_y)
    train_eval = 'SKIP'
    test_eval = 'SKIP'
    if niter % 50 == 0:
        train_eval = mdl.eval_step(sess, train_x, train_y, metrics)
    # if niter % 50 == 0:
    #     test_eval = mdl.eval_step(sess, test_x, test_y, metrics)
    print niter, 'train:', train_eval, 'test:', test_eval

save_path = mdl.saver.save(sess, MODEL_CKPT_DIR, global_step=mdl.global_step)
print "model saved:", save_path

sess.close()
