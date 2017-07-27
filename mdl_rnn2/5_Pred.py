#!/usr/bin/env python

import numpy as np
import sys
sys.path.append('../../MLFlow/utils')
sys.path.append('../../TFModels/rnn_classifier')
import dataproc
import tensorflow as tf
from text_rnn import TextRNNClassifier
from operator import itemgetter


NCLASS = 1999
NWORDS = 411721
SEQ_LEN = 30
LABEL_REPR = 'dense'
MAX_ITER = 30000
BATCH_SIZE = 128
MODEL_CKPT_DIR = './model_ckpt/'


def inp_fn(data, word_tb):
    inp_x = []
    inp_y = []
    for inst in data:
        flds = inst.split('\t')
        qid = flds[0]
        title_words = flds[2]
        doc_words = flds[4]
        word_ids = word_tb.lookup(title_words.split(',')) + \
            word_tb.lookup(doc_words.split(','))
        feats = [x if x else 0 for x in word_ids]
        inp_x.append(dataproc.zero_padding(feats, SEQ_LEN))
        inp_y.append(qid)
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
pred_file = '../data_all/question_eval_set.txt'
pred_reader = dataproc.BatchReader(pred_file, max_epoch=1)
if out_mode == 'cls':
    pred_res = 'res_pred/res_pred.csv'
else:
    pred_res = 'res_pred/res_pred_prob.tsv'
pred_writer = open(pred_res, 'w')

# load model
mdl = TextRNNClassifier(
    seq_len=SEQ_LEN,
    emb_dim=256,
    nclass=NCLASS,
    vocab_size=NWORDS,
    reg_lambda=0.0,
    lr=1e-3,
    label_repr=LABEL_REPR,
    obj='softmax',
    nsample=1)
sess = tf.Session()
mdl.saver.restore(sess, tf.train.latest_checkpoint(MODEL_CKPT_DIR))

niter = 0
while niter < MAX_ITER:
    niter += 1
    batch_data = pred_reader.get_batch(BATCH_SIZE)
    if not batch_data:
        break
    pred_x, qid = inp_fn(batch_data, word_tb)
    preds = mdl.predict_proba(sess, pred_x)
    pred5 = map(get_top5, preds)
    pred5 = map(label_tb.lookup_rev, pred5)
    for t, p in zip(qid, pred5):
        print >> pred_writer, t + ',' + ','.join(p)


sess.close()
pred_writer.close()