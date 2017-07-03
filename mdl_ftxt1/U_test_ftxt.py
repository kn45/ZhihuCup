#!/usr/bin/env python

import fasttext
import itertools
import sys

mdl = fasttext.load_model('mdl_fasttext_supervised.bin')
#test_file = '../data_test/data_test.small.tsv'
test_file = '../data_test/data_test.tsv'
test_res = 'res_test/res_test.tsv'

def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

BATCH_SIZE = 256
with open(test_file) as f, open(test_res, 'w') as fo:
    # qid, label, title_chars, title_words
    iter_cnt = 0
    feats_buff = []
    label_buff = []
    for line in f:
        iter_cnt += 1
        flds = line.rstrip('\n').split('\t')
        test_label = flds[1]
        test_feats = ' '.join(flds[3].split(','))
        if len(test_feats) == 0:
            test_feats = ' '
        feats_buff.append(test_feats)
        label_buff.append(test_label)
        if iter_cnt % BATCH_SIZE == 0:
            pred_raws = mdl.predict(feats_buff, 5)
            for pred_raw, test_label in zip(pred_raws, label_buff):
                pred_label = ','.join(map(lambda x: x[9:], pred_raw))
                print >> fo, test_label + '\t' + pred_label

            iter_cnt = 0
            feats_buff = []
            label_buff = []
