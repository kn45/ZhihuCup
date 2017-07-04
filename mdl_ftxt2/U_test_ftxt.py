#!/usr/bin/env python

import fasttext
import itertools
import sys

out_mode = 'prob' if len(sys.argv) > 1 and sys.argv[1] == 'prob' else 'cls'
#test_file = '../data_test/data_test.1k.tsv'
test_file = '../data_test/data_test.tsv'
if out_mode == 'cls':
    test_res = 'res_test/res_test.tsv'
else:
    test_res = 'res_test/res_test_prob.tsv'
BATCH_SIZE = 256

mdl = fasttext.load_model('mdl_fasttext_supervised.bin')
with open(test_file) as f, open(test_res, 'w') as fo:
    # qid, label, title_chars, title_words
    iter_cnt = 0
    feats_buff = []
    label_buff = []
    for line in f:
        iter_cnt += 1
        flds = line.rstrip('\n').split('\t')
        test_label = flds[1]
        test_feats = ' '.join(flds[5].split(','))
        if len(test_feats) == 0:
            test_feats = ' '
        feats_buff.append(test_feats)
        label_buff.append(test_label)
        if iter_cnt % BATCH_SIZE != 0:
            continue
        if out_mode == 'cls':
            pred_raws = mdl.predict(feats_buff, 5)
            for pred_raw, test_label in zip(pred_raws, label_buff):
                pred_label = ','.join(map(lambda x: x[9:], pred_raw))
                print >> fo, test_label + '\t' + pred_label
        if out_mode == 'prob':
            pred_raws = mdl.predict_proba(feats_buff, 15)
            for pred_raw, test_label in zip(pred_raws, label_buff):
                pred_label = ','.join(map(lambda t: t[0][9:]+':'+str(t[1]), pred_raw))
                print >> fo, test_label + '\t' + pred_label
        # reset buffer
        iter_cnt = 0
        feats_buff = []
        label_buff = []
