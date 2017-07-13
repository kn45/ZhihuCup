#!/usr/bin/env python

import fasttext
import sys

out_mode = 'prob' if len(sys.argv) > 1 and sys.argv[1] == 'prob' else 'cls'
pred_file = '../data_all/question_eval_set.txt'
if out_mode == 'cls':
    pred_res = 'res_pred/res_pred.csv'
else:
    pred_res = 'res_pred/res_pred_prob.tsv'

mdl = fasttext.load_model('mdl_fasttext_supervised.bin')

with open(pred_file) as f, open(pred_res, 'w') as fo:
    # qid, label, title_chars, title_words
    for n, line in enumerate(f):
        flds = line.rstrip('\n').split('\t')
        pred_id = flds[0]
        pred_feats = ' '.join(flds[4].split(','))
        if len(pred_feats) == 0:
            pred_feats = ' '
        if out_mode == 'cls':
            pred_raw = mdl.predict([pred_feats], 5)
            pred_label = ','.join(map(lambda x: x[9:], pred_raw[0]))
            print >> fo, pred_id + ',' + pred_label
        if out_mode == 'prob':
            pred_raws = mdl.predict_proba([pred_feats], 15)
            for pred_raw in pred_raws:
                pred_label = ','.join(map(lambda t: t[0][9:]+':'+str(t[1]), pred_raw))
                print >> fo, pred_id + '\t' + pred_label
