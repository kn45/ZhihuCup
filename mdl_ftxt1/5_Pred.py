#!/usr/bin/env python

import fasttext

mdl = fasttext.load_model('mdl_fasttext_supervised.bin')

pred_file = '../data_all/question_eval_set.txt'
pred_res = 'res_pred/res_pred.csv'

with open(pred_file) as f, open(pred_res, 'w') as fo:
    # qid, label, title_chars, title_words
    for n, line in enumerate(f):
        flds = line.rstrip('\n').split('\t')
        pred_id = flds[0]
        pred_feats = ' '.join(flds[2].split(','))
        if len(pred_feats) == 0:
            pred_feats = ' '
        try:
            pred_raw = mdl.predict([pred_feats], 5)
        except:
            sys.stderr.write(str(n) + '\t' + line)
            sys.exit(1)
        pred_label = ','.join(map(lambda x: x[9:], pred_raw[0]))
        print >> fo, pred_id + ',' + pred_label
