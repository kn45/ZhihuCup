#!/usr/bin/env python

import fasttext

mdl = fasttext.load_model('mdl_fasttext_supervised.bin')

pred_file = '../data_all/question_eval_set.txt'
with open(pred_file) as f:
    pred_data = [x.rstrip('\n') for x in f.readlines()]

pred_id = []
pred_feats = []
for inl in pred_data:
    flds = inl.split('\t')
    pred_id.append(flds[0])
    pred_feats.append(' '.join(flds[2].split(',')))
del pred_data

print len(pred_feats)
print pred_feats[:2]

for p in pred_feats:
    pred_raw = mdl.predict_proba([p], 5)
    print pred_raw
#print pred_raw
