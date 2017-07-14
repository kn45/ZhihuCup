#!/usr/bin/env python

import fasttext
import sys
sys.path.append('../utils/')
import metricz
from operator import itemgetter


def get_mixed_top(p1, p2, r=0.0):
    res = {}
    for label, score in p1:
        if label not in res:
            res[label] = 0.0
        res[label] += score * r
    for label, score in p2:
        if label not in res:
            res[label] = 0.0
        res[label] += score * (1 - r)
    res_top = sorted(res.items(), key=itemgetter(1), reverse=True)[0:5]
    res_top = [x[0] for x in res_top]  # take label without proba
    return res_top

BEST_R = 0.57

mdl1_file = '../mdl_ftxt1/res_pred/res_pred_prob.tsv'
mdl2_file = '../mdl_ftxt2/res_pred/res_pred_prob.tsv'
pred_file = 'res_pred/res_pred.csv'
mdl1_preds = []
mdl2_preds = []
qids = []
with open(mdl1_file) as f1, open(mdl2_file) as f2:
    for inl1, inl2 in zip(f1, f2):
        qid, preds1 = inl1.split('\t')
        qids.append(qid)
        preds1 = preds1.split(',')
        preds1 = [(x.split(':')[0], float(x.split(':')[1])) for x in preds1]
        mdl1_preds.append(preds1)

        qid, preds2 = inl2.split('\t')
        preds2 = preds2.split(',')
        preds2 = [(x.split(':')[0], float(x.split(':')[1])) for x in preds2]
        mdl2_preds.append(preds2)

fo = open(pred_file, 'w')
for qid, p1, p2 in zip(qids, mdl1_preds, mdl2_preds):
    mix_pred = get_mixed_top(p1, p2, BEST_R)
    print >> fo, ','.join([qid] + mix_pred)
fo.close()
