#!/usr/bin/env python

import fasttext
import sys
sys.path.append('../utils/')
import metricz
from operator import itemgetter


def rgrid(st, ed, step):
    r = st
    while r <= ed:
        yield r
        r += step

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

mdl1_file = '../mdl_ftxt1/res_test/res_test_prob.tsv'
mdl2_file = '../mdl_ftxt2/res_test/res_test_prob.tsv'
mdl1_preds = []
mdl2_preds = []
labels = []
with open(mdl1_file) as f1, open(mdl2_file) as f2:
    for inl1, inl2 in zip(f1, f2):
        label, preds1 = inl1.split('\t')
        labels.append(label.split(','))
        preds1 = preds1.split(',')
        preds1 = [(x.split(':')[0], float(x.split(':')[1])) for x in preds1]
        mdl1_preds.append(preds1)

        label, preds2 = inl2.split('\t')
        preds2 = preds2.split(',')
        preds2 = [(x.split(':')[0], float(x.split(':')[1])) for x in preds2]
        mdl2_preds.append(preds2)
print 'separate model data loaded'

eval_r = {}
for r in rgrid(0.00, 1.00, 0.01):
    print 'r:', r
    mix_preds = []
    for p1, p2 in zip(mdl1_preds, mdl2_preds):
        mix_preds.append(get_mixed_top(p1, p2, r))
    prec, rec, score = metricz.eval_score(mix_preds, labels)
    print 'evals:', prec, rec, score
    eval_r[r] = score
print 'best r:', sorted(eval_r.items(), key=itemgetter(1), reverse=True)[0]
