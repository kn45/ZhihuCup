#!/usr/bin/env python

import sys
sys.path.append('../utils/')
import metricz

test_file = sys.argv[1]
with open(test_file) as f:
    data_test = [l.rstrip('\n').split('\t') for l in f.readlines()]

t_values = [l[0].split(',') for l in data_test]
p_values = [l[1].split(',') for l in data_test]

prec, rec, score = metricz.eval_score(p_values, t_values)
print "Precision:", prec
print "Recall:", rec
print "Score:", score

