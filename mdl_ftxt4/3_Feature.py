#!/usr/bin/env python

import sys

nlabel_th = int(sys.argv[1])
for line in sys.stdin:
    qid, labels, title_chars, title_words, _, _ = line.rstrip('\n').split('\t')
    label_pref = '__label__'
    words = ' '.join(title_words.split(','))
    labelsv = labels.split(',')
    if len(labelsv) > nlabel_th:  # drop inst with label_cnt > nlabel_th
        continue
    for label in labelsv:
        print label_pref + label + ' ' + words
