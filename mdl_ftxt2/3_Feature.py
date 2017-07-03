#!/usr/bin/env python

import sys

for line in sys.stdin:
    qid, labels, _, _, _, doc_words = line.rstrip('\n').split('\t')
    label_pref = '__label__'
    words = ' '.join(doc_words.split(','))
    for label in labels.split(','):
        print label_pref + label + ' ' + words
