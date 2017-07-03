#!/usr/bin/env python

import sys

for line in sys.stdin:
    qid, labels, title_chars, title_words, _, _ = line.rstrip('\n').split('\t')
    label_pref = '__label__'
    words = ' '.join(title_words.split(','))
    for label in labels.split(','):
        print label_pref + label + ' ' + words
