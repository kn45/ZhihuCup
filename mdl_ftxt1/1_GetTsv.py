#!/usr/bin/env python

import sys

for line in sys.stdin:
    labels, words = line.rstrip('\n').split('\t')
    label_pref = '__label__'
    words = ' '.join(words.split(','))
    for label in labels.split(','):
        print label_pref + label + ' ' + words
