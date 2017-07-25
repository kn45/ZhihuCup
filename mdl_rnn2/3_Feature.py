#!/usr/bin/env python

import sys
sys.path.append('../utils')
from dataproc import DictTable

word_tb = DictTable('word_dict')
label_tb = DictTable('label_dict')

for line in sys.stdin:
    qid, labels, _, title_words, _, doc_words = line.rstrip('\n').split('\t')
    word_ids = word_tb.lookup(title_words.split(',')) \
        + word_tb.lookup(doc_words.split(','))
    word_ids = [x if x else 0 for x in word_ids]  # id of UNK is 0
    label_ids = label_tb.lookup(labels.split(','))
    for label in label_ids:
        print '\t'.join(map(str, [label] + word_ids))
