#!/usr/bin/env python

import sys
sys.path.append('../utils')
from dataproc import DictTable

word_tb = DictTable('word_dict', UNK=1)
label_tb = DictTable('label_dict')

for line in sys.stdin:
    qid, labels, _, title_words, _, doc_words = line.rstrip('\n').split('\t')
    word_ids = word_tb.lookup(title_words.split(','))
    label_ids = label_tb.lookup(labels.split(','))
    print ','.join(map(str, label_ids)) + '\t' + ','.join(map(str, word_ids))
