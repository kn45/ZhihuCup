#!/usr/bin/env python

import fasttext
import sys


mode = 'supervised'
input_file = 'data_train/data_trnvld.ssv'
model_file = 'mdl_fasttext_' + mode

test_file = 'data_test/data_test.ssv'

model = fasttext.supervised(
    input_file=input_file,
    output=model_file,
    min_count=5,
    word_ngrams=2,
    bucket=2000000,
    minn=1,
    maxn=15,
    dim=256,
    epoch=5,
    neg=5,
    thread=4,
    loss='ns',
    silent=0)

model.test(test_file)
