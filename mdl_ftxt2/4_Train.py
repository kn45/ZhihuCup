#!/usr/bin/env python

import fasttext
import sys


mode = 'supervised'
input_file = 'feat_train/trnvld_feature.ssv'
model_file = 'mdl_fasttext_' + mode

valid_file = 'feat_train/valid_feature.ssv'

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

# model.test(valid_file)  # too slow
