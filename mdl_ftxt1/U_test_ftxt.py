#!/usr/bin/env python

import fasttext

classifier = fasttext.load_model('mdl_fasttext_supervised.bin')

test_file = 'data_test/data_test.ssv.small'
result = classifier.test(test_file)
print 'P@1:', result.precision
print 'R@1:', result.recall
print 'Number of examples:', result.nexamples
