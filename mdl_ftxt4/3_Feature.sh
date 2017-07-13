#!/bin/bash

# tsv -> feature
 
data_trnvld=../data_train/data_trnvld.tsv
data_test=../data_test/data_test.tsv
 
feat_trnvld=feat_train/trnvld_feature.ssv
feat_test=feat_test/test_feature.ssv
 
cat $data_trnvld | python 3_Feature.py 1 > $feat_trnvld
cat $data_test | python 3_Feature.py 100 > $feat_test
