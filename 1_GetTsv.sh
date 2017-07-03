#!/bin/bash

feat_raw=data_all/question_train_set.txt
label=data_all/question_topic_train_set.txt
data_all=data_all/data_all.tsv

paste $label $feat_raw | cut -d'	' -f1,2,4,5,6,7 > $data_all
