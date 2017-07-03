#!/bin/bash

data=../data_all
# topic count
cat $data/question_topic_train_set.txt | cut -d'	' -f2 | sed 's/,/\
/g' | sort | uniq -c > topic_inst_cnt

