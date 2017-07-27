#!/bin/bash

# small word dict
# words in embedding file, without infrequent words
# 0 is used for zero padding
echo "UNK	1" > word_dict
cat ../data_all/word_embedding.txt | awk -F' ' '{print $1"\t"NR+1}' >> word_dict



<<!
# complete label dict
cat ../data_all/question_topic_train_set.txt | cut -d'	' -f2 | sed 's/,/\
/g' | sort | uniq | awk '{print $1"\t"NR-1}' > label_dict
!
