#!/bin/bash

# Split data, make data sets:
# @Train+Valid
# @Train
# @Valid
# @Test

data_src=data_all/data_all.tsv
data_trnvld=data_train/data_trnvld.tsv
<<!
data_train=data_train/data_train.tsv
data_valid=data_train/data_valid.tsv
!
data_test=data_test/data_test.tsv

test_ratio=0.15
valid_ratio=0.15


# For regression task, random sampling
rand_samp()
{
  all_cnt=`cat $data_src | wc -l | sed -e 's/^[ 	]*//'`

  # split to train_valid + test
  data_src_shuf=${data_src}.shuf
  test_cnt=`echo | awk 'BEGIN{print int("'"$all_cnt"'"*"'"$test_ratio"'")}'`
  trnvld_cnt=`echo ""| awk 'BEGIN{print "'"$all_cnt"'"-"'"$test_cnt"'"}'`
  cat $data_src | perl -MList::Util=shuffle -e'print shuffle<>' > $data_src_shuf
  head -n $trnvld_cnt $data_src_shuf | perl -MList::Util=shuffle -e'print shuffle<>' > $data_trnvld
  tail -n $test_cnt $data_src_shuf > $data_test
  echo -e "   (all)"$all_cnt"  = (trnvld)"$trnvld_cnt"  +  (test)"$test_cnt"\tTest Ratio: "$test_ratio
  rm -f $data_src_shuf

<<!
  # split train_valid to train + valid
  # for the training without cross-validation
  valid_cnt=`echo | awk 'BEGIN{print int("'"$trnvld_cnt"'"*"'"$valid_ratio"'")}'`
  train_cnt=`echo | awk 'BEGIN{print "'"$trnvld_cnt"'"-"'"$valid_cnt"'"}'`
  head -n $train_cnt $data_trnvld > $data_train
  tail -n $valid_cnt $data_trnvld > $data_valid
  echo -e "(trnvld)"$trnvld_cnt"  =  (train)"$train_cnt"  + (valid)"$valid_cnt"\tValid Ratio: "$valid_ratio
!
}
rand_samp


# For classification task, stratified sampling
strf_samp()
{
  python 2_Split.py $data_src $data_trnvld $data_test $test_ratio
  python 2_Split.py $data_trnvld $data_train $data_valid $valid_ratio
}
#strf_samp
