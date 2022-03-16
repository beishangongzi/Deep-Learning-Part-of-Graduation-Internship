#!/bin/bash
echo "split some data as test dataset"
# shellcheck disable=SC2046
export $(xargs <../.env)
train_dir=$1
for class_dir in `ls $train_dir`; do
  echo $class_dir

done



