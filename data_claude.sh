#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_main
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

datasets="xsum writing pubmed"
source_models="claude-3-sonnet-20240229 claude-3-opus-20240229"

# preparing dataset
openai_base="https://api.claude-Plus.top/v1"
openai_key="xxxxxxxx"  # replace with your own key for generating your own test set

# We use a temperature of 0.8 for creativity writing
for M in $source_models; do
  for D in $datasets; do
    echo `date`, Preparing dataset ${D} by sampling from ${M} ...
    python scripts/data_builder.py --openai_model $M --openai_key $openai_key --openai_base $openai_base \
                --dataset $D --n_samples 150 --do_temperature --temperature 0.8 --batch_size 1 --device cpu \
                --output_file $data_path/${D}_${M}
  done
done