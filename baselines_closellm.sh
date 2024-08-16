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
source_models="gpt-3.5-turbo gpt-4 claude-3-sonnet-20240229 claude-3-opus-20240229 gemini-1.5-pro"

api_endpoint="https://xxxx.openai.azure.com/"
api_key="xxxxxxxx"
api_version="2024-02-15-preview"
scoring_models="gpt-35-turbo-1106"

for M in $source_models; do
  for D in $datasets; do
    for M1 in $scoring_models; do
      echo `date`, Evaluating DNA-GPT on ${D}_${M}.${M1} ...
      python scripts/dna_gpt.py --api_endpoint $api_endpoint --api_version $api_version \
                                --api_key $api_key --base_model_name $M1 \
                                --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}
    done
  done
done
