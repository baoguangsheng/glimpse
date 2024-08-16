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

# note: replace the endpoint, key, and version with yours
api_endpoint="https://xxxx.openai.azure.com/"
api_key="xxxxxxxx"
api_version="2024-02-15-preview"
scoring_models="babbage-002:prompt3 davinci-002:prompt3 gpt-35-turbo-1106:prompt3 gpt-4-1106:prompt4"

for M in $source_models; do
  for D in $datasets; do
    for S in $scoring_models; do
      IFS=':' read -r -a S <<< $S && M1=${S[0]} && P=${S[1]}
      echo `date`, Evaluating PDE on ${D}_${M}.${M1} with ${P} ...
      python scripts/probability_distribution_estimation.py --api_endpoint $api_endpoint --api_version $api_version \
                                --api_key $api_key --scoring_model_name $M1 --prompt $P \
                                --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}
    done
  done
done
