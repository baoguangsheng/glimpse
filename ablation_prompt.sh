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

datasets="xsum writing pubmed"
source_models="gpt-4"

# note: replace the endpoint, key, and version with yours
api_endpoint="https://xxxx.openai.azure.com/"
api_key="xxxxxxxx"
api_version="2024-02-15-preview"
settings="babbage-002 davinci-002 gpt-35-turbo-1106 gpt-4-1106"

for P in prompt0 prompt1 prompt2 prompt3 prompt4; do
  res_path=$exp_path/results_${P}
  mkdir -p $res_path

  for M1 in $settings; do
    for M in $source_models; do
      for D in $datasets; do
        echo `date`, Evaluating PDE (geometric) on ${D}_${M}.${M1} with ${P} ...
        python scripts/probability_distribution_estimation.py --api_endpoint $api_endpoint --api_version $api_version \
                                  --api_key $api_key --scoring_model_name $M1 --prompt $P \
                                  --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}
      done
    done
  done
done
