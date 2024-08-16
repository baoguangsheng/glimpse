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
settings="gpt-35-turbo-1106:prompt3 gpt-4-1106:prompt4"

for K in 1 3 5 7 10; do
  res_path=$exp_path/results_top${K}
  mkdir -p $res_path

  for estimator in geometric zipfian mlp; do
    for S in $settings; do
      IFS=':' read -r -a S <<< $S && M1=${S[0]} && P=${S[1]}
      for M in $source_models; do
        # copy the evaluation raw files from the main results
        cp $exp_path/results/*_${M}.*.raw_data.json $res_path/. -rf
        for D in $datasets; do
          echo `date`, "Evaluating PDE (${estimator}) on ${D}_${M}.${M1} with top-${K} ..."
          python scripts/probability_distribution_estimation.py --api_endpoint $api_endpoint --api_version $api_version \
                                    --api_key $api_key --scoring_model_name $M1 --prompt $P --top_k $K --estimator $estimator \
                                    --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}
        done
      done
    done
  done
done
