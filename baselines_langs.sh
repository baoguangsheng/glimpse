#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_langs
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

datasets="chinese russian urdu indonesian arabic bulgarian"
source_models="chatgpt"

# evaluate Fast-DetectGPT in the black-box setting
settings="gpt-j-6B:gpt-neo-2.7B"
for M in $source_models; do
  for D in $datasets; do
    for S in $settings; do
      IFS=':' read -r -a S <<< $S && M1=${S[0]} && M2=${S[1]}
      echo `date`, Evaluating Fast-DetectGPT on ${D}_${M}.${M1}_${M2} ...
      python scripts/fast_detect_gpt.py --reference_model_name $M1 --scoring_model_name $M2 --discrepancy_analytic \
                          --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2}
    done
  done
done

# evaluate supervised detectors
supervised_models="roberta-base-openai-detector roberta-large-openai-detector"
for M in $source_models; do
  for D in $datasets; do
    for SM in $supervised_models; do
      echo `date`, Evaluating ${SM} on ${D}_${M} ...
      python scripts/supervised.py --model_name $SM --dataset $D \
                            --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
    done
  done
done
