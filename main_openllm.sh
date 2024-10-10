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
mkdir -p $exp_path $data_path

datasets="xsum writing pubmed"
source_models="gpt-4"

scoring_models="gpt-neo-2.7B"
estimators="geometric:1000 zipfian:1000 mlp:1000"

for top_k in 1 2 3 5 7 10; do
  res_path=$exp_path/results_openllm_top${top_k}
  mkdir -p $res_path

  for ER in $estimators; do
    IFS=':' read -r -a ER <<< $ER && E=${ER[0]} && R=${ER[1]}
    for M in $source_models; do
      for D in $datasets; do
        for M1 in $scoring_models; do
          echo `date`, "Evaluating PDE (${E}:${R}) on ${D}_${M}.${M1} ..."
          python scripts/probability_distribution_estimation_openllm.py --scoring_model_name $M1 --estimator $E --rank_size $R \
                                    --max_topk 10 --top_k $top_k \
                                    --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}
        done
      done
    done
  done
done

# evaluate Fast-DetectGPT in the black-box setting
res_path=$exp_path/results_openllm
mkdir -p $res_path

settings="gpt-neo-2.7B:gpt-neo-2.7B"
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

# evaluate baselines
scoring_models="gpt-neo-2.7B"
for M in $source_models; do
  for D in $datasets; do
    for M2 in $scoring_models; do
      echo `date`, Evaluating baseline methods on ${D}_${M}.${M2} ...
      python scripts/baselines.py --scoring_model_name ${M2} --dataset $D \
                            --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M2}
    done
  done
done