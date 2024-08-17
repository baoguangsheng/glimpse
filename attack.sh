#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_attack
src_path=exp_main
src_data_path=$src_path/data
mkdir -p $exp_path

datasets="xsum writing pubmed"
source_models="gpt-3.5-turbo"
settings="0:60 60:0"  # lex_diversity:order_diversity

# preparing dataset
for SS in $settings; do
  IFS=':' read -r -a SS <<< $SS && lex=${SS[0]} && order=${SS[1]}

  data_path=$exp_path/data_lex${lex}_order${order}
  mkdir -p $data_path

  for D in $datasets; do
    for M in $source_models; do
      echo `date`, Preparing dataset ${D}_${M} by paraphrasing  ${src_data_path}/${D}_${M} ...
      python scripts/paraphrasing.py --dataset $D --dataset_file $src_data_path/${D}_${M} \
                                     --lex_diversity $lex --order_diversity $order \
                                     --output_file $data_path/${D}_${M}
    done
  done
done

for SS in $settings; do
  IFS=':' read -r -a SS <<< $SS && lex=${SS[0]} && order=${SS[1]}

  data_path=$exp_path/data_lex${lex}_order${order}
  res_path=$exp_path/results_lex${lex}_order${order}
  mkdir -p $data_path $res_path

  # PDE
  # note: replace the endpoint, key, and version with yours
  api_endpoint="https://xxxx.openai.azure.com/"
  api_key="xxxxxxxx"
  api_version="2024-02-15-preview"
  scoring_models="babbage-002:prompt3 davinci-002:prompt3 gpt-35-turbo-1106:prompt3"
  estimators="geometric:1000 zipfian:100 mlp:100"

  for ER in $estimators; do
    IFS=':' read -r -a ER <<< $ER && E=${ER[0]} && R=${ER[1]}
    for M in $source_models; do
      for D in $datasets; do
        for S in $scoring_models; do
          IFS=':' read -r -a S <<< $S && M1=${S[0]} && P=${S[1]}
          echo `date`, Evaluating PDE on ${D}_${M}.${M1} with ${P} ...
          python scripts/probability_distribution_estimation.py --api_endpoint $api_endpoint --api_version $api_version \
                                    --api_key $api_key --scoring_model_name $M1 --prompt $P --estimator $E --rank_size $R \
                                    --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}
        done
      done
    done
  done

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
done

