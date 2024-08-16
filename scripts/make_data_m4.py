# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

import numpy as np
import random
import argparse
import os
import json
from data_builder import save_data


def make_data(args, data_file, result_file):
    with open(data_file, "r") as fin:
        data = [json.loads(line) for line in fin]
        print(f"M4 data loaded from {data_file}")
    # sample data
    random.seed(args.seed)
    random.shuffle(data)
    data = data[:args.n_samples]
    print(f'Sampled {len(data)} pairs of human and machine text.')
    # convert format
    results = {
        "original": [],
        "sampled": [],
    }
    for item in data:
        results['original'].append(item['human_text'])
        results['sampled'].append(item['machine_text'])
    # save to file
    save_data(result_file, None, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default="./exp_langs/data")
    parser.add_argument('--m4data_path', type=str, default="../M4/data")
    parser.add_argument('--n_samples', type=int, default=150)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    langs = {
        'chinese': 'qazh_chatgpt.jsonl',
        'arabic': 'arabic_chatGPT.jsonl',
        'russian': 'russian_chatGPT.jsonl',
        'bulgarian': 'bulgarian_true_and_fake_news_chatGPT.jsonl',
        'urdu': 'urdu_chatGPT.jsonl',
        'indonesian': 'id-newspaper_chatGPT.jsonl'
    }

    for lang in langs:
        data_file = os.path.join(args.m4data_path, langs[lang])
        result_file = os.path.join(args.output_path, f'{lang}_chatgpt')
        make_data(args, data_file, result_file)

