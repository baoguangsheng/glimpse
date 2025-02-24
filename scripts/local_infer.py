# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import numpy as np
import torch
import os
import glob
import argparse
import json
from model import load_tokenizer, load_model
from probability_distributions import GeometricDistribution
from probability_distribution_estimation import OpenAIGPT, PdeFastDetectGPT

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Glimpse:
    def __init__(self, args):
        self.args = args
        self.gpt = OpenAIGPT(args)
        self.criterion_fn = PdeFastDetectGPT(GeometricDistribution(args.top_k, args.rank_size))
        # pre-calculated parameters by fitting a LogisticRegression on detection results
        # babbage-002_geometric: k: 1.06, b: 3.39, acc: 0.83
        # davinci-002_geometric: k: 1.34, b: 2.41, acc: 0.86
        # gpt-35-turbo-1106_geometric: k: 1.31, b: 3.77, acc: 0.90
        linear_params = {
            'babbage-002': (1.06, 3.39),
            'davinci-002': (1.34, 2.41),
            'gpt-35-turbo-1106': (1.31, 3.77),
        }
        key = args.scoring_model_name
        self.linear_k, self.linear_b = linear_params[key]

    # compute conditional probability curvature
    def compute_crit(self, text):
        tokens, logprobs, toplogprobs = self.gpt.eval(text)
        result = { 'text': text, 'tokens': tokens,
                   'logprobs': logprobs, 'toplogprobs': toplogprobs}
        crit = self.criterion_fn(args, result)
        return crit, len(tokens)

    # compute probability
    def compute_prob(self, text):
        crit, ntoken = self.compute_crit(text)
        prob = sigmoid(self.linear_k * crit + self.linear_b)
        return prob, crit, ntoken


# run interactive local inference
def run(args):
    detector = Glimpse(args)
    # input text
    print('Local demo for Glimpse, where the longer text has more reliable result.')
    print('')
    while True:
        print("Please enter your text: (Press Enter twice to start processing)")
        lines = []
        while True:
            line = input()
            if len(line) == 0:
                break
            lines.append(line)
        text = "\n".join(lines)
        if len(text) == 0:
            break
        # estimate the probability of machine generated text
        prob, crit, ntokens = detector.compute_prob(text)
        print(f'Glimpse criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be machine-generated.')
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # use babbage-002 for least cost
    # use davinci-002 for better detection accuracy
    parser.add_argument('--scoring_model_name', type=str, default='davinci-002')
    parser.add_argument('--api_base', type=str, default='https://api.openai.com/v1')
    parser.add_argument('--api_key', type=str, default='xxxxxxxx')
    parser.add_argument('--api_version', type=str, default='2023-09-15-preview')
    parser.add_argument('--estimator', type=str, default='geometric', choices=['geometric', 'zipfian', 'mlp'])
    parser.add_argument('--prompt', type=str, default='prompt3', choices=['prompt0', 'prompt1', 'prompt2', 'prompt3', 'prompt4'])
    parser.add_argument('--rank_size', type=int, default=1000)
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()
    
    run(args)



