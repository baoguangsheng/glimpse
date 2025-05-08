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
from scipy.stats import norm


# Considering balanced classification that p(D0) equals to p(D1), we have
#   p(D1|x) = p(x|D1) / (p(x|D1) + p(x|D0))
def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
    pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
    pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
    prob = pdf_value1 / (pdf_value0 + pdf_value1)
    return prob

class Glimpse:
    def __init__(self, args):
        self.args = args
        self.gpt = OpenAIGPT(args)
        self.criterion_fn = PdeFastDetectGPT(GeometricDistribution(args.top_k, args.rank_size))
        # To obtain probability values that are easy for users to understand, we assume normal distributions
        # of the criteria and statistic the parameters on a group of dev samples. The normal distributions are defined
        # by mu0 and sigma0 for human texts and by mu1 and sigma1 for AI texts. We set sigma1 = 2 * sigma0 to
        # make sure of a wider coverage of potential AI texts.
        # Note: the probability could be high on both left side and right side of Normal(mu0, sigma0).
        #   babbage-002_geometric: mu0: -5.1874, sigma0: 2.0760, mu1: -1.3959, sigma1: 4.1521, acc:0.8215
        #   davinci-002_geometric: mu0: -3.8289, sigma0: 2.0131, mu1: -0.0733, sigma1: 4.0261, acc:0.8460
        #   gpt-35-turbo-1106_geometric: mu0: -5.2040, sigma0: 2.0716, mu1: -0.7821, sigma1: 4.1432, acc:0.8894
        distrib_params = {
            'babbage-002_geometric': {'mu0': -5.1874, 'sigma0': 2.0760, 'mu1': -1.3959, 'sigma1': 4.1521},
            'davinci-002_geometric': {'mu0': -3.8289, 'sigma0': 2.0131, 'mu1': -0.0733, 'sigma1': 4.0261},
            'gpt-35-turbo-1106_geometric': {'mu0': -5.2040, 'sigma0': 2.0716, 'mu1': -0.7821, 'sigma1': 4.1432},
        }
        key = f'{args.scoring_model_name}_{args.estimator}'
        self.classifier = distrib_params[key]

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
        mu0 = self.classifier['mu0']
        sigma0 = self.classifier['sigma0']
        mu1 = self.classifier['mu1']
        sigma1 = self.classifier['sigma1']
        prob = compute_prob_norm(crit, mu0, sigma0, mu1, sigma1)
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



