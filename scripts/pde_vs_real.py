# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import scipy.stats as ss
import torch
import tqdm
import argparse
import os
from data_builder import load_data
from model import load_tokenizer, load_model
from probability_distributions import GeometricDistribution, ZipfianDistribution, MlpDistribution, safe_log
import matplotlib.pyplot as plt

def get_distrib(logits, rank_size):
    assert logits.shape[0] == 1
    logits = logits.to('cpu')
    logits = logits.view(-1, logits.shape[-1])
    lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    lprobs = lprobs.topk(rank_size).values
    return lprobs.tolist()

def get_distribs(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])

    distribs = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Get distributions"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        # original text
        tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        with torch.no_grad():
            logits = scoring_model(**tokenized).logits[:, :-1]
            distrib = get_distrib(logits, args.vocab_size)
            distribs.extend(distrib)
        # sampled text
        tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        with torch.no_grad():
            logits = scoring_model(**tokenized).logits[:, :-1]
            distrib = get_distrib(logits, args.vocab_size)
            distribs.extend(distrib)
    return distribs

def estimate_distribs(args, distribs, estimator):
    def _get_toplogprobs(distrib):
        idx = np.argpartition(distrib, -args.top_k)[-args.top_k:]
        vals = np.array(distrib)[idx]
        toplogprobs = dict(zip(idx, vals))
        return toplogprobs

    if estimator == 'geometric':
        estimator = GeometricDistribution(args.top_k, args.rank_size)
    elif estimator == 'zipfian':
        estimator = ZipfianDistribution(args.top_k, args.rank_size)
    elif estimator == 'mlp':
        estimator = MlpDistribution(args.top_k, args.rank_size, args.device)
    else:
        raise NotImplementedError

    new_distribs = []
    for distrib in distribs:
        toplogprobs = _get_toplogprobs(distrib)
        probs = estimator.estimate_distrib_token(toplogprobs)
        new_distribs.append(safe_log(probs).tolist())
    return new_distribs

def kl(p, q):
    q = q[:len(p)]
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def dkl_distrib(distribs, ref_distribs):
    dkls = []
    for d, r in zip(distribs, ref_distribs):
        d = np.exp(d)
        r = np.exp(r)
        dkls.append(kl(d, r))
    dkl = np.mean(dkls)
    return dkl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_main/data/xsum_gpt-4")
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B") # gpt2
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--rank_size', type=int, default=1000)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    distribs = get_distribs(args)
    dkl_geo = []
    dkl_zipf = []
    dkl_mlp = []
    for top_k in range(1, 11):
        args.top_k = top_k
        distribs_geo = estimate_distribs(args, distribs, 'geometric')
        distribs_zipf = estimate_distribs(args, distribs, 'zipfian')
        distribs_mlp = estimate_distribs(args, distribs, 'mlp')
        dkl_geo.append(dkl_distrib(distribs_geo, distribs))
        dkl_zipf.append(dkl_distrib(distribs_zipf, distribs))
        dkl_mlp.append(dkl_distrib(distribs_mlp, distribs))

    print(f'Geometric DKL: {dkl_geo}')
    print(f'Zipfian DKL: {dkl_zipf}')
    print(f'Mlp DKL: {dkl_mlp}')