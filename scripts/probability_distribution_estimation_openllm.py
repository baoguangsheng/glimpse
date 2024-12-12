# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import numpy as np
import tqdm
import argparse
import json
import torch
from model import load_tokenizer, load_model
from data_builder import load_data, save_data
from metrics import get_roc_metrics, get_precision_recall_metrics
from probability_distributions import GeometricDistribution, ZipfianDistribution, MlpDistribution, safe_log
from probability_distribution_estimation import PdeRank, PdeLogRank, PdeEntropy, PdeFastDetectGPT, get_likelihood

class OpenLLM:
    def __init__(self, args):
        self.args = args
        self.max_topk = args.max_topk
        self.tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
        self.model = load_model(args.scoring_model_name, args.device, args.cache_dir)
        self.model.eval()

    def eval(self, text):
        tokenized = self.tokenizer(text, truncation=True, return_tensors="pt", padding=True,
                                      return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits = self.model(**tokenized).logits[:, :-1]
        # outputs
        tokens = self.tokenizer.convert_ids_to_tokens([id for id in labels[0]])
        logits = logits.view(-1, logits.shape[-1])
        lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        logprobs = [lprobs[i, id].item() for i, id in enumerate(labels[0])]
        lprobs, indices = lprobs.topk(self.max_topk)
        toplogprobs = [dict((k, v) for k, v in zip(self.tokenizer.convert_ids_to_tokens(kk), vv)) for kk, vv in zip(indices.tolist(), lprobs.tolist())]
        return tokens, logprobs, toplogprobs

# Evaluate passages by calling to the OpenLLM
def evaluate_passages(args, gpt):
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])

    random.seed(args.seed)
    np.random.seed(args.seed)

    results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Evaluating passages"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        try:
            # original text
            tokens, logprobs, toplogprobs = gpt.eval(original_text)
            original_result = { 'text': original_text, 'tokens': tokens,
                       'logprobs': logprobs, 'toplogprobs': toplogprobs}
            # sampled text
            tokens, logprobs, toplogprobs = gpt.eval(sampled_text)
            sampled_result = { 'text': original_text, 'tokens': tokens,
                       'logprobs': logprobs, 'toplogprobs': toplogprobs}
            results.append({"original": original_result,
                            "sampled": sampled_result})
        except Exception as ex:
            print(ex)

    result_file = f'{args.output_file}_top{gpt.max_topk}'
    save_data(result_file, None, results)


# Experiment the criteria upon estimated distributions
def experiment(args):
    # prepare Completion API results
    gpt = OpenLLM(args)
    result_file = f'{args.output_file}_top{gpt.max_topk}.raw_data.json'
    if os.path.exists(result_file):
        print(f'Use existing result file: {result_file}')
    else:
        evaluate_passages(args, gpt)
    data = load_data(f'{args.output_file}_top{gpt.max_topk}')
    # random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    # evaluate criterion
    estimators = {"geometric": GeometricDistribution(args.top_k, args.rank_size),
                  "zipfian": ZipfianDistribution(args.top_k, args.rank_size),
                  "mlp": MlpDistribution(args.top_k, args.rank_size, args.device), }
    estimator = args.estimator
    distrib = estimators[estimator]
    criterion_fns = {
        "likelihood": get_likelihood,
        f"pde_entropy_{estimator}": PdeEntropy(distrib),
        f"pde_rank_{estimator}": PdeRank(distrib),
        f"pde_logrank_{estimator}": PdeLogRank(distrib),
        f"pde_fastdetect_{estimator}": PdeFastDetectGPT(distrib),
    }
    # Calculate the criteria
    n_samples = len(data)
    results = dict([(name, []) for name in criterion_fns])
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing criteria PDE with {estimator} distribution"):
        original_result = data[idx]["original"]
        sampled_result = data[idx]["sampled"]
        # original text
        original_text = original_result["text"]
        original_crit = dict([(name, criterion_fns[name](args, original_result)) for name in criterion_fns])
        # sampled text
        sampled_text = sampled_result["text"]
        sampled_crit = dict([(name, criterion_fns[name](args, sampled_result)) for name in criterion_fns])
        # result
        for name in criterion_fns:
            results[name].append({"original": original_text,
                            "original_crit": original_crit[name],
                            "sampled": sampled_text,
                            "sampled_crit": sampled_crit[name]})
    # output results
    for name in criterion_fns:
        # compute prediction scores for real/sampled passages
        predictions = {'real': [x["original_crit"] for x in results[name] if x["original_crit"] is not None],
                       'samples': [x["sampled_crit"] for x in results[name] if x["sampled_crit"] is not None]}
        print(f"Total {len(predictions['real'])}, Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
        fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
        p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
        print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
        # results
        results_file = f'{args.output_file}.{name}.json'
        results_output = { 'name': f'{name}_threshold',
                    'info': {'n_samples': n_samples},
                    'predictions': predictions,
                    'raw_results': results[name],
                    'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                    'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                    'loss': 1 - pr_auc}
        with open(results_file, 'w') as fout:
            json.dump(results_output, fout)
            print(f'Results written into {results_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_test/results/xsum_gpt-3.5-turbo.babbage-002")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_test/data/xsum_gpt-3.5-turbo")
    parser.add_argument('--scoring_model_name', type=str, default='gpt-neo-2.7B')
    parser.add_argument('--estimator', type=str, default='geometric', choices=['geometric', 'zipfian', 'mlp'])
    parser.add_argument('--rank_size', type=int, default=1000)
    parser.add_argument('--max_topk', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)
