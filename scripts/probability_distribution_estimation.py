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
import time
from data_builder import load_data, save_data
from metrics import get_roc_metrics, get_precision_recall_metrics
from probability_distributions import GeometricDistribution, ZipfianDistribution, MlpDistribution


class OpenAIGPT:
    def __init__(self, args):
        self.args = args
        if args.api_base.find('azure.com') > 0:
            self.client = self.create_client_azure()
        else:
            self.client = self.create_client_openai()
        # predefined prompts
        self.prompts = {
            "prompt0": "",
            "prompt1": f"You serve as a valuable aide, capable of generating clear and persuasive pieces of writing given a certain context. Now, assume the role of an author and strive to finalize this article.\n",
            "prompt2": f"You serve as a valuable aide, capable of generating clear and persuasive pieces of writing given a certain context. Now, assume the role of an author and strive to finalize this article.\nI operate as an entity utilizing GPT as the foundational large language model. I function in the capacity of a writer, authoring articles on a daily basis. Presented below is an example of an article I have crafted.\n",
            "prompt3": f"System:\nYou serve as a valuable aide, capable of generating clear and persuasive pieces of writing given a certain context. Now, assume the role of an author and strive to finalize this article.\nAssistant:\nI operate as an entity utilizing GPT as the foundational large language model. I function in the capacity of a writer, authoring articles on a daily basis. Presented below is an example of an article I have crafted.\n",
            "prompt4": f"Assistant:\nYou serve as a valuable aide, capable of generating clear and persuasive pieces of writing given a certain context. Now, assume the role of an author and strive to finalize this article.\nUser:\nI operate as an entity utilizing GPT as the foundational large language model. I function in the capacity of a writer, authoring articles on a daily basis. Presented below is an example of an article I have crafted.\n",
        }
        self.max_topk = 10

    def create_client_azure(self):
        from openai import AzureOpenAI
        return AzureOpenAI(
            azure_endpoint=self.args.api_base,
            api_key=self.args.api_key,
            api_version=self.args.api_version)

    def create_client_openai(self):
        from openai import OpenAI
        return OpenAI(
            base_url=self.args.api_base,
            api_key=self.args.api_key)

    def eval(self, text):
        prompt = self.prompts[self.args.prompt]
        nretry = 3
        while nretry > 0:
            try:
                # get top alternative tokens
                kwargs = {"model": self.args.scoring_model_name, "max_tokens": 0, "echo": True, "logprobs": self.max_topk}
                response = self.client.completions.create(prompt=f"<|endoftext|>{prompt}{text}", **kwargs)
                result = response.choices[0]
                # decide the prefix length
                prefix = ""
                nprefix = 1
                while len(prefix) < len(prompt):
                    prefix += result.logprobs.tokens[nprefix]
                    nprefix += 1
                assert prefix == prompt, f"Mismatch: {prompt} .vs. {prefix}"
                tokens = result.logprobs.tokens[nprefix:]
                logprobs = result.logprobs.token_logprobs[nprefix:]
                toplogprobs = result.logprobs.top_logprobs[nprefix:]
                toplogprobs = [dict(item) for item in toplogprobs]
                assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"
                assert len(tokens) == len(toplogprobs), f"Expected {len(tokens)} toplogprobs, got {len(toplogprobs)}"
                return tokens, logprobs, toplogprobs
            except Exception as ex:
                print(ex)
                print('Sleep 10 seconds before retry ...')
                time.sleep(10)
                nretry -= 1

class PdeBase:
    def __init__(self, distrib):
        self.distrib = distrib

    def estimate_distrib_sequence(self, item):
        key = f'{self.distrib.name}-top{self.distrib.top_k}'
        if key in item:
            probs = item[key]
        else:
            toplogprobs = item["toplogprobs"]
            probs = [self.distrib.estimate_distrib_token(v) for v in toplogprobs]
            item[key] = probs
        return np.array(probs)

# Extension of Entropy
class PdeEntropy(PdeBase):
    def __call__(self, args, item):
        probs = self.estimate_distrib_sequence(item)
        lprobs = np.nan_to_num(np.log(probs))
        entropy = - (probs * lprobs).sum(axis=-1)
        return np.mean(entropy)

# Extension of Rank
class PdeRank(PdeBase):
    def __call__(self, args, item):
        logprobs = item["logprobs"]
        probs = self.estimate_distrib_sequence(item)
        ranks = []
        for logprob, prob in zip(logprobs, probs):
            p_tok = np.exp(logprob)
            rank = 0
            while rank < len(prob) and prob[rank] > p_tok:
                rank += 1
            if rank < len(prob):
                ranks.append(rank + 1)
        return -np.mean(ranks)

# Extension of Log-Rank
class PdeLogRank(PdeBase):
    def __call__(self, args, item):
        logprobs = item["logprobs"]
        probs = self.estimate_distrib_sequence(item)
        logranks = []
        for logprob, prob in zip(logprobs, probs):
            p_tok = np.exp(logprob)
            rank = 0
            while rank < len(prob) and prob[rank] > p_tok:
                rank += 1
            if rank < len(prob):
                logranks.append(np.log(rank + 1))
        return -np.mean(logranks)

# Extension of Fast-DetectGPT
class PdeFastDetectGPT(PdeBase):
    def __call__(self, args, item):
        logprobs = item["logprobs"]
        probs = self.estimate_distrib_sequence(item)
        log_likelihood = np.array(logprobs)
        lprobs = np.nan_to_num(np.log(probs))
        mean_ref = (probs * lprobs).sum(axis=-1)
        lprobs2 = np.nan_to_num(np.square(lprobs))
        var_ref = (probs * lprobs2).sum(axis=-1) - np.square(mean_ref)
        discrepancy = (log_likelihood.sum(axis=-1) - mean_ref.sum(axis=-1)) / np.sqrt(var_ref.sum(axis=-1))
        discrepancy = discrepancy.mean()
        return discrepancy.item()


# Likelihood
def get_likelihood(args, item):
    logprobs = item["logprobs"]
    log_likelihood = np.array(logprobs)
    return log_likelihood.mean().item()


# Evaluate passages by calling to the OpenAI API
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

    result_file = f'{args.output_file}_{args.prompt}_top{gpt.max_topk}'
    save_data(result_file, None, results)


# Experiment the criteria upon estimated distributions
def experiment(args):
    # prepare Completion API results
    gpt = OpenAIGPT(args)
    result_file = f'{args.output_file}_{args.prompt}_top{gpt.max_topk}.raw_data.json'
    if os.path.exists(result_file):
        print(f'Use existing result file: {result_file}')
    else:
        evaluate_passages(args, gpt)
    data = load_data(f'{args.output_file}_{args.prompt}_top{gpt.max_topk}')
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
    parser.add_argument('--scoring_model_name', type=str, default='davinci-002')
    parser.add_argument('--api_base', type=str, default='https://xxxx.openai.azure.com/')
    parser.add_argument('--api_key', type=str, default='xxxxxxxx')
    parser.add_argument('--api_version', type=str, default='2023-09-15-preview')
    parser.add_argument('--estimator', type=str, default='geometric', choices=['geometric', 'zipfian', 'mlp'])
    parser.add_argument('--prompt', type=str, default='prompt3', choices=['prompt0', 'prompt1', 'prompt2', 'prompt3', 'prompt4'])
    parser.add_argument('--rank_size', type=int, default=1000)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    experiment(args)
