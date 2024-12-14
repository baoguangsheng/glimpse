# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os.path
import argparse
import json
import numpy as np
from metrics import get_roc_metrics

dataset_names = {'xsum': 'XSum',
                 'writing': 'Writing',
                 'pubmed': 'PubMed',
                 'chinese': 'Chinese',
                 'russian': 'Russian',
                 'urdu': 'Urdu',
                 'indonesian': 'Indonesian',
                 'arabic': 'Arabic',
                 'bulgarian': 'Bulgarian',
                 }

estimator_names = {'geometric': 'Geometric',
                   'zipfian': 'Zipfian',
                   'mlp': 'MLP'}

model_names = {'gpt-3.5-turbo': 'ChatGPT',
               'gpt-4': 'GPT-4',
               'gpt-35-turbo-1106': 'GPT-3.5',
               'gpt-4-1106': 'GPT-4',
               'claude-3-sonnet-20240229': 'Claude-3 Sonnet',
               'claude-3-opus-20240229': 'Claude-3 Opus',
               'gemini-1.5-pro': 'Gemini-1.5 Pro',
               't5-11b': 'T5-11B',
               'gpt-neo-2.7B': 'Neo-2.7',
               'gpt-j-6B': 'GPT-J',
               'babbage-002': 'Babbage',
               'davinci-002': 'Davinci',
               't5-11b_gpt-neo-2.7B': 'T5-11B/Neo-2.7',
               'gpt-j-6B_gpt-neo-2.7B': 'GPT-J/Neo-2.7',
               'gpt-neo-2.7B_gpt-neo-2.7B': 'Neo-2.7',
               'phi-2_phi-2': 'Phi2-2.7B',
               'llama3-8b_llama3-8b': 'Llama3-8B',
               'qwen2.5-7b_qwen2.5-7b': 'Qwen2.5-7B',
               'gpt-neox-20b_gpt-neox-20b': 'gpt-neox-20B',
               'phi-2': 'Phi2-2.7B',
               'llama3-8b': 'Llama3-8B',
               'qwen2.5-7b': 'Qwen2.5-7B',
               'gpt-neox-20b': 'gpt-neox-20B',
               }

method_names = {'gptzero': 'GPTZero',
                'likelihood': 'Likelihood',
                'entropy': 'Entropy',
                'rank': 'Rank',
                'logrank': 'LogRank',
                'dna_gpt': 'DNA-GPT',
                'perturbation_100': 'DetectGPT',
                'sampling_discrepancy_analytic': 'Fast-Detect',
                'pde_entropy_geometric': 'Entropy',
                'pde_rank_geometric': 'Rank',
                'pde_logrank_geometric': 'LogRank',
                'pde_fastdetect_geometric': 'Fast-Detect',
                'pde_entropy_zipfian': 'Entropy',
                'pde_rank_zipfian': 'Rank',
                'pde_logrank_zipfian': 'LogRank',
                'pde_fastdetect_zipfian': 'Fast-Detect',
                'pde_entropy_mlp': 'Entropy',
                'pde_rank_mlp': 'Rank',
                'pde_logrank_mlp': 'LogRank',
                'pde_fastdetect_mlp': 'Fast-Detect',
                'pde_rank': 'Rank',
                'pde_logrank': 'LogRank',
                'pde_fastdetect': 'Fast-Detect',
                }

def save_lines(lines, file):
    with open(file, 'w') as fout:
        fout.write('\n'.join(lines))

def get_auroc(result_file):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        return res['metrics']['roc_auc']

def get_fpr_tpr(result_file):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        return res['metrics']['fpr'], res['metrics']['tpr']

def get_resultfiles(result_path, method, scoring_model, source_models, datasets):
    result_files = []
    for source_model in source_models:
        for dataset in datasets:
            if scoring_model is None:
                result_file = f'{result_path}/{dataset}_{source_model}.{method}.json'
            else:
                result_file = f'{result_path}/{dataset}_{source_model}.{scoring_model}.{method}.json'
            if os.path.exists(result_file):
                result_files.append(result_file)
    return result_files

def get_auroc_of_dataset(result_path, method, source_model, scoring_model, dataset):
    if scoring_model is None:
        result_file = f'{result_path}/{dataset}_{source_model}.{method}.json'
    else:
        result_file = f'{result_path}/{dataset}_{source_model}.{scoring_model}.{method}.json'
    if os.path.exists(result_file):
        auroc = get_auroc(result_file)
    else:
        auroc = 0.0
    return auroc

def get_auroc_of_mixture(result_files):
    real = []
    samples = []
    for result_file in result_files:
        with open(result_file, 'r') as fin:
            res = json.load(fin)
            real.extend(res['predictions']['real'])
            samples.extend(res['predictions']['samples'])
    # get auroc
    if len(real) > 0 and len(samples) > 0:
        fpr, tpr, roc_auc = get_roc_metrics(real, samples)
        return roc_auc
    else:
        return 0.0

def get_auroc_of_datasets(result_path, method, source_model, scoring_model, datasets):
    cols = []
    for dataset in datasets:
        cols.append(get_auroc_of_dataset(result_path, method, source_model, scoring_model, dataset))
    # calculate mixture of the datasets
    auroc = get_auroc_of_mixture(get_resultfiles(result_path, method, scoring_model, [source_model], datasets))
    cols.append(auroc)
    return cols

def get_tpr_of_dataset(result_path, method, source_model, scoring_model, dataset):
    fpr_fix = 0.01
    if scoring_model is None:
        result_file = f'{result_path}/{dataset}_{source_model}.{method}.json'
    else:
        result_file = f'{result_path}/{dataset}_{source_model}.{scoring_model}.{method}.json'
    if os.path.exists(result_file):
        fprs, tprs = get_fpr_tpr(result_file)
        for fpr, tpr in zip(fprs, tprs):
            if fpr >= fpr_fix:
                return tpr
    else:
        return 0.0

def get_tpr_of_datasets(result_path, method, source_model, scoring_model, datasets):
    cols = []
    for dataset in datasets:
        cols.append(get_tpr_of_dataset(result_path, method, source_model, scoring_model, dataset))
    cols.append(np.mean(cols))
    return cols

def get_results_ablation_topk(args):
    estimators = ['geometric', 'zipfian', 'mlp']
    datasets = ['xsum', 'writing', 'pubmed']
    scoring_models = ['gpt-4-1106', 'gpt-35-turbo-1106']
    methods = ['pde_fastdetect', 'pde_logrank']
    source_model = 'gpt-4'
    topks = ['top1', 'top3', 'top5', 'top7', 'top10']

    results = {}
    for estimator in estimators:
        result = {}
        for method in methods:
            for scoring_model in scoring_models:
                key = f'{method_names[f'{method}_{estimator}']} ({model_names[scoring_model]})'
                vals = []
                for topk in topks:
                    cols = get_auroc_of_datasets(f'{args.result_path}_{topk}', f'{method}_{estimator}', source_model, scoring_model, datasets)
                    vals.append(cols[-1])  # the average
                result[key] = vals
        results[estimator_names[estimator]] = result
    return results

def get_results_openllm_topk(args):
    methods = ['pde_fastdetect', 'pde_logrank', 'pde_rank']
    estimators = ['geometric', 'zipfian', 'mlp']
    datasets = ['xsum']  # , 'writing', 'pubmed'
    scoring_model = 'gpt-neo-2.7B'
    source_models = ['gpt-4'] # 'gpt-3.5-turbo',
    topks = ['top1', 'top2', 'top3', 'top5', 'top7', 'top10']

    results = {}
    for method in methods:
        result = {'Real': {}}
        for estimator in estimators:
            vals = {}  # each item represents an average value over source models
            for topk in topks:
                val = []  # each item represents an average value over datasets
                for source_model in source_models:
                    cols = get_auroc_of_datasets(f'{args.result_path}_openllm_{topk}', f'{method}_{estimator}', source_model, scoring_model, datasets)
                    val.append(cols[-1])
                vals[topk] = np.mean(val)
            result[estimator_names[estimator]] = vals
        results[method_names[method]] = result
    # baseline method
    methods = [('sampling_discrepancy_analytic', f'{scoring_model}_{scoring_model}'),
               ('logrank', scoring_model),
               ('rank', scoring_model)]
    for method, scoring_model in methods:
        result = results[method_names[method]]
        val = []  # each item represents an average value over datasets
        for source_model in source_models:
            cols = get_auroc_of_datasets(f'{args.result_path}_openllm', method, source_model, scoring_model, datasets)
            val.append(cols[-1])
        result['Real'] = {'real': np.mean(val)}
    return results

def get_results_ablation_ranksize(args):
    estimators = ['geometric', 'zipfian']
    datasets = ['xsum', 'writing', 'pubmed']
    scoring_models = ['gpt-4-1106', 'gpt-35-turbo-1106']
    methods = ['pde_fastdetect', 'pde_logrank']
    source_model = 'gpt-4'
    ranksizes = [i * 100 for i in range(1, 11)]

    results = {}
    for estimator in estimators:
        result = {}
        for method in methods:
            for scoring_model in scoring_models:
                key = f'{method_names[f'{method}_{estimator}']} ({model_names[scoring_model]})'
                vals = []
                for ranksize in ranksizes:
                    cols = get_auroc_of_datasets(f'{args.result_path}_ranksize{ranksize}', f'{method}_{estimator}', source_model, scoring_model, datasets)
                    vals.append(cols[-1])  # the average
                result[key] = vals
        results[estimator_names[estimator]] = result
    return results


def get_results_ablation_estimator(args):
    estimators = ['geometric', 'zipfian', 'mlp']
    datasets = ['xsum', 'writing', 'pubmed']
    source_models = ['gpt-3.5-turbo', 'gpt-4', 'claude-3-sonnet-20240229', 'claude-3-opus-20240229', 'gemini-1.5-pro']
    scoring_models = ['gpt-35-turbo-1106', 'gpt-4-1106']

    results = {'rows': [model_names[scoring_model] for scoring_model in scoring_models],
               'cols': [dataset_names[dataset] for dataset in datasets],
               'cells': [[None for _ in datasets] for _ in scoring_models]}
    for i, scoring_model in enumerate(scoring_models):
        for j, dataset in enumerate(datasets):
            result = {}
            for estimator in estimators:
                method = f'pde_fastdetect_{estimator}'
                aurocs = []
                for source_model in source_models:
                    auroc = get_auroc_of_dataset(args.result_path, method, source_model, scoring_model, dataset)
                    aurocs.append(auroc)
                result[estimator_names[estimator]] = np.mean(aurocs)
            results['cells'][i][j] = result
    return results


def get_results_auroc_curve(args):
    def load_experiment(json_file):
        with open(json_file, 'r') as fin:
            return json.load(fin)

    def get_mixture_metrics(datasets, source_model, method, scoring_model):
        real = []
        samples = []
        for dataset in datasets:
            json_file = f'{args.result_path}/{dataset}_{source_model}.{scoring_model}.{method}.json'
            results = load_experiment(json_file)
            real.extend(results['predictions']['real'])
            samples.extend(results['predictions']['samples'])
        fpr, tpr, roc_auc = get_roc_metrics(real, samples)
        return fpr, tpr

    datasets = ['xsum', 'writing', 'pubmed']
    source_models = ['gpt-3.5-turbo', 'gpt-4', 'claude-3-sonnet-20240229', 'claude-3-opus-20240229', 'gemini-1.5-pro']
    methods = [('likelihood', 'gpt-35-turbo-1106'),
               ('sampling_discrepancy_analytic', 'gpt-j-6B_gpt-neo-2.7B'),
               ('pde_fastdetect_geometric', 'babbage-002'),               ('pde_fastdetect_geometric', 'gpt-35-turbo-1106'),
               ('pde_fastdetect_geometric', 'gpt-4-1106'),]

    results = {}
    for source_model in source_models:
        result = {}
        for method, scoring_model in methods:
            key = f'{method_names[method]} ({model_names[scoring_model]})'
            result[key] = get_mixture_metrics(datasets, source_model, method, scoring_model)
        results[model_names[source_model]] = result
    return results


def report_main_results(args, type):
    datasets = ['xsum', 'writing', 'pubmed']

    if type == 'main':
        source_models = ['gpt-3.5-turbo', 'gpt-4', 'claude-3-sonnet-20240229', 'claude-3-opus-20240229', 'gemini-1.5-pro']
    elif type == 'ext1':
        source_models = ['gpt-4', 'gemini-1.5-pro']
    elif type == 'ext2':
        source_models = ['claude-3-sonnet-20240229', 'claude-3-opus-20240229']

    headers1 = ['--'] + [model_names[model] for model in source_models]
    headers2 = ['Method'] + [dataset_names[dataset] for dataset in datasets] + ['Mix3']
    print(' '.join(headers1))
    print(' '.join(headers2))

    section_methods = {
        'Trained Detectors and Commercial Systems': [
            ('gptzero', None)],
        'Zero-Shot Detectors Using Open-Source LLMs': [
            ('likelihood', 'gpt-neo-2.7B'),
            ('entropy', 'gpt-neo-2.7B'),
            ('rank', 'gpt-neo-2.7B'),
            ('logrank', 'gpt-neo-2.7B'),
            ('dna_gpt', 'gpt-neo-2.7B'),
            ('perturbation_100', 't5-11b_gpt-neo-2.7B'),
            ('sampling_discrepancy_analytic', 'gpt-j-6B_gpt-neo-2.7B'),
            ('sampling_discrepancy_analytic', 'phi-2_phi-2'),
            ('sampling_discrepancy_analytic', 'qwen2.5-7b_qwen2.5-7b'),
            ('sampling_discrepancy_analytic', 'llama3-8b_llama3-8b'),
        ],
        'Zero-Shot Detectors Using Proprietary LLMs': [
            ('likelihood', 'gpt-35-turbo-1106'),
            ('dna_gpt', 'gpt-35-turbo-1106')],
        'PDE using Geometric': [
            (f'pde_entropy_geometric', 'gpt-35-turbo-1106'),
            (f'pde_rank_geometric', 'gpt-35-turbo-1106'),
            (f'pde_logrank_geometric', 'gpt-35-turbo-1106'),
            (f'pde_fastdetect_geometric', 'babbage-002'),
            (f'pde_fastdetect_geometric', 'davinci-002'),
            (f'pde_fastdetect_geometric', 'gpt-35-turbo-1106'),
            (f'pde_fastdetect_geometric', 'gpt-4-1106')],
        'PDE using Zipfian': [
            (f'pde_fastdetect_zipfian', 'gpt-35-turbo-1106'),
            (f'pde_fastdetect_zipfian', 'gpt-4-1106')],
        'PDE using MLP': [
            (f'pde_fastdetect_mlp', 'gpt-35-turbo-1106'),
            (f'pde_fastdetect_mlp', 'gpt-4-1106')],
    }

    for section, methods in section_methods.items():
        print(section)
        for method, scoring_model in methods:
            method_name = method_names[method]
            model_name = '' if scoring_model is None else f' ({model_names[scoring_model]})'
            cols = []
            avgs = []
            for idx, source_model in enumerate(source_models):
                ss = get_auroc_of_datasets(args.result_path, method, source_model, scoring_model, datasets)
                if idx == 0 or len(source_models) <= 2:
                    cols.extend(ss)
                    avgs.append(ss[-1])
                else:
                    cols.append(ss[-1])
                    avgs.append(ss[-1])
            if len(avgs) > 2:
                cols.append(np.mean(avgs))
            cols = [f'{col:.4f}' if col != 0 else '-' for col in cols]
            print(f'{method_name}{model_name}', '&', ' & '.join(cols), '\\\\')

def report_langs_results(args):
    datasets = ['chinese', 'russian', 'urdu', 'indonesian', 'arabic', 'bulgarian']
    source_model = 'chatgpt'
    section_methods = {
        'Existing Methods': [
            ('sampling_discrepancy_analytic', 'gpt-j-6B_gpt-neo-2.7B'),
            ('sampling_discrepancy_analytic', 'phi-2_phi-2'),
            ('sampling_discrepancy_analytic', 'qwen2.5-7b_qwen2.5-7b'),
            ('sampling_discrepancy_analytic', 'llama3-8b_llama3-8b')],
        'PDE using Geometric': [
            (f'pde_fastdetect_geometric', 'babbage-002'),
            (f'pde_fastdetect_geometric', 'davinci-002'),
            (f'pde_fastdetect_geometric', 'gpt-35-turbo-1106')],
    }

    for section, methods in section_methods.items():
        print(section)
        for method, scoring_model in methods:
            method_name = method_names[method]
            model_name = '' if scoring_model is None else f' ({model_names[scoring_model]})'
            cols = []
            # ss[-1] is the result for Mix6, which is a mixture of the six languages
            ss = get_auroc_of_datasets(args.result_path, method, source_model, scoring_model, datasets)
            cols.extend(ss)
            cols = [f'{col:.4f}' if col != 0 else '-' for col in cols]
            print(f'{method_name}{model_name}', '&', ' & '.join(cols), '\\\\')


def report_attack_results(args):
    datasets = ['xsum', 'writing', 'pubmed']
    source_models = ['gpt-3.5-turbo']

    headers1 = ['--'] + [model_names[model] for model in source_models]
    headers2 = ['Method'] + [dataset_names[dataset] for dataset in datasets] + ['Avg.']
    print(' '.join(headers1))
    print(' '.join(headers2))

    section_methods = {
        'Baselines': [
            ('sampling_discrepancy_analytic', 'gpt-j-6B_gpt-neo-2.7B'),
            ('sampling_discrepancy_analytic', 'phi-2_phi-2'),
            ('sampling_discrepancy_analytic', 'qwen2.5-7b_qwen2.5-7b'),
            ('sampling_discrepancy_analytic', 'llama3-8b_llama3-8b'),
            ('likelihood', 'gpt-35-turbo-1106')],
        'PDE using Geometric': [
            (f'pde_fastdetect_geometric', 'babbage-002'),
            (f'pde_fastdetect_geometric', 'davinci-002'),
            (f'pde_fastdetect_geometric', 'gpt-35-turbo-1106')]
    }

    for section, methods in section_methods.items():
        print(section)
        for method, scoring_model in methods:
            method_name = method_names[method]
            model_name = '' if scoring_model is None else f' ({model_names[scoring_model]})'
            cols = []
            avgs = []
            for source_model in source_models:
                ss = get_tpr_of_datasets(args.result_path, method, source_model, scoring_model, datasets)
                cols.extend(ss)
                avgs.append(ss[-1])

            if len(avgs) > 2:
                cols.append(np.mean(avgs))
            cols = [f'{col*100:.1f}' if col != 0 else '-' for col in cols]
            print(f'{method_name}{model_name}', '&', ' & '.join(cols), '\\\\')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default="./exp_main/results/")
    parser.add_argument('--report_name', type=str, default="main_results")
    # parser.add_argument('--result_path', type=str, default="./exp_langs/results/")
    # parser.add_argument('--report_name', type=str, default="langs_results")
    # parser.add_argument('--result_path', type=str, default="./exp_attack/results_lex60_order0/")
    # parser.add_argument('--report_name', type=str, default="attack_results")
    args = parser.parse_args()

    if args.report_name == 'main_results':
        report_main_results(args, 'main')
        report_main_results(args, 'ext1')
        report_main_results(args, 'ext2')
    elif args.report_name == 'langs_results':
        report_langs_results(args)
    elif args.report_name == 'attack_results':
        report_attack_results(args)
