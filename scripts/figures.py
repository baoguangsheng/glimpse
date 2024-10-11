# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
from os import path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def auroc_linechat_log(args):
    from report_results import get_results_auroc_curve
    results = get_results_auroc_curve(args)
    colors = ['tab:brown', 'tab:grey', 'tab:green', 'tab:blue', 'tab:orange', ]
    lws = [1, 1, 1, 1, 1]

    # plot
    nrows = 1
    ncols = len(results)
    plt.clf()
    fig = plt.figure(figsize=(2.5 * ncols, 2.3 * nrows))
    grids = fig.add_gridspec(nrows, ncols)
    axs = grids.subplots(sharex=True, sharey=True)

    sources = list(results.keys())
    methods = list(list(results.values())[0].keys())
    for j in range(ncols):
        source = sources[j]
        curves = results[source]
        for method, color, lw in zip(methods, colors, lws):
            xs, ys = curves[method]
            axs[j].plot(xs, ys, color=color, lw=lw, label=method)
        axs[j].plot([k/100 for k in range(101)], [k/100 for k in range(101)], color='black', lw=1, linestyle='--')
        axs[j].plot([0.01 for _ in range(2)], [k for k in range(2)], color='red', lw=0.5, linestyle='--')
        axs[j].plot([0.1 for _ in range(2)], [k for k in range(2)], color='red', lw=0.5, linestyle='--')
        axs[j].set_title(source)
        axs[j].set_xlabel('False Positive Rate')

    axs[0].set_ylabel('True Positive Rate')

    plt.figlegend(labels=methods, loc='lower center', fontsize=9, ncol=5, handlelength=2)
    plt.ylim(0.0, 1.05)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xscale('log')
    plt.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.subplots_adjust(bottom=0.32)
    plt.savefig(path.join(args.output_path, 'auroc_curve_log.pdf'))

def ablation_topk_linechart(args):
    from report_results import get_results_ablation_topk
    results = get_results_ablation_topk(args)

    colors = ['tab:orange', 'tab:blue', 'tab:pink', 'tab:olive']
    markers = ['*', '.', '^', '+']
    lws = [1, 1, 1, 1]
    xs = ['1', '3', '5', '7', '10']

    # plot
    nrows = 1
    ncols = 3
    plt.clf()
    fig = plt.figure(figsize=(2 * ncols, 2 * nrows))
    grids = fig.add_gridspec(nrows, ncols)
    axs = grids.subplots(sharex=True, sharey=True)

    distribs = list(results.keys())
    for j in range(ncols):
        distrib_name = distribs[j]
        labels = list(results[distrib_name].keys())
        for label, color, marker, lw in zip(labels, colors, markers, lws):
            ys = results[distrib_name][label]
            axs[j].set_axisbelow(True)
            axs[j].grid(axis='y', color='lightgrey', lw=0.2, linestyle='-')
            axs[j].plot(xs, ys, color=color, marker=marker, lw=lw, label=label)
            axs[j].set_title(distrib_name)
            axs[j].set_xlabel('Top-$K$')

    axs[0].set_ylabel('AUROC')

    # axs[0].legend(loc="lower right", fontsize=5, ncol=1)
    plt.figlegend(labels=labels, loc='lower center', fontsize=6.5, ncol=4, handlelength=1.5)

    plt.ylim(0.75, 1.00)
    plt.yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    plt.xticks(xs)
    plt.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.subplots_adjust(bottom=0.35)
    plt.savefig(path.join(args.output_path, 'ablation_topk_linechart.pdf'))


def ablation_ranksize_linechart(args):
    from report_results import get_results_ablation_ranksize
    results = get_results_ablation_ranksize(args)

    colors = ['tab:orange', 'tab:blue', 'tab:pink', 'tab:olive']
    markers = ['*', '.', '^', '+']
    lws = [1, 1, 1, 1]
    xs = [i * 100 for i in range(1, 11)]

    # plot
    nrows = 1
    ncols = 2
    plt.clf()
    fig = plt.figure(figsize=(2.5 * ncols, 2 * nrows))
    grids = fig.add_gridspec(nrows, ncols)
    axs = grids.subplots(sharex=True, sharey=True)

    distribs = list(results.keys())
    for j in range(ncols):
        distrib_name = distribs[j]
        labels = list(results[distrib_name].keys())
        for label, color, marker, lw in zip(labels, colors, markers, lws):
            ys = results[distrib_name][label]
            axs[j].set_axisbelow(True)
            axs[j].grid(axis='y', color='lightgrey', lw=0.2, linestyle='-')
            axs[j].plot(xs, ys, color=color, marker=marker, lw=lw, label=label)
            axs[j].set_title(distrib_name)
            axs[j].set_xlabel('Rank List Size')

    axs[0].set_ylabel('AUROC')
    axs[0].legend(loc="lower right", fontsize=5, ncol=1)

    # plt.figlegend(labels=datasets, loc='upper left', fontsize=7, ncol=1, handlelength=2)

    plt.ylim(0.70, 1.00)
    plt.yticks([0.70, 0.80, 0.90, 1.0])
    plt.xticks([100, 300, 500, 700, 900])
    plt.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.subplots_adjust(bottom=0.25)
    plt.savefig(path.join(args.output_path, 'ablation_ranksize_linechart.pdf'))


def ablation_prompt_barchart(args):
    # top-k results on GPT-4 generations
    results = {
        'Fast-Detect(Babbage)': [0.8921, 0.9034, 0.9205, 0.9299, 0.9251],
        # 'Fast-Detect(Davinci)': [0.9047, 0.9360, 0.9298, 0.9268, 0.9274],
        'Fast-Detect(GPT-3.5)': [0.9071, 0.9589, 0.9541, 0.9545, 0.9511],
        'Fast-Detect(GPT-4)': [0.7289, 0.8239, 0.8775, 0.9341, 0.9682],
    }
    # methods = ['Fast-Detect(Babbage)', 'Fast-Detect(Davinci)', 'Fast-Detect(GPT-3.5)', 'Fast-Detect(GPT-4)']
    methods = ['Fast-Detect(Babbage)', 'Fast-Detect(GPT-3.5)', 'Fast-Detect(GPT-4)']
    colors = ['tab:grey', 'tab:brown', 'tab:blue', 'tab:green', 'tab:orange']
    xs = ['0', '1', '2', '3', '4']

    # plot
    nrows = 1
    ncols = len(methods)
    plt.clf()
    fig = plt.figure(figsize=(2.5 * ncols, 2 * nrows))
    grids = fig.add_gridspec(nrows, ncols)
    axs = grids.subplots(sharex=True, sharey=True)

    for j in range(ncols):
        method_name = methods[j]
        ys = results[method_name]
        axs[j].set_axisbelow(True)
        axs[j].grid(axis='y', color='lightgrey', lw=0.2, linestyle='-')
        axs[j].bar(xs, ys, color=colors)
        for x, y in zip(xs, ys):
            axs[j].text(x, y, y, ha='center', va='bottom', fontsize=6)
        axs[j].set_title(method_name)
        axs[j].set_xlabel('Prompt')

    axs[0].set_ylabel('AUROC')
    # axs[0].legend(loc="upper left", fontsize=6, ncol=1)

    # plt.figlegend(labels=methods, loc='upper left', fontsize=7, ncol=1, handlelength=2)

    plt.ylim(0.60, 1.05)
    plt.yticks([0.60, 0.70, 0.80, 0.90, 1.0])
    plt.xticks(xs)
    plt.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.subplots_adjust(bottom=0.25)
    plt.savefig(path.join(args.output_path, 'ablation_prompt_barchart.pdf'))

def ablation_estimator_barchart(args):
    from report_results import get_results_ablation_estimator
    results = get_results_ablation_estimator(args)

    colors = ['tab:green', 'tab:blue', 'tab:orange']
    rows = results['rows']
    cols = results['cols']
    cells = results['cells']

    # plot
    nrows = 1
    ncols = len(cols)
    plt.clf()
    fig = plt.figure(figsize=(2 * ncols, 2 * nrows))
    grids = fig.add_gridspec(nrows, ncols)
    axs = grids.subplots(sharex=True, sharey=True)

    for i in range(len(rows)):
        for j in range(len(cols)):
            result = cells[i][j]
            labels = list(result.keys())
            xs = np.arange(len(cols))
            xs = i + xs * 0.25
            ys = list(result.values())
            axs[j].set_axisbelow(True)
            axs[j].grid(axis='y', color='lightgrey', lw=0.2, linestyle='-')
            axs[j].bar(xs, ys, width=0.20, color=colors, label=labels)
            axs[j].set_title(cols[j])
            axs[j].set_xlabel('Scoring Model')
        axs[0].set_ylabel('AUROC')

    # axs[2].legend(loc="upper right", labels=labels, fontsize=7, ncol=1)
    plt.figlegend(labels=labels, loc='lower center', fontsize=8, ncol=3, handlelength=2)


    plt.ylim(0.90, 1.00)
    plt.yticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.0])
    plt.xticks([0.25, 1.25], rows)
    plt.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.subplots_adjust(bottom=0.35)
    plt.savefig(path.join(args.output_path, f'ablation_estimator_barchart.pdf'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default="./exp_main/results")
    parser.add_argument('--output_path', type=str, default="./exp_analysis")
    args = parser.parse_args()

    ablation_topk_linechart(args)
    ablation_ranksize_linechart(args)
    ablation_prompt_barchart(args)
    ablation_estimator_barchart(args)
    auroc_linechat_log(args)
