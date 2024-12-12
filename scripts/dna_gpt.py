# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os.path

import numpy as np
import time
import re
import tqdm
import argparse
import json
import torch
from data_builder import load_data
from metrics import get_roc_metrics, get_precision_recall_metrics
import custom_datasets

class PrefixSampler:
    def __init__(self, args):
        self.args = args
        if self.is_blackbox():
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                azure_endpoint=args.api_endpoint,
                api_key=args.api_key,
                api_version=args.api_version)
        else:
            from model import load_tokenizer, load_model
            self.base_tokenizer = load_tokenizer(args.base_model_name, args.cache_dir)
            self.base_model = load_model(args.base_model_name, args.device, args.cache_dir)

    def is_blackbox(self):
        return self.args.base_model_name.startswith('gpt-3') \
                or self.args.base_model_name.startswith('gpt-4')

    def get_prefix(self, text):
        if self.args.dataset == 'pubmed':
            pubmed_sep = ' Answer:'
            return text[:text.index(pubmed_sep) + len(pubmed_sep)]
        else:
            words = text.split(' ')
            return ' '.join(words[: int(len(words) * self.args.truncate_ratio)])

    def get_suffix(self, text, text_orig=None):
        if self.args.dataset == 'pubmed':
            pubmed_sep = ' Answer:'
            return text[text.index(pubmed_sep) + len(pubmed_sep):]
        else:
            start = int(len(text_orig.split(' ') if text_orig else text.split(' ')) * self.args.truncate_ratio)
            words = text.split(' ')
            return ' '.join(words[start:])

    def _sample_from_api(self, texts, min_words=55):
        assert all([t == texts[0] for t in texts])  # all are the same
        prefix = self.get_prefix(texts[0])
        batch_size = len(texts)

        tries = 0
        m = 0
        while m < min_words:
            if tries != 0:
                print()
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

            response = self.client.chat.completions.create(model=args.base_model_name,
                          messages=[{"role": "system",
                                     "content": "You are a helpful assistant that continues the passage from the sentences provided."},
                                    {"role": "assistant", "content": prefix},
                                    ],
                          temperature=args.temperature,
                          max_tokens=200,
                          n=batch_size)
            gens = [choice.message.content for choice in response.choices]
            gens = [prefix + ' ' + gen for gen in gens]
            m = min(len(x.split()) for x in gens)
            tries += 1
        return gens

    def _sample_from_model(self, texts, min_words=55):
        # encode each text as a list of token ids
        texts = [self.get_prefix(t) for t in texts]
        all_encoded = self.base_tokenizer(texts, return_tensors="pt", padding=True).to(self.args.device)

        self.base_model.eval()
        decoded = ['' for _ in range(len(texts))]

        # sample from the model until we get a sample with at least min_words words for each example
        # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
        tries = 0
        m = 0
        while m < min_words:
            if tries != 0:
                print()
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

            sampling_kwargs = {'temperature': self.args.temperature}
            min_length = 50 if self.args.dataset in ['pubmed'] else 150
            outputs = self.base_model.generate(**all_encoded, min_length=min_length, max_length=200, do_sample=True,
                                               **sampling_kwargs, pad_token_id=self.base_tokenizer.eos_token_id,
                                               eos_token_id=self.base_tokenizer.eos_token_id)
            decoded = self.base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            m = min(len(x.split()) for x in decoded)
            tries += 1

        return decoded

    def generate_samples(self, raw_data, batch_size):
        # trim to shorter length
        def _trim_to_shorter_length(texta, textb):
            # truncate to shorter of o and s
            shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
            texta = ' '.join(texta.split(' ')[:shorter_length])
            textb = ' '.join(textb.split(' ')[:shorter_length])
            return texta, textb

        def _truncate_to_substring(text, substring, idx_occurrence):
            # truncate everything after the idx_occurrence occurrence of substring
            assert idx_occurrence > 0, 'idx_occurrence must be > 0'
            idx = -1
            for _ in range(idx_occurrence):
                idx = text.find(substring, idx + 1)
                if idx == -1:
                    return text
            return text[:idx]

        data = {
            "original": [],
            "sampled": [],
        }

        assert len(raw_data) % batch_size == 0
        for batch in range(len(raw_data) // batch_size):
            # print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
            original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
            if self.is_blackbox():
                sampled_text = self._sample_from_api(original_text, min_words=30 if self.args.dataset in ['pubmed'] else 55)
            else:
                sampled_text = self._sample_from_model(original_text, min_words=30 if self.args.dataset in ['pubmed'] else 55)

            for o, s in zip(original_text, sampled_text):
                if self.args.dataset == 'pubmed':
                    s = _truncate_to_substring(s, 'Question:', 2)
                    o = o.replace(custom_datasets.SEPARATOR, ' ')

                o, s = _trim_to_shorter_length(o, s)

                # add to the data
                data["original"].append(o)
                data["sampled"].append(s)

        return data

def get_regen_samples(sampler, text):
    data = [text] * sampler.args.regen_number
    data = sampler.generate_samples(data, batch_size=sampler.args.batch_size)
    return data['sampled']

def get_dna_gpt_whitebox(sampler, text):
    def get_likelihood(logits, labels, pad_index):
        labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
        lprobs = torch.log_softmax(logits, dim=-1)
        log_likelihood = lprobs.gather(dim=-1, index=labels)
        mask = labels != pad_index
        log_likelihood = (log_likelihood * mask).sum(dim=1) / mask.sum(dim=1)
        return log_likelihood.squeeze(-1)

    def get_log_prob(sampler, text):
        tokenized = sampler.base_tokenizer(text, return_tensors="pt", padding=True).to(sampler.args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = sampler.base_model(**tokenized).logits[:, :-1]
            return get_likelihood(logits_score, labels, sampler.base_tokenizer.pad_token_id)

    def get_log_probs(sampler, texts):
        batch_size = sampler.args.batch_size
        batch_lprobs = []
        for batch in range(len(texts) // batch_size):
            tokenized = sampler.base_tokenizer(texts[batch * batch_size:(batch + 1) * batch_size], return_tensors="pt",
                                               padding=True).to(sampler.args.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits_score = sampler.base_model(**tokenized).logits[:, :-1]
                lprobs = get_likelihood(logits_score, labels, sampler.base_tokenizer.pad_token_id)
                batch_lprobs.append(lprobs)
        return torch.cat(batch_lprobs, dim=0)

    lprob = get_log_prob(sampler, text)
    regens = get_regen_samples(sampler, text)
    lprob_regens = get_log_probs(sampler, regens)
    wscore = lprob[0] - lprob_regens.mean()
    return wscore.item()

def get_dna_gpt_blackbox(sampler, text):
    from rouge_score.rouge_scorer import _create_ngrams
    from nltk.stem.porter import PorterStemmer
    import six

    stemmer = PorterStemmer()

    def tokenize(text, stemmer, stopwords=[]):
        """Tokenize input text into a list of tokens.

        This approach aims to replicate the approach taken by Chin-Yew Lin in
        the original ROUGE implementation.

        Args:
        text: A text blob to tokenize.
        stemmer: An optional stemmer.

        Returns:
        A list of string tokens extracted from input text.
        """

        # Convert everything to lowercase.
        text = text.lower()
        # Replace any non-alpha-numeric characters with spaces.
        text = re.sub(r"[^a-z0-9]+", " ", six.ensure_str(text))

        tokens = re.split(r"\s+", text)
        if stemmer:
            # Only stem words more than 3 characters long.
            tokens = [stemmer.stem(x) if len(x) > 3 else x for x in tokens if x not in stopwords]

        # One final check to drop any empty or invalid tokens.
        tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", six.ensure_str(x))]

        return tokens
    def get_score_ngrams(target_ngrams, prediction_ngrams):
        intersection_ngrams_count = 0
        ngram_dict = {}
        for ngram in six.iterkeys(target_ngrams):
            intersection_ngrams_count += min(target_ngrams[ngram],
                                            prediction_ngrams[ngram])
            ngram_dict[ngram] = min(target_ngrams[ngram], prediction_ngrams[ngram])
        target_ngrams_count = sum(target_ngrams.values()) # prediction_ngrams
        return intersection_ngrams_count / max(target_ngrams_count, 1), ngram_dict

    def get_ngram_info(article_tokens, summary_tokens, _ngram):
        article_ngram = _create_ngrams(article_tokens , _ngram)
        summary_ngram = _create_ngrams(summary_tokens , _ngram)
        ngram_score, ngram_dict = get_score_ngrams(article_ngram, summary_ngram)
        return ngram_score, ngram_dict, sum(ngram_dict.values())

    def N_gram_detector(ngram_n_ratio):
        score = 0
        non_zero = []

        for idx, key in enumerate(ngram_n_ratio):
            if idx in range(3) and 'score' in key or 'ratio' in key:
                score += 0. * ngram_n_ratio[key]
                continue
            if 'score' in key or 'ratio' in key:
                score += (idx + 1) * np.log((idx + 1)) * ngram_n_ratio[key]
                if ngram_n_ratio[key] != 0:
                    non_zero.append(idx + 1)
        return score / (sum(non_zero) + 1e-8)

    suffix = sampler.get_suffix(text)
    suffix_tokens = tokenize(suffix, stemmer=stemmer)
    regens = get_regen_samples(sampler, text)
    regens = [sampler.get_suffix(regen, text) for regen in regens]

    scores = []
    for regen in regens:
        item = {}
        regen_tokens = tokenize(regen, stemmer=stemmer)
        for _ngram in range(1, 25):
            ngram_score, ngram_dict, overlap_count = get_ngram_info(suffix_tokens, regen_tokens, _ngram)
            item['truncate_ngram_{}_score'.format(_ngram)] = ngram_score / len(regen_tokens)
            item['truncate_ngram_{}_count'.format(_ngram)] = overlap_count
        score = N_gram_detector(item)
        scores.append(score)
    score = np.sum(scores)
    return score

def get_dna_gpt(sampler, text):
    if sampler.is_blackbox():
        return get_dna_gpt_blackbox(sampler, text)
    return get_dna_gpt_whitebox(sampler, text)

def experiment(args):
    sampler = PrefixSampler(args)
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    # evaluate criterion
    name = "dna_gpt"
    criterion_fn = get_dna_gpt

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        try:
            # original text
            original_crit = criterion_fn(sampler, original_text)
            # sampled text
            sampled_crit = criterion_fn(sampler, sampled_text)
            # result
            results.append({"original": original_text,
                            "original_crit": original_crit,
                            "sampled": sampled_text,
                            "sampled_crit": sampled_crit})
        except Exception as e:
            print(e)

    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in results],
                   'samples': [x["sampled_crit"] for x in results]}
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    # results
    results_file = f'{args.output_file}.{name}.json'
    results = { 'name': f'{name}_threshold',
                'info': {'n_samples': n_samples},
                'predictions': predictions,
                'raw_results': results,
                'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                'loss': 1 - pr_auc}
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_test/results/pubmed_davinci")
    parser.add_argument('--dataset', type=str, default="pubmed")
    parser.add_argument('--dataset_file', type=str, default="./exp_test/data/pubmed_davinci")
    parser.add_argument('--truncate_ratio', type=float, default=0.5)
    parser.add_argument('--regen_number', type=int, default=10)
    parser.add_argument('--base_model_name', type=str, default="gpt2")
    parser.add_argument('--temperature', type=float, default=0.7)
    # black-box model settings
    parser.add_argument('--api_endpoint', type=str, default='https://xxxx.openai.azure.com/')
    parser.add_argument('--api_key', type=str, default='xxxxxxxx')
    parser.add_argument('--api_version', type=str, default='2024-02-15-preview')
    # white-box model settings
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)
