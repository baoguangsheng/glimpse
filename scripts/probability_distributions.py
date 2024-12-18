# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import numpy as np
import torch
from mlp_distribution_model import MlpDistributionModel


def safe_log(prob):
    return np.log(np.array(prob) + 1e-8)

class GeometricDistribution:
    '''
    Top-K probabilities: p_1, p_2, ..., p_K
    Estimated probabilities: Pr(X=k) = p_K * lambda ^ (k - K), for k > K.
    '''
    def __init__(self, top_k, rank_size):
        self.name = "GeometricDistribution"
        self.top_k = top_k
        self.rank_size = rank_size

    def estimate_distrib_token(self, toplogprobs):
        M = self.rank_size  # assuming rank list size
        K = self.top_k  # assuming top-K tokens
        assert K <= M
        toplogprobs = sorted(toplogprobs.values(), reverse=True)
        assert len(toplogprobs) >= K
        toplogprobs = toplogprobs[:K]
        probs = np.exp(toplogprobs)  # distribution over ranks
        if probs.sum() > 1.0:
            # print(f'Warnining: Probability {probs.sum()} excels 1.0')
            probs = probs / (probs.sum() + 1e-6)
        p_K = probs[-1]  # the k-th top token
        p_rest = 1 - probs.sum()  # the rest probability mass
        _lambda = p_rest / (p_K + p_rest)  # approximate the decay factor
        if _lambda ** (M - K + 1) > 1e-6:
            # If the condition was not satisfied, use the following code to calculate the decay factor iteratively
            _lambda_old = _lambda
            last_diff = 1.0
            while True:
                _lambda0 = _lambda
                minor = _lambda ** (M - K + 1)  # the minor part
                assert p_rest > 0, f'Error: Invalid p_rest={p_rest}'
                _lambda = 1 - (_lambda - minor) * p_K / p_rest
                # check convergence
                diff = abs(_lambda - _lambda0)
                if _lambda < 0 or diff < 1e-6 or diff >= last_diff:
                    _lambda = _lambda0
                    break
                last_diff = diff
            # print(f'Warnining: Invalid lambda={_lambda_old}, re-calculate lambda={_lambda}')
        assert p_rest >= 0, f'Error: Invalid p_rest={p_rest}'
        assert 0 <= _lambda <= 1, f'Error: Invalid lambda={_lambda} calculated by p_K={p_K} and p_rest={p_rest}.'
        # estimate the probabilities of the rest tokens
        probs_rest = np.exp(safe_log(p_K) + np.arange(1, M - K + 1) * safe_log(_lambda))
        probs = np.concatenate([probs, probs_rest])
        # check total probability
        # if abs(probs.sum() - 1.0) >= 1e-2:
            # print(f'Warnining: Invalid total probability: {probs.sum()}')
        probs = probs / probs.sum()
        return probs.tolist()

class ZipfianDistribution:
    '''
    Top-K probabilities: p_1, p_2, ..., p_K
    Estimated probabilities: Pr(X=k) = p_K * (beta / (k - K + beta)) ^ alpha, for k > K.
    '''
    def __init__(self, top_k, rank_size):
        self.name = "ZipfianDistribution"
        self.top_k = top_k
        self.rank_size = rank_size
        M = rank_size
        K = top_k
        nalpha = 100
        nbeta = 100
        self.alpha_table = np.zeros((nalpha, 1))
        self.beta_table = np.zeros((1, nbeta))
        self.alpha_beta_table = np.zeros((nalpha, nbeta))
        for a in range(nalpha):
            for b in range(nbeta):
                alpha = a / 10  # alpha in (0, 10)
                beta = b / 5  # beta in (0, 20)
                series = (beta / (np.arange(1, M - K) + beta)) ** alpha
                self.alpha_beta_table[a, b] = series.sum()
                self.alpha_table[a, 0] = alpha
                self.beta_table[0, b] = beta

    def _find_alpha_beta(self, ratio):
        k1 = 1.0
        k2 = 0.001
        dist_ratio = np.square(self.alpha_beta_table - ratio)
        dist_alpha = np.square(self.alpha_table - 1)
        dist_beta = np.square(self.beta_table - 2.7)
        dist = dist_ratio + k1 * dist_alpha + k2 * dist_beta
        a, b = np.unravel_index(dist.argmin(), dist.shape)
        alpha = a / 10
        beta = b / 5
        return alpha, beta

    def estimate_distrib_token(self, toplogprobs):
        M = self.rank_size  # assuming rank list size
        K = self.top_k  # assuming top-K tokens
        toplogprobs = sorted(toplogprobs.values(), reverse=True)
        assert len(toplogprobs) >= K
        toplogprobs = toplogprobs[:K]
        probs = np.exp(toplogprobs)  # distribution over ranks
        if probs.sum() > 1.0:
            # print(f'Warnining: Probability {probs.sum()} excels 1.0')
            probs = probs / (probs.sum() + 1e-6)
        p_K = probs[-1]  # the k-th top token
        p_rest = 1 - probs.sum()  # the rest probability mass
        alpha, beta = self._find_alpha_beta(p_rest / p_K)
        assert p_rest >= 0, f'Error: Invalid p_rest={p_rest}'
        assert 0 <= alpha < 10, f'Error: Invalid alpha={alpha}'
        assert 0 <= beta < 20, f'Error: Invalid beta={beta}'
        # estimate the probabilities of the rest tokens
        probs_rest = np.exp(safe_log(p_K) + alpha * safe_log(beta / (np.arange(1, M - K + 1) + beta)))
        probs = np.concatenate([probs, probs_rest])
        # check total probability
        # if abs(probs.sum() - 1.0) >= 1e-2:
        #     print(f'Warnining: Invalid total probability: {probs.sum()}')
        probs = probs / probs.sum()
        return probs.tolist()


class MlpDistribution:
    '''
    Top-K probabilities: p_1, p_2, ..., p_K
    Estimated probabilities: Pr(X=k) = p_rest * p_MLP(k - K), for k > K.
    '''
    def __init__(self, top_k, rank_size, device):
        self.name = "MlpDistribution"
        self.top_k = top_k
        self.rank_size = rank_size
        self.device = device

    def _get_model(self):
        if getattr(self, 'model', None) is None:
            self.model = MlpDistributionModel(10, self.rank_size)
            self.model.load_state_dict(torch.load(f'./exp_main/model/mlp_distribution_model.ranksize{self.rank_size}.pt'))
            self.model.move_to(self.device)
        return self.model

    def estimate_distrib_token(self, toplogprobs):
        K = self.top_k
        toplogprobs = sorted(toplogprobs.values(), reverse=True)
        assert len(toplogprobs) >= K
        topprobs = np.exp(toplogprobs[:K]).tolist()
        model = self._get_model()
        probs = model.predict(topprobs, K)
        return probs
