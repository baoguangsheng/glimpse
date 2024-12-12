import os.path
import random
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from math import floor
import tqdm
import argparse
from data_builder import load_data
from model import load_tokenizer, load_model


def safe_log(prob):
    return torch.log(prob + 1e-6)


# Set up MLP model.
class MlpDistributionModel(nn.Module):
    def __init__(self, max_topk, rank_size, hidden_size=100):
        super().__init__()
        self.max_topk = max_topk
        self.input_size = max_topk * 2
        self.output_size = rank_size
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.output_size)
        self.ln1 = nn.LayerNorm(hidden_size)

    def move_to(self, device):
        self.device = device
        self.to(device)

    def forward(self, input, top_k):
        # mask for top-K
        mask = - torch.ones_like(input)
        mask[:, :top_k] = 1
        input[:, top_k:] = 1  # clear the input after top-K
        x = torch.cat([safe_log(input) / 10, mask], dim=-1)  # reduce the scale of input
        x = F.relu(self.ln1(self.fc1(x)))
        logits = self.fc2(x)
        # recover the full distribution
        p_top = input[:, :top_k].sum(-1, keepdims=True)
        p_rest = 1 - p_top
        probs = torch.cat([input[:, :top_k], p_rest * torch.softmax(logits[:, top_k:], dim=-1)], dim=-1)
        return probs

    def predict(self, x, top_k):
        x = x[:self.max_topk] if len(x) >= self.max_topk else x + [1] * (self.max_topk - len(x))
        x = np.array([x])
        input = torch.tensor(x).float().to(self.device)
        output = self(input, top_k)
        return output.tolist()[0]

    def get_batch(self, features, targets, batch_size=32):
        n_batch = floor(len(features) / batch_size)

        feature = features[:n_batch * batch_size]
        target = targets[:n_batch * batch_size]

        for i in range(0, n_batch * batch_size, batch_size):
            batch_feature = feature[i:i + batch_size]
            batch_target = target[i:i + batch_size]
            yield batch_feature, batch_target

    def valid(self, data):
        # Hyper-parameters.
        batch_size = 16
        valid_loss = 0
        # valid
        x_valid = [distrib[:self.max_topk] for distrib in data]
        y_valid = [distrib[:self.output_size] for distrib in data]
        generator = self.get_batch(x_valid, y_valid, batch_size)
        for idx, (feature, target) in enumerate(generator):
            feature = torch.tensor(feature, dtype=torch.float).to(self.device)
            target = torch.tensor(target, dtype=torch.float).to(self.device)
            top_k = (idx % self.max_topk) + 1
            output = self(feature, top_k)
            loss = - (target * safe_log(output)).sum(-1).mean()
            valid_loss += loss.item()
        return valid_loss / len(data)

    def train(self, data_train, data_valid, model_file):
        # Hyper-parameters.
        learning_rate = 0.00001
        batch_size = 16
        epochs = 20
        step = 0
        running_loss = 0
        print_every = 1000

        # train
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        best_valid_loss = float('inf')
        for e in range(epochs):
            random.shuffle(data_train)
            x_train = [distrib[:self.max_topk] for distrib in data_train]
            y_train = [distrib[:self.output_size] for distrib in data_train]
            generator = self.get_batch(x_train, y_train, batch_size)
            for feature, target in generator:
                step += 1
                optimizer.zero_grad()
                feature = torch.tensor(feature, dtype=torch.float).to(self.device)
                target = torch.tensor(target, dtype=torch.float).to(self.device)
                top_k = random.randint(1, args.max_topk)
                output = self(feature, top_k)
                loss = - (target * safe_log(output)).sum(-1).mean()
                loss.backward()
                optimizer.step()
                # report loss
                running_loss += loss.item()
                if step % print_every == 0:
                    print('Epoch: {}/{}...'.format(e + 1, epochs),
                          'Loss: {:.4f}'.format(running_loss / print_every / batch_size))
                    running_loss = 0

            # valid and save
            valid_loss = self.valid(data_valid)
            print(f'Epoch {e+1}, valid loss: {valid_loss:.4f}')
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.state_dict(), model_file)
                print(f'Saved model to {model_file}.')


def get_distrib_data(args):
    def _get_distrib(logits, rank_size):
        assert logits.shape[0] == 1
        logits = logits.view(-1, logits.shape[-1])
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.topk(rank_size).values
        return probs.tolist()

    datasets = ['xsum', 'writing', 'pubmed']
    sources = ['gpt-4']
    # load model
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    # eval
    distribs = []
    for dataset in datasets:
        scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
        for source in sources:
            dataset_file = os.path.join(args.dataset_path, f'{dataset}_{source}')
            data = load_data(dataset_file)
            n_samples = len(data["sampled"])
            for idx in tqdm.tqdm(range(n_samples), desc=f"Get distributions"):
                original_text = data["original"][idx]
                sampled_text = data["sampled"][idx]
                # original text
                tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                with torch.no_grad():
                    logits = scoring_model(**tokenized).logits[:, :-1]
                    distrib = _get_distrib(logits, args.rank_size)
                    distribs.extend(distrib)
                # sampled text
                tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                with torch.no_grad():
                    logits = scoring_model(**tokenized).logits[:, :-1]
                    distrib = _get_distrib(logits, args.rank_size)
                    distribs.extend(distrib)
    return distribs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default="./exp_test/model/mlp_distribution_model.pt")
    parser.add_argument('--dataset_path', type=str, default="./exp_test/data/")
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--rank_size', type=int, default=1000)
    parser.add_argument('--max_topk', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()
    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # prepare data
    data = get_distrib_data(args)
    random.shuffle(data)
    nvalid = len(data) // 10
    data_valid = data[:nvalid]
    data_train = data[nvalid:]
    # train model
    model = MlpDistributionModel(args.max_topk, args.rank_size)
    model.move_to(args.device)
    model.train(data_train, data_valid, args.model_file)
