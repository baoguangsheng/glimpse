import random
import argparse
from tqdm import tqdm
import torch
import numpy as np
import nltk
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_builder import load_data, save_data
from model import from_pretrained

class DipperParaphraser:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.tokenizer = from_pretrained(T5Tokenizer, 'google/t5-v1_1-xxl', {}, args.cache_dir)
        self.model = from_pretrained(T5ForConditionalGeneration, args.model_name, {}, args.cache_dir)
        self.model = self.model.to(args.device)
        self.model.eval()

    def paraphrase(self, sample):
        lex_code = int(100 - self.args.lex_diversity)
        order_code = int(100 - self.args.order_diversity)
        # remove spurious newlines
        input_gen = sample
        sentences = nltk.sent_tokenize(input_gen)
        output_text = [sentences[0]]
        for sent_idx in range(1, len(sentences), args.sent_interval):
            prefix = output_text[-1]
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + args.sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code} {prefix} <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}
            with torch.inference_mode():
                outputs = self.model.generate(**final_input, do_sample=True, top_p=0.75, top_k=None, max_length=512)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text.append(outputs[0])
        output_text = " ".join(output_text).strip()
        return output_text

def generate_data(args):

    print(f'Loading model {args.model_name}...')
    dipper = DipperParaphraser(args)

    data = load_data(args.dataset_file)
    originals = data['original']
    samples = data['sampled']
    print(f"Total number of samples: {len(samples)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in samples])}")

    new_samples = []
    for sample in tqdm(samples):
        new_samples.append(dipper.paraphrase(sample))
    new_data = {'original': originals, 'sampled': new_samples}
    save_data(args.output_file, args, new_data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_test/results/xsum_gpt2-xl")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_test/data/xsum_gpt2-xl")
    parser.add_argument('--model_name', type=str, default='kalpeshk2011/dipper-paraphraser-xxl')
    parser.add_argument('--sent_interval', type=int, default=3)
    parser.add_argument('--lex_diversity', type=int, default=60)
    parser.add_argument('--order_diversity', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import nltk
    nltk.download('punkt')

    generate_data(args)
