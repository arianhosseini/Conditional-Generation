# run_evaluation.py
# 2019-12-01
# COMP550

import argparse
import random
import numpy as np
import torch
from scipy import linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from InferSent.models import InferSent
from bert_score.bert_score import score as bert_score
from bert_score.bert_score import plot_example




_MODEL_PATH = 'encoder/infersent1.pkl'
_PARAMS_MODEL = {
        'bsize': 64,
        'word_emb_dim': 300,
        'enc_lstm_dim': 2048,
        'pool_type': 'max',
        'dpout_model': 0.0,
        'version': 1
}
_W2V_PATH = 'GloVe/glove.840B.300d.txt'
_K_WORDS_VOCAB = 100000

_REAL_FILENAME = "data/bc_50k.txt"


def compute_bert_score(ref_text, gen_text, plot_hist=False, plot_similarity=False, verbose=True):
    P, R, F1 = bert_score(gen_text, ref_text, lang='en', verbose=verbose)
    if plot_hist:
        plt.hist(F1, bins=20)
        plt.savefig("bert_score_hist.png")

    if plot_similarity:
        rand_index = random.randint(0, len(gen_text)-1)
        plot_example(gen_text[rand_index], ref_text[rand_index], lang="en", fname="bert_score_similarity.png")

    print(f"System level F1 score: {F1.mean():.3f}")

def _compute_fd(dist1, dist2, eps=1.e-6):
    """
    Compute Frechet Distance

    Inspired from from https://github.com/bioinf-jku/TTUR/blob/master/fid.py
    """
    mu1 = np.atleast_1d(np.mean(dist1, axis=0))
    mu2 = np.atleast_1d(np.mean(dist2, axis=0))

    sigma1 = np.atleast_2d(np.cov(dist1, rowvar=False))
    sigma2 = np.atleast_2d(np.cov(dist2, rowvar=False))

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    sqrt_covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(sqrt_covmean).all():
        print("fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps)
        offset = np.eye(sigma1.shape[0]) * eps
        sqrt_covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(sqrt_covmean):
        if not np.allclose(np.diagonal(sqrt_covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(sqrt_covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        sqrt_covmean = sqrt_covmean.real

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(sqrt_covmean)

def _to_json_file(data, filename, verbose=True):
    if verbose:
        print(f">>> Writing results to file {filename}")

    with open(filename, "w") as f:
        json.dump(data, f)

def _read_txt_file(filename):
    with open(filename, "r") as f:
        return f.read()

def _load_samples(gen_filename, max_real_samples=10, verbose=True):
    gen = _get_gen_samples(gen_filename, verbose=verbose)
    real = _get_real_samples(max_real_samples, verbose=verbose)
    # Returns a list of raw samples and a list of generated samples
    return real, gen

def _get_cand_and_ref_samples(gen_filename, verbose=True):
    if verbose:
        print(f">>> Read generated samples from {gen_filename}")

    gen_raw_data = _read_txt_file(gen_filename)

    # Returns a list of generated samples
    return zip(*[
        # Seed and generated_text are separated by "\n\n"
        (sample.split("\n\n")[0] , sample.split("\n\n")[1])
        # Samples are separated by "\n-"
        for sample in gen_raw_data.split("\n-")
        # Skip empty sample (EOF)
        if sample != "\n"
    ])

def _get_gen_samples(gen_filename, verbose=True):
    if verbose:
        print(f">>> Read generated samples from {gen_filename}")

    gen_raw_data = _read_txt_file(gen_filename)

    # Returns a list of generated samples
    return [
        # Seed and generated_text are separated by "\n\n"
        sample.split("\n\n")[1]
        # Samples are separated by "\n-"
        for sample in gen_raw_data.split("\n-")
        # Skip empty sample (EOF)
        if sample != "\n"
    ]

def _get_real_samples(max_real_samples, verbose=True):
    if verbose:
        print(f">>> Read read samples from {_REAL_FILENAME}")

    # Get a list of generated samples
    real = _read_txt_file(_REAL_FILENAME)
    return real.split("\n")[:max_real_samples]

def _evaluate_samples(model, real_text, gen_text, verbose=True):
    if verbose:
        print(f">>> Encode the {len(real_text)} samples")

    real_embeddings = model.encode(real_text, tokenize=True)
    gen_embeddings = model.encode(gen_text, tokenize=True)
    if verbose:
        print(f">>> Compute Frechet Distance")

    return _compute_fd(real_embeddings, gen_embeddings)

def _load_pretrained_model(verbose=True):
    if verbose:
        print(f">>> Loading pretrained model from {_MODEL_PATH}")
    infersent = InferSent(_PARAMS_MODEL)
    infersent.load_state_dict(torch.load(_MODEL_PATH))
    infersent.set_w2v_path(_W2V_PATH)
    infersent.build_vocab_k_words(K=_K_WORDS_VOCAB)
    return infersent

def main_FID(input_filename, max_real_samples, verbose=True):
    model = _load_pretrained_model(verbose=verbose)

    # A sample is a pair of real text and generated text
    real_text, gen_text = _load_samples(input_filename,
                                        max_real_samples=max_real_samples,
                                        verbose=verbose)


    score = _evaluate_samples(model, real_text, gen_text, verbose=verbose)

    print(f"\n{score}")

def main_bert_score(input_filename, verbose=True, plot_hist=False, plot_similarity=False):
    ref_text, gen_text = _get_cand_and_ref_samples(input_filename, verbose=verbose)
    compute_bert_score(ref_text, gen_text, plot_hist=plot_hist, plot_similarity=plot_similarity, verbose=verbose)


if __name__ == "__main__":
    import nltk
    nltk.download('punkt')

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--evaluation", default="bert_score", choices=["bert_score", "FID"], help="What evaluation to run")
    parser.add_argument("-f", "--input-file", help="Input file with generated text")
    parser.add_argument("-m", "--max-real-samples", type=int,
                        help="Maximum number of real samples to compare to. Keep it small to debug.")
    parser.add_argument("-v", "--verbose", action="store_true")

    parser.add_argument("--plot_hist", action="store_true", help="Plot histogram")
    parser.add_argument("--plot_similarity", action="store_true", help="Plot histogram")
    args = parser.parse_args()
    if args.evaluation == "FID":
        main_FID(input_filename=args.input_file,
             max_real_samples=args.max_real_samples,
             verbose=args.verbose)
    elif args.evaluation == "bert_score":
        main_bert_score(input_filename=args.input_file,
                        plot_hist=args.plot_hist,
                        plot_similarity=args.plot_similarity)
