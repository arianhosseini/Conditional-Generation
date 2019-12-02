# run_evaluation.py
# 2019-12-01
# COMP550

import os
import pprint
import argparse
import json
import numpy as np
import torch
from scipy import linalg

from InferSent.models import InferSent

pp = pprint.PrettyPrinter(depth=6)

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

_AGGREGATORS = {
    "mean": np.mean,
    "median": np.median,
    "max": np.max,
    "min": np.min
}


def _compute_fd(dist1, dist2):
    """
    Compute Frechet Distance

    Inspired from from https://github.com/bioinf-jku/TTUR/blob/master/fid.py
    """
    mu1 = np.atleast_1d(np.mean(dist1))
    mu2 = np.atleast_1d(np.mean(dist2))

    sigma1 = np.atleast_2d(np.cov(dist1))
    sigma2 = np.atleast_2d(np.cov(dist2))

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

def _load_samples(filename, verbose=True):
    if verbose:
        print(f">>> Read samples from {filename}")

    with open(filename, "r") as f:
        raw_data = f.read()
    # Samples are separated by "\n-"
    samples = raw_data.split("\n-")
    # Returns a list of raw samples and a list of generated samples
    return list(zip(*[
        # Seed, real_text and generated_text are separated by "\n\n"
        # TODO: Add[1:] to keep only real and seed once the real text to comprae with is in the input file
        tuple(sample.split("\n\n"))
        for sample in samples
        # Skip empty sample (EOF)
        if sample != "\n"
    ]))

def _evaluate_samples(model, real_text, gen_text, verbose=True):
    if verbose:
        print(f">>> Encode the {len(real_text)} samples")

    real_embeddings = model.encode(real_text, tokenize=True)
    gen_embeddings = model.encode(gen_text, tokenize=True)

    if verbose:
        print(f">>> Compute Frechet Distance")

    return [
        _compute_fd(s, g)
        for s, g in zip(real_embeddings, gen_embeddings)
    ]

def _load_pretrained_model(verbose=True):
    if verbose:
        print(f">>> Loading pretrained model from {_MODEL_PATH}")
    infersent = InferSent(_PARAMS_MODEL)
    infersent.load_state_dict(torch.load(_MODEL_PATH))
    infersent.set_w2v_path(_W2V_PATH)
    infersent.build_vocab_k_words(K=_K_WORDS_VOCAB)
    return infersent

def main(input_filename, verbose=True):
    model = _load_pretrained_model(verbose=verbose)

    # A sample is a pair of real text and generated text
    real_text, gen_text = _load_samples(input_filename, verbose=verbose)

    _raw_results = _evaluate_samples(model, real_text, gen_text, verbose=verbose)

    # Aggregate results
    results = {agg_name: float(agg(_raw_results)) for agg_name, agg in _AGGREGATORS.items()}
    # Append raw results
    results["raw"] = _raw_results

    # Output results to json file
    output_filename = f"{os.path.splitext(input_filename)[0]}_result.json"
    _to_json_file(results, output_filename, verbose=verbose)

    if verbose:
        pp.pprint(results)


if __name__ == "__main__":
    import nltk
    nltk.download('punkt')

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", help="Input file with generated text")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    main(input_filename=args.input_file, verbose=args.verbose)
