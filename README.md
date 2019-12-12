# Conditional-Generation

## Prerequisites

```
Python 3.5 +
```

## Setup

To setup, you need to clone the repo with submodules. 
```
git clone --recurse-submodules git@github.com:arianhosseini/Conditional-Generation.git
```

Create and actvate a virtual environment. Then install the dependencies.
```
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Generate text data

`run_generation.py` script generates data from Book Corpus using the given 
sampling and refinement method. It outputs a txt file with the seed sentences 
and the generated text. 
```
python run_generation.py \ 
    --model_type xlnet 
    --model_name_or_path xlnet-base-cased
    --mode ltr
    --refine gibbs
```


## Evaluate the generated text
First download the embedding data and the GloVe pretrained models
```
# Embedding data
mkdir GloVe
curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip GloVe/glove.840B.300d.zip -d GloVe/
# Pretrained model
mkdir encoder
curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
```

Then run the evalution script. This script takes in a text file with seed 
sentences and generated text (output of `run_generation.py`). It computes
the Frechet Distance on InferSent embeddings of real and generated text.
The script prints the FD score.
```
python run_evaluation.py  --input-file my_input_file.txt --verbose
```


## Batch processing 

If you want to run the generation and evalutation in batch, you can use the 
bash scripts provided.
```
bash run_generation.sh
bash run_evaluation.sh
```
