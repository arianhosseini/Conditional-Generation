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

```
python run_generation.py  # use --help to see the options
```


## Evaluate the generated text
Download the embedding data 
```
mkdir GloVe
curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip GloVe/glove.840B.300d.zip -d GloVe/
mkdir fastText
curl -Lo fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip fastText/crawl-300d-2M.vec.zip -d fastText/
```

```
python run_evaluation.py  # use --help to see the options
```


