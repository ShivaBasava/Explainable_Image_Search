# Explainable Image Search

An art image search tool built on SigLIP2 embeddings and FAISS, with a set of explainability views layered on top of results 

The repo has two apps that share the same index/metadata:

- `build_index.py` - pulls artworks from Wikidata and builds the local FAISS index + metadata file. Run this first if you don't already have `demo_data/` populated.
- `proto_type_main.py` - the actual search app.

## Setup

Clone the repo:

```
git clone https://github.com/ShivaBasava/Explainable_Image_Search.git
cd Explainable_Image_Search
```

Create and activate a Python 3.11 environment:

```
conda create -n eis python=3.11
conda activate eis
```

Install dependencies:

```
pip install -r requirements.txt
```

## To Run

```
streamlit run proto_type_main.py --logger.level=debug
```

If the required index files are missing or an error occurs, run the index builder app first. It will create the index and meta.json files.

```
streamlit run build_index.py --logger.level=debug
```

