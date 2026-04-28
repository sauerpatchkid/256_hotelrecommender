# HotelRec: Hotel Recommendation System

A recommendation system built on the [HotelRec dataset](https://github.com/Diego999/HotelRec) (~50M interactions, ~22M users, ~365K hotels) for CMPE 256.

## Project Structure

### Root — Exploration & Baselines

| File | Description |
|------|-------------|
| `256groupproj_explore.ipynb` | Exploratory data analysis — rating distributions, user/item activity, sparsity, cold-start analysis, sub-rating breakdowns |
| `256groupproj_pipeline.ipynb` | Data cleaning and preprocessing — deduplication, invalid user removal, temporal train/valid/test split, ID mapping, parquet export |
| `256groupproj_popbaseline.ipynb` | Popularity-based baseline model — global popularity ranking evaluated with Hit@K and NDCG@K |

### `BaselineMF/`

| File | Description |
|------|-------------|
| `256groupproj_MFbaseline.ipynb` | Matrix factorization baseline — PyTorch embedding-based MF with bias terms, grid search tuning, RMSE evaluation |

### `LightGCN/`

| File | Description |
|------|-------------|
| `lightgcn.ipynb` | LightGCN model — graph-based collaborative filtering on user-item interaction graph |
| `reranker_candidate_generation.ipynb` | Generates top-K candidate hotels per user from LightGCN for downstream reranking |
| `reranker_feature_generation.ipynb` | Engineers features for each user-candidate pair (LightGCN scores, aspect subratings, geo, popularity, etc.) |
| `reranker_model.ipynb` | LambdaMART reranker — trains and evaluates the ranking model on generated features |

### `SASRec/`

| File | Description |
|------|-------------|
| `sasrec.ipynb` | SASRec model — self-attention based sequential recommender capturing user interaction history |

### `LLM/`

| File | Description |
|------|-------------|
|  256_two_tower.ipynb | LLM-based recommendation approach — prompt-based or embedding-based hotel recommendation using a large language model |

**Recommended run order:** explore → pipeline → popbaseline → `BaselineMF/` → `LightGCN/` (lightgcn → candidate gen → feature gen → reranker)

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/sauerpatchkid/256_hotelrecommender.git
cd 256_hotelrecommender
```

### 2. Set up environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Download the dataset

The HotelRec dataset is ~50GB and is not included in this repo. Download it from the [original source](https://github.com/Diego999/HotelRec) and place the JSON file in a `data/` directory at the project root.

### 4. Run the notebooks

Open the notebooks in Jupyter or VS Code and run them in the order listed above. The exploration and pipeline notebooks use DuckDB to process the data without loading it all into memory. The pipeline notebook exports train/valid/test parquet files that the model notebooks read in.

## Requirements

- Python 3.10+
- See `requirements.txt` for packages
