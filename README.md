# HotelRec: Hotel Recommendation System

A recommendation system built on the [HotelRec dataset](https://github.com/Diego999/HotelRec) (~50M interactions, ~22M users, ~365K hotels) for CMPE 256.

## Project Structure

| File | Description |
|------|-------------|
| `256groupproj_explore.ipynb` | Exploratory data analysis — rating distributions, user/item activity, sparsity, cold-start analysis, sub-rating breakdowns |
| `256groupproj_pipeline.ipynb` | Data cleaning and preprocessing — deduplication, invalid user removal, temporal train/valid/test split, ID mapping, parquet export |
| `256groupproj_popbaseline.ipynb` | Popularity-based baseline model — global popularity ranking evaluated with Hit@K and NDCG@K |
| `256groupproj_MFbaseline.ipynb` | Matrix factorization baseline — PyTorch embedding-based MF with bias terms, grid search tuning, RMSE evaluation |

**Recommended run order:** explore → pipeline → popbaseline → MFbaseline

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
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

Open the notebooks in Jupyter or VS Code and run them in the order listed above. The exploration and pipeline notebooks use DuckDB to process the data without loading it all into memory. The pipeline notebook exports train/valid/test parquet files that the two model notebooks read in.

## Requirements

- Python 3.10+
- See `requirements.txt` for packages
