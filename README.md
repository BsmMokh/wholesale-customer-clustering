# Wholesale customer clustering

A simple K-means analysis for INST414 Module 3.

**Question:** Can a wholesale account manager use purchasing patterns to choose better product bundles than business type alone?

## Run it

Use Python 3.13 and install the packages:

```sh
python -m pip install -r requirements.txt
python analyze.py
```

## What the code does

1. Downloads the 440-customer UCI dataset and checks for missing values, duplicates, and negative spending.
2. Uses the six spending columns. Channel and Region are not clustering inputs.
3. Applies `log1p` and standardization so large purchases and large-scale categories have less influence.
4. Compares K-means with 2–8 clusters using silhouette scores.
5. Prints each group's size, median spending, business types, and two customers closest to its center.
6. Saves four CSV files and a chart in `results/`.

`row_id` is the one-based data row in the source CSV, excluding its header. It is not an official customer ID. Group numbers are ordered by median grocery spending. Each model uses seed 42 and 50 initializations for reproducibility.

The analysis selects two clusters with 252 and 188 customers. The silhouette score is about 0.2903, so the groups overlap. These results suggest product bundles to test; they do not prove that a marketing offer will improve sales.

## Sources

- [UCI Wholesale customers dataset](https://archive.ics.uci.edu/dataset/292/wholesale+customers)
- [K-means documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Silhouette score documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
