# Wholesale customer clustering

INST414 Module 3 assignment: Can spending-based groups help a wholesale account manager target product bundles beyond the existing retail/Horeca labels?

## Reproduce

Use Python 3.13. Create a virtual environment, install the requirements, then run:

```sh
python -m pip install -r requirements.txt
python analyze.py
```

The script downloads the UCI archive over HTTPS, checks its 440 records, transforms the six spending columns with `log1p` and `StandardScaler`, compares K-means for k=2 through 8, and selects the largest silhouette score. Each fit uses 50 initializations and seed 42. Channel and Region are excluded from the clustering features.

Outputs in `results/` include model comparisons, every row's cluster, original-unit medians, two centroid-nearest actual examples per cluster, validation statistics, and two figures. `row_id` means the one-based data row in the original CSV, excluding its header; it is not an official customer identifier. Display cluster numbers are ordered by median grocery spending.

## Validation and limitations

Assertions check shape, missing values, duplicates, nonnegative spending, published region/channel counts, finite transformed features, complete assignments, example uniqueness, and agreement between fitted and predicted labels. Ten additional random seeds check initialization stability. A second fit without the log transform measures preprocessing sensitivity. These are internal checks, not proof of business impact or performance on future customers. No dates, margins, order quantities, or campaign outcomes are available in this analysis.

## Sources and attribution

- [UCI Wholesale customers](https://archive.ics.uci.edu/dataset/292/wholesale+customers), DOI: https://doi.org/10.24432/C5030X
- [KMeans documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Silhouette documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

AI assistance was used to develop the analysis code and article. The code was executed against the source archive, and its generated summaries, examples, and checks were used to validate the written claims. The dataset's monetary units are not assumed to be dollars or euros.
