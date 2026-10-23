"""Reproduce the INST414 Module 3 analysis. Run: python analyze.py"""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import zipfile

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
URL = 'https://archive.ics.uci.edu/static/public/292/wholesale%2Bcustomers.zip'
FEATURES = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
response = requests.get(URL, timeout=60)
response.raise_for_status()
with zipfile.ZipFile(BytesIO(response.content)) as archive:
    raw = archive.read('Wholesale customers data.csv')
df = pd.read_csv(BytesIO(raw))
assert df.shape == (440, 8)
assert df.isna().sum().sum() == 0
assert df.duplicated().sum() == 0
assert (df[FEATURES] >= 0).all().all()
assert df.Channel.value_counts().to_dict() == {1: 298, 2: 142}
assert df.Region.value_counts().to_dict() == {3: 316, 1: 77, 2: 47}
df.insert(0, 'row_id', np.arange(1, len(df) + 1))

# Similarity is Euclidean distance on standardized log(1 + annual spending).
# Channel, Region, and row_id are excluded from all model fitting.
X = StandardScaler().fit_transform(np.log1p(df[FEATURES]))
assert np.isfinite(X).all()
models, scores = {}, []
for k in range(2, 9):
    model = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X)
    models[k] = model
    scores.append({'k': k, 'silhouette': silhouette_score(X, model.labels_),
                   'inertia': model.inertia_,
                   'smallest_cluster': int(np.bincount(model.labels_).min())})
scores = pd.DataFrame(scores)
best_k = int(scores.loc[scores.silhouette.idxmax(), 'k'])
model = models[best_k]
# Stable display numbering: cluster 1 has the lower median grocery spending.
order = df.assign(label=model.labels_).groupby('label').Grocery.median().sort_values().index
mapping = {int(old): new + 1 for new, old in enumerate(order)}
df['cluster'] = [mapping[int(label)] for label in model.labels_]
assert len(df) == df.groupby('cluster').size().sum()
assert set(df.cluster) == set(range(1, best_k + 1))
medians = df.groupby('cluster')[FEATURES].median()
counts = df.groupby('cluster').size().rename('customers')
channels = pd.crosstab(df.cluster, df.Channel).rename(columns={1: 'Horeca', 2: 'Retail'})
summary = pd.concat([counts, medians, channels], axis=1)

# Examples are the two actual observations nearest each fitted centroid.
distances = model.transform(X)
example_indices = []
for old in order:
    indices = np.flatnonzero(model.labels_ == old)
    nearest = indices[np.argsort(distances[indices, old])[:2]]
    example_indices.extend(nearest.tolist())
examples = df.iloc[example_indices].copy()
assert len(examples) == 2 * best_k
assert examples.row_id.is_unique
assert np.array_equal(model.labels_, model.predict(X))

stability = []
for seed in range(10):
    other = KMeans(n_clusters=best_k, n_init=50, random_state=seed).fit_predict(X)
    stability.append(adjusted_rand_score(model.labels_, other))
# Sensitivity: the same k with unlogged but standardized spending.
raw_scaled = StandardScaler().fit_transform(df[FEATURES])
raw_labels = KMeans(n_clusters=best_k, n_init=50, random_state=42).fit_predict(raw_scaled)
validation = {
    'n': len(df), 'missing_cells': int(df.isna().sum().sum()),
    'duplicate_source_rows': 0, 'selected_k': best_k,
    'source_csv_sha256': hashlib.sha256(raw).hexdigest(),
    'seed_ARI_min': float(min(stability)), 'seed_ARI_max': float(max(stability)),
    'seed_ARI_values': stability,
    'negative_silhouette_count': int((silhouette_samples(X, model.labels_) < 0).sum()),
    'raw_vs_log_ARI': float(adjusted_rand_score(model.labels_, raw_labels)),
    'channel_vs_cluster_ARI': float(adjusted_rand_score(df.Channel, model.labels_)),
    'raw_scaled_cluster_sizes': np.bincount(raw_labels).tolist(),
}
scores.to_csv(OUT / 'k_selection.csv', index=False)
summary.to_csv(OUT / 'cluster_summary.csv')
examples.to_csv(OUT / 'examples.csv', index=False)
df.to_csv(OUT / 'assignments.csv', index=False)
(OUT / 'validation.json').write_text(json.dumps(validation, indent=2))

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
ax.plot(scores.k, scores.silhouette, 'o-', color='#157a78', linewidth=2)
ax.scatter([best_k], [scores.silhouette.max()], s=120, color='#d46a35', zorder=3)
ax.set(xlabel='Number of clusters (k)', ylabel='Mean silhouette score',
       title='Two clusters have the strongest separation among k = 2–8', xticks=scores.k)
ax.grid(axis='y', alpha=.2)
fig.savefig(OUT / 'k_selection.png', dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
x = np.arange(len(FEATURES)); width = .36
for j, cluster in enumerate(medians.index):
    bars = ax.bar(x + (j - .5) * width, medians.loc[cluster], width,
                 label=f'Cluster {cluster} (n={counts.loc[cluster]})',
                 color=['#157a78', '#d46a35'][j % 2])
    ax.bar_label(bars, fmt='%.0f', fontsize=9, padding=3)
ax.set(xticks=x, xticklabels=['Fresh','Milk','Grocery','Frozen','Detergents / paper','Delicatessen'],
       ylabel='Median annual spending (monetary units)',
       title='Customer groups differ most in grocery and cleaning purchases')
ax.legend(); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
ax.set_ylim(0, medians.to_numpy().max() * 1.2)
fig.savefig(OUT / 'cluster_profiles.png', dpi=180)
plt.close(fig)
print(scores.round(4).to_string(index=False))
print('\nCLUSTER SUMMARY\n', summary.to_string())
print('\nEXAMPLES\n', examples.to_string(index=False))
print('\nVALIDATION\n', json.dumps(validation, indent=2))
