"""INST414 Module 3: group wholesale customers by what they buy."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Download the data and check for basic problems.
url = 'https://archive.ics.uci.edu/static/public/292/wholesale%2Bcustomers.zip'
response = requests.get(url, timeout=60)
response.raise_for_status()
with ZipFile(BytesIO(response.content)) as archive:
    data = pd.read_csv(archive.open('Wholesale customers data.csv'))

features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
assert data.shape == (440, 8)
assert not data.isna().any().any()
assert not data.duplicated().any()
assert (data[features] >= 0).all().all()
data['row_id'] = range(1, len(data) + 1)

# 2. Reduce the influence of large purchases, then put categories on equal scales.
X = StandardScaler().fit_transform(np.log1p(data[features]))

# 3. Compare 2 through 8 clusters. A higher silhouette score is better.
scores = []
for k in range(2, 9):
    model = KMeans(n_clusters=k, random_state=42, n_init=50).fit(X)
    scores.append(silhouette_score(X, model.labels_))
best_k = scores.index(max(scores)) + 2
model = KMeans(n_clusters=best_k, random_state=42, n_init=50).fit(X)
data['cluster'] = model.labels_ + 1

# Number the groups from lowest to highest median grocery spending.
order = data.groupby('cluster')['Grocery'].median().sort_values().index
names = dict(zip(order, range(1, best_k + 1)))
data['cluster'] = data['cluster'].map(names)

# 4. Summarize each group using the original spending amounts.
medians = data.groupby('cluster')[features].median()
counts = data.groupby('cluster').size()
channels = pd.crosstab(data['cluster'], data['Channel'])
channels.columns = ['Horeca', 'Retail']
summary = medians.copy()
summary.insert(0, 'customers', counts)
summary = summary.join(channels)

# Pick the two real customers closest to each cluster center.
data['distance'] = model.transform(X).min(axis=1)
examples = data.sort_values('distance').groupby('cluster').head(2)
examples = examples.sort_values(['cluster', 'distance'])
assert counts.sum() == len(data)
assert len(examples) == 2 * best_k
assert np.array_equal(model.labels_, model.predict(X))

# 5. Save the results and a simple comparison chart.
output = Path(__file__).resolve().parent / 'results'
output.mkdir(exist_ok=True)
comparison = pd.DataFrame({'k': range(2, 9), 'silhouette': scores})
comparison.to_csv(output / 'k_selection.csv', index=False)
summary.to_csv(output / 'cluster_summary.csv')
examples.to_csv(output / 'examples.csv', index=False)
data.to_csv(output / 'assignments.csv', index=False)

medians.T.plot.bar(figsize=(10, 6), color=['#157a78', '#d46a35'])
plt.title('What each customer group buys')
plt.ylabel('Median annual spending (monetary units)')
plt.xticks(rotation=20, ha='right')
plt.legend([f'Cluster {c} (n={counts[c]})' for c in medians.index])
plt.tight_layout()
plt.savefig(output / 'cluster_profiles.png', dpi=180)
plt.close()

print('Silhouette scores:\n', comparison.round(4).to_string(index=False))
print('\nSelected number of clusters:', best_k)
print('\nCluster summary:\n', summary)
print('\nExample customers:\n', examples[['row_id', 'cluster'] + features])
print('\nAll data and output checks passed.')
