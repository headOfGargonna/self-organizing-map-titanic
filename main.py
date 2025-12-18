import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
titanic = pd.read_csv(url)

df = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']].copy()
df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df.dropna(inplace=True)

y = df['survived'].values
X_raw = df.drop(columns=['survived']).values

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

class SOM:
    def __init__(self, m, n, dim, learning_rate=0.5, sigma=None, decay=0.99):
        self.m = m
        self.n = n
        self.dim = dim
        self.learning_rate = learning_rate
        self.sigma = sigma if sigma else max(m, n) / 2.0
        self.decay = decay
        self.weights = np.random.rand(m, n, dim)

    def _find_bmu(self, x):
        distances = np.linalg.norm(self.weights - x, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx

    def _get_neighborhood(self, bmu_idx, sigma):
        m, n = self.m, self.n
        i, j = bmu_idx
        ii, jj = np.meshgrid(range(m), range(n), indexing='ij')
        dist_sq = (ii - i) ** 2 + (jj - j) ** 2
        neighborhood = np.exp(-dist_sq / (2 * sigma ** 2))
        return neighborhood[..., np.newaxis]

    def train(self, X, epochs=1000):
        for epoch in range(epochs):
            lr = self.learning_rate * (self.decay ** epoch)
            sigma = self.sigma * (self.decay ** epoch)
            print(epoch, self.weights)
            for x in X:
                bmu_idx = self._find_bmu(x)
                neighborhood = self._get_neighborhood(bmu_idx, sigma)
                self.weights += lr * neighborhood * (x - self.weights)

    def predict(self, X):
        bmu_coords = []
        for x in X:
            bmu_idx = self._find_bmu(x)
            bmu_coords.append(bmu_idx)
        return np.array(bmu_coords)

som = SOM(m=10, n=10, dim=X.shape[1], learning_rate=0.5, decay=0.995)
som.train(X, epochs=2000)
bmu_coords = som.predict(X)

def compute_u_matrix(weights):
    m, n, dim = weights.shape
    u_matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            dists = []
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    dists.append(np.linalg.norm(weights[i,j] - weights[ni,nj]))
            u_matrix[i, j] = np.mean(dists) if dists else 0
    return u_matrix

u_matrix = compute_u_matrix(som.weights)

survival_sum = np.full((som.m, som.n), 0.0)
count_map = np.zeros((som.m, som.n))
for (i, j), survived in zip(bmu_coords, y):
    survival_sum[i, j] += survived
    count_map[i, j] += 1
survival_rate = np.divide(survival_sum, count_map, out=np.full_like(survival_sum, np.nan), where=count_map != 0)

X_original = df.drop(columns=['survived']).values
fare_sum = np.full((som.m, som.n), 0.0)
fare_count = np.zeros((som.m, som.n))
for (i, j), fare in zip(bmu_coords, X_original[:, -1]):
    fare_sum[i, j] += fare
    fare_count[i, j] += 1
avg_fare_map = np.divide(fare_sum, fare_count, out=np.full_like(fare_sum, np.nan), where=fare_count != 0)

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.heatmap(u_matrix, cmap='viridis', square=True, cbar=True)
plt.title('U-Matrix: границы кластеров')

plt.subplot(1, 3, 2)
sns.heatmap(survival_rate, cmap='RdYlGn', square=True, cbar=True, vmin=0, vmax=1,
            cbar_kws={'label': 'Доля выживших'})
plt.title('Доля выживших')

plt.subplot(1, 3, 3)
sns.heatmap(avg_fare_map, cmap='Blues', square=True, cbar=True,
            cbar_kws={'label': 'Средний тариф ($)'})
plt.title('Средний тариф')

plt.tight_layout()
plt.show()