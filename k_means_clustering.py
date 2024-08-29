import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import feature_selection 

dataset = pd.read_parquet('challenge_campus_biomedico_2024.parquet')
dataset = feature_selection.dataset_preprocessing(dataset)


# Creazione di una copia del dataset per evitare SettingWithCopyWarning
dataset = dataset.copy()

# Trasformazione delle colonne booleani in numeriche
bool_columns = dataset.select_dtypes(include='bool').columns
dataset[bool_columns] = dataset[bool_columns].astype(int)

# Trasformazione della colonna 'label' in numerica
dataset['label'] = dataset['label'].astype('category').cat.codes

# Separazione delle feature e della label
X = dataset.drop(['label'], axis=1)
y = dataset['label']

# Standardizzazione delle feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applicazione della PCA per la riduzione della dimensionalità a 2 componenti per la visualizzazione
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Metodo del gomito per determinare il numero ottimale di cluster
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot del metodo del gomito
plt.figure(figsize=(10, 7))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-')
plt.title('Metodo del Gomito')
plt.xlabel('Numero di Cluster')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()

# Da questo grafico, scegli il numero di cluster k
optimal_k = 4  # Supponiamo di scegliere 4 cluster (da regolare in base al grafico)

# Applicazione di K-Means con il numero di cluster ottimale
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Calcolo del punteggio di silhouette per valutare la qualità del clustering
silhouette_avg = silhouette_score(X_scaled, y_kmeans)
print(f"Punteggio medio di silhouette: {silhouette_avg:.4f}")

# Creazione di un DataFrame con le componenti principali e i cluster
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Cluster'] = y_kmeans

# Visualizzazione dei cluster
plt.figure(figsize=(10, 7))
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'], cmap='viridis', alpha=0.7)
plt.title('Cluster di K-Means')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()
