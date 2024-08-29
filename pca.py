import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import feature_selection 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Caricamento del dataset
dataset = pd.read_parquet('challenge_campus_biomedico_2024.parquet')
dataset = feature_selection.dataset_preprocessing(dataset)




# Trasformazione delle colonne booleani in numeriche
bool_columns = dataset.select_dtypes(include='bool').columns
dataset[bool_columns] = dataset[bool_columns].astype(int)

# Trasformazione della colonna 'label' in numerica se non lo è già
dataset['label'] = dataset['label'].astype('category').cat.codes

# Separazione delle feature e della label
X = dataset.drop(['label'], axis=1)
y = dataset['label']

# Standardizzazione delle feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applicazione della PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Creare un DataFrame con le componenti principali
pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
pca_df['label'] = y.values

# Percentuale di varianza spiegata da ciascuna componente principale
explained_variance_ratio = pca.explained_variance_ratio_
print("Percentuale di varianza spiegata da ciascuna componente principale:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f'PC{i+1}: {ratio:.4f}')

# Plot delle prime due componenti principali
plt.figure(figsize=(10, 7))
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['label'], cmap=plt.get_cmap('tab10', len(np.unique(y))))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA delle Prime Due Componenti Principali')
plt.colorbar(scatter, label='Label')
plt.show()
