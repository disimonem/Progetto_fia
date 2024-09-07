import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class ClusteringAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=None, max_clusters=10):
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.kmeans = None
        self.cluster_labels = None

    def fit(self, X, y=None):
        print("Colonne nel DataFrame:", X.columns.tolist())

        # Seleziona le colonne da utilizzare per il clustering
        features = ['codice_regione_erogazione', 'età', 'generazione', 'durata_visita', 
            'anno', 'quadrimestre', 'incremento_percentuale']

        # Separazione delle variabili categoriali e numeriche
        categorical_features = ['generazione']
        numeric_features = ['età','codice_regione_erogazione', 'durata_visita', 'anno', 'quadrimestre', 'incremento_percentuale']
        

        # Preprocessing per variabili numeriche e categoriali
        preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

        # Preprocessing dei dati
        X_preprocessed = preprocessor.fit_transform(X[features])    

        # Se n_clusters non è specificato, utilizzare il metodo Elbow per determinarlo
        if self.n_clusters is None:
            self.elbow_method(X_preprocessed)
            self.n_clusters = int(input("Inserisci il numero ottimale di cluster basato sull'Elbow Method: "))
        
        # Eseguire il clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
        self.cluster_labels = self.kmeans.fit_predict(X_preprocessed)
        return self

    def transform(self, X):
        # Ritorna X con una colonna aggiuntiva per le etichette dei cluster
        return np.column_stack((X, self.kmeans.predict(X)))

    def elbow_method(self, X):
        inertias = []
        for k in range(1, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.max_clusters + 1), inertias, marker='o')
        plt.xlabel('Numero di cluster')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.show()

    def calculate_silhouette(self, X):
        sample_silhouette_values = silhouette_samples(X, self.cluster_labels)
        silhouette_avg = silhouette_score(X, self.cluster_labels)
        silhouette_per_cluster = [sample_silhouette_values[self.cluster_labels == i].mean() 
                                  for i in range(self.n_clusters)]

        print(f"Silhouette media totale: {silhouette_avg:.4f}")
        print("Silhouette media per cluster:")
        for i, sil in enumerate(silhouette_per_cluster):
            print(f"Cluster {i}: {sil:.4f}")

        plt.figure(figsize=(10, 6))
        plt.bar(range(self.n_clusters), silhouette_per_cluster)
        plt.axhline(y=silhouette_avg, color='r', linestyle='--', label='Media totale')
        plt.xlabel('Cluster')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score per Cluster')
        plt.legend()
        plt.show()

    def purity_score(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

    def calculate_purity(self, y_true):
        purity_per_cluster = []
        for i in range(self.n_clusters):
            cluster_mask = self.cluster_labels == i
            cluster_true_labels = y_true[cluster_mask]
            cluster_pred_labels = self.cluster_labels[cluster_mask]
            purity_per_cluster.append(self.purity_score(cluster_true_labels, cluster_pred_labels))

        purity_avg = np.mean(purity_per_cluster)

        print(f"Purity media totale: {purity_avg:.4f}")
        print("Purity per cluster:")
        for i, pur in enumerate(purity_per_cluster):
            print(f"Cluster {i}: {pur:.4f}")

        plt.figure(figsize=(10, 6))
        plt.bar(range(self.n_clusters), purity_per_cluster)
        plt.axhline(y=purity_avg, color='r', linestyle='--', label='Media totale')
        plt.xlabel('Cluster')
        plt.ylabel('Purity Score')
        plt.title('Purity Score per Cluster')
        plt.legend()
        plt.show()

    def visualize_clusters(self, X):
        plt.figure(figsize=(10, 8))
        plt.scatter(X[:, 0], X[:, 1], c=self.cluster_labels, cmap='viridis')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Visualizzazione dei Cluster')
        plt.colorbar(label='Cluster')
        plt.show()