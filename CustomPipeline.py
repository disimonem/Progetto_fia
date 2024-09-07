import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from DataCleaner import DataCleaner
from FeatureExtractor import FeatureExtractor
from FeatureSelection import FeatureSelection
from ClusteringAnalyzer import ClusteringAnalyzer

class CustomPipeline:
    def __init__(self):
        """
        Inizializza la pipeline con tutte le fasi necessarie.
        """
        self.pipeline = Pipeline(steps=[
            ('cleaner', DataCleaner()),               # Pulizia dei dati
            ('feature_extractor', FeatureExtractor()),  # Estrazione delle feature              
            ('feature_selection', FeatureSelection()), # Selezione delle feature
            #('scaler', StandardScaler()),              # Normalizzazione delle feature
            ('clustering_analyzer', ClusteringAnalyzer(max_clusters=10))  # Analisi del clustering
        ])

    def fit(self, X, y=None):
        """
        Fit della pipeline sui dati.
        
        Parameters:
        X (DataFrame): Il dataset su cui eseguire la pipeline.
        y (Series, optional): Le etichette, se necessarie.
        """
        print("Avvio del fit della pipeline...")
        self.pipeline.fit(X)
        print("Fit completato.")

    def transform(self, X):
        """
        Trasforma i dati applicando le trasformazioni definite nella pipeline.
        
        Parameters:
        X (DataFrame): Il dataset da trasformare.
        
        Returns:
        DataFrame: Il dataset trasformato.
        """
        print("Trasformazione dei dati in corso...")
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        """
        Esegue fit e trasformazione dei dati in un'unica operazione.
        
        Parameters:
        X (DataFrame): Il dataset su cui eseguire la pipeline.
        y (Series, optional): Le etichette, se necessarie.
        
        Returns:
        DataFrame: Il dataset trasformato.
        """
        print("Fit e trasformazione dei dati in corso...")
        return self.pipeline.fit_transform(X)

    def analyze_clusters(self, X):
        """
        Esegue l'analisi del clustering sui dati (Silhouette Score, visualizzazione dei cluster).
        
        Parameters:
        X (DataFrame): Il dataset su cui calcolare i cluster.
        """
        clustering_analyzer = self.pipeline['clustering_analyzer']

        # Calcola il silhouette score
        print("Calcolo del Silhouette Score...")
        clustering_analyzer.calculate_silhouette(X.values)

        # Visualizza i cluster
        print("Visualizzazione dei cluster...")
        clustering_analyzer.visualize_clusters(X.values)

    def analyze_purity(self, X, y_true):
        """
        Calcola il Purity Score se sono presenti etichette reali.
        
        Parameters:
        X (DataFrame): Il dataset.
        y_true (Series): Le etichette reali.
        """
        clustering_analyzer = self.pipeline['clustering_analyzer']

        # Calcola il Purity Score
        print("Calcolo del Purity Score...")
        clustering_analyzer.calculate_purity(y_true)

    def execute_pipeline(self, df, y_true_column=None):
        """
        Esegue l'intero processo della pipeline.
        
        Parameters:
        df (DataFrame): Il dataset completo.
        y_true_column (str, optional): Nome della colonna che contiene le etichette reali (per il Purity Score).
        """
        # 1. Prepara il dataset eliminando colonne non necessarie
        X_transformed = self.fit_transform(df)
        
        # 3. Analizza i cluster
        self.analyze_clusters(X_transformed)

        # 4. Se sono presenti le etichette reali, analizza il Purity Score
        if y_true_column and y_true_column in df.columns:
            y_true = df[y_true_column]
            self.analyze_purity(X_transformed, y_true)

