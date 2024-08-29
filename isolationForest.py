from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import feature_selection


# Carica il dataset
dataset = pd.read_parquet('challenge_campus_biomedico_2024.parquet')

# Preprocessing del dataset
dataset = feature_selection.dataset_preprocessing(dataset)

def update_dataset_with_outliers(dataset, contamination=0.05, n_estimators=100, max_samples='auto'):
    """
    Identifica e rimuove outliers da 'età' e 'duration_of_visit' usando l'Isolation Forest.
    Aggiunge anche le righe con età > 100 come outliers.

    Parameters:
    dataset (DataFrame): Il DataFrame contenente il dataset.
    contamination (float): La proporzione di outliers nel dataset.
    n_estimators (int): Il numero di base estimatori nell'ensemble.
    max_samples (int o float o 'auto'): Il numero di campioni da estrarre da X per allenare ciascun base estimatore.

    Returns:
    DataFrame: Il DataFrame aggiornato senza outliers e con età <= 100.
    """
    # Seleziona solo le colonne 'età' e 'duration_of_visit'
    numeric_data = dataset[['età', 'duration_of_visit']]
    
    # Modello Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, n_estimators=n_estimators, max_samples=max_samples, random_state=42)
    
    # Fitting del modello
    outliers = iso_forest.fit_predict(numeric_data)
    
    # Aggiungi colonne per i punteggi di anomalie e decisioni
    dataset['anomaly_score'] = iso_forest.decision_function(numeric_data)
    dataset['outlier'] = outliers

    # Aggiungi un controllo per le righe con età > 100
    dataset.loc[dataset['età'] > 100, 'outlier'] = -1  # Segna come outlier se età > 100

    # Filtra i dati normali
    dataset_cleaned = dataset[dataset['outlier'] == 1]

    return dataset_cleaned

# Applicazione del modello al dataset
dataset_cleaned = update_dataset_with_outliers(dataset)


# Grafici di confronto

# Scatter Plot: Originale vs Pulito
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=dataset['età'], y=dataset['duration_of_visit'], hue=dataset['outlier'], palette='viridis', legend='full')
plt.title('Scatter Plot - Dati Originali')
plt.xlabel('Età')
plt.ylabel('Durata Visita')

plt.subplot(1, 2, 2)
sns.scatterplot(x=dataset_cleaned['età'], y=dataset_cleaned['duration_of_visit'], hue=dataset_cleaned['outlier'], palette='viridis', legend='full')
plt.title('Scatter Plot - Dati Puliti')
plt.xlabel('Età')
plt.ylabel('Durata Visita')

plt.tight_layout()
plt.show()

# Distribuzione delle Decisioni di Classificazione: Originale vs Pulito
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(dataset['outlier'], bins=3, kde=False)
plt.title('Distribuzione Classificazione - Dati Originali')
plt.xlabel('Classe')
plt.ylabel('Frequenza')
plt.xticks(ticks=[-1, 1], labels=['Outlier', 'Normale'])

plt.subplot(1, 2, 2)
sns.histplot(dataset_cleaned['outlier'], bins=3, kde=False)
plt.title('Distribuzione Classificazione - Dati Puliti')
plt.xlabel('Classe')
plt.ylabel('Frequenza')
plt.xticks(ticks=[-1, 1], labels=['Outlier', 'Normale'])

plt.tight_layout()
plt.show()