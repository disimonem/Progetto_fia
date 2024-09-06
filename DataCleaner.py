from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Puoi aggiungere parametri di inizializzazione qui, se necessari
        pass

    def fit(self, X, y=None):
        # Non facciamo nulla nel metodo fit, ma potresti aggiungere logica se necessario
        return self

    def transform(self, X):
        self.dataset = X.copy()
        
        # Inserisci le tue operazioni di pulizia qui
        comune_to_codice = {}  # Supponiamo che tu abbia una mappa dei codici
        self.riempimento_codice_provincia(comune_to_codice)
        self.riempimento_codice_provincia_erogazione(comune_to_codice)
        self.drop_duplicate()
        self.drop_visit_cancellation()
        self.fill_duration_of_visit()
        self.delete_column_date_disdetta()
        self.drop_columns_inio_e_fine_prestazione()
        return self.dataset

    def fetch_province_code_data(self, url):
        # Questo metodo potrebbe non essere necessario nella pipeline, ma se lo è, puoi tenerlo
        response = requests.get(url)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')

        table = soup.find('table', {'class': 'table table-striped table-hover table-bordered table-header'})
        codice_to_comune = {}
        comune_to_codice = {}

        for row in table.find_all('tr')[1:]:
            cells = row.find_all('td')
            if len(cells) >= 2:
                codice = cells[0].text.strip()
                comune = cells[1].text.strip()
                codice_to_comune[codice] = comune
                comune_to_codice[comune] = codice

        return codice_to_comune, comune_to_codice

    def riempimento_codice_provincia(self, comune_to_codice):
        if 'provincia_residenza' not in self.dataset.columns or 'codice_provincia_residenza' not in self.dataset.columns:
            raise KeyError("Le colonne 'provincia_residenza' o 'codice_provincia_residenza' non sono presenti nel dataset.")

        df = self.dataset.copy()
        df['provincia_residenza_upper'] = df['provincia_residenza'].str.upper()
        mask = df['codice_provincia_residenza'].isnull()
        df.loc[mask, 'codice_provincia_residenza'] = df.loc[mask, 'provincia_residenza_upper'].map(comune_to_codice)
        df.drop(columns=['provincia_residenza_upper'], inplace=True)
        self.dataset = df

    def riempimento_codice_provincia_erogazione(self, comune_to_codice):
        if 'provincia_erogazione' not in self.dataset.columns or 'codice_provincia_erogazione' not in self.dataset.columns:
            raise KeyError("Le colonne 'provincia_erogazione' o 'codice_provincia_erogazione' non sono presenti nel dataset.")

        df = self.dataset.copy()
        df['provincia_erogazione_upper'] = df['provincia_erogazione'].str.upper()
        mask = df['codice_provincia_erogazione'].isnull()
        df.loc[mask, 'codice_provincia_erogazione'] = df.loc[mask, 'provincia_erogazione_upper'].map(comune_to_codice)
        df.drop(columns=['provincia_erogazione_upper'], inplace=True)
        self.dataset = df

    def drop_duplicate(self):
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError("Expected self.dataset to be a DataFrame.")
        self.dataset = self.dataset.drop_duplicates()

    def drop_visit_cancellation(self):
        self.dataset = self.dataset[pd.isna(self.dataset['data_disdetta'])]

    def fill_duration_of_visit(self):
        if 'duration_of_visit' not in self.dataset.columns:
            raise KeyError("Colonna 'duration_of_visit' non trovata nel dataset.")
        self.dataset['duration_of_visit'] = self.dataset['duration_of_visit'].fillna(self.dataset['duration_of_visit'].mean())

    def delete_column_date_disdetta(self):
        self.dataset.drop('data_disdetta', axis=1, inplace=True)

    def drop_columns_inio_e_fine_prestazione(self):
      """Rimuove le colonne 'ora_inizio_erogazione' e 'ora_fine_erogazione' dal DataFrame esistente."""
      self.dataset.drop(columns=['ora_inizio_erogazione', 'ora_fine_erogazione'], inplace=True)

    def update_dataset_with_outliers(self, contamination=0.05, n_estimators=100, max_samples='auto'):
        if not all(col in self.dataset.columns for col in ['età', 'duration_of_visit']):
            raise KeyError("Le colonne 'età' o 'duration_of_visit' non sono presenti nel dataset.")

        numeric_data = self.dataset[['età', 'duration_of_visit']]
        iso_forest = IsolationForest(contamination=contamination, n_estimators=n_estimators, max_samples=max_samples, random_state=42)
        outliers = iso_forest.fit_predict(numeric_data)

        self.dataset['anomaly_score'] = iso_forest.decision_function(numeric_data)
        self.dataset['outlier'] = outliers

        self.dataset.loc[self.dataset['età'] > 100, 'outlier'] = -1

        original_dataset = self.dataset.copy()
        cleaned_dataset = self.dataset[self.dataset['outlier'] == 1].copy()

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=original_dataset['età'], y=original_dataset['duration_of_visit'], hue=original_dataset['outlier'], palette='viridis', legend='full')
        plt.title('Scatter Plot - Dati Originali')
        plt.xlabel('Età')
        plt.ylabel('Durata Visita')

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=cleaned_dataset['età'], y=cleaned_dataset['duration_of_visit'], hue=cleaned_dataset['outlier'], palette='viridis', legend='full')
        plt.title('Scatter Plot - Dati Ripuliti')
        plt.xlabel('Età')
        plt.ylabel('Durata Visita')

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(original_dataset['outlier'], bins=3, kde=False)
        plt.title('Distribuzione Decisioni - Dati Originali')
        plt.xlabel('Classe')
        plt.ylabel('Frequenza')
        plt.xticks(ticks=[-1, 1], labels=['Outlier', 'Normale'])

        plt.subplot(1, 2, 2)
        sns.histplot(cleaned_dataset['outlier'], bins=3, kde=False)
        plt.title('Distribuzione Decisioni - Dati Ripuliti')
        plt.xlabel('Classe')
        plt.ylabel('Frequenza')
        plt.xticks(ticks=[-1, 1], labels=['Outlier', 'Normale'])

        plt.tight_layout()
        plt.show()

        self.dataset = cleaned_dataset.drop(columns=['anomaly_score', 'outlier'])
