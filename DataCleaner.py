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
        self.delete_column_date_disdetta()
        return self.dataset

    def fetch_province_code_data(self):
        # Questo metodo potrebbe non essere necessario nella pipeline, ma se lo è, puoi tenerlo
        response = requests.get('https://www1.agenziaentrate.gov.it/servizi/codici/ricerca/VisualizzaTabella.php?ArcName=00T2')
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
    
    def delete_column_date_disdetta(self):
        self.dataset.drop('data_disdetta', axis=1, inplace=True)

    