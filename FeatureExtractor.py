from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Metodo fit, che non fa nulla in questo caso specifico.
        """
        # Non è necessario fare nulla in fit per questa classe
        return self

    def transform(self, X):
        """
        Applica le trasformazioni al dataset.
        
        Parameters:
        X (DataFrame): Il DataFrame da trasformare.
        
        Returns:
        DataFrame: Il DataFrame trasformato.
        """
        self.dataset = X

        # Calcola l'età e assegna la generazione
        self.calculate_age_and_assign_generation()

        # Calcola la durata della visita
        self.duration_of_visit()

        # Aggiunge il quadrimestre
        self.quadrimesters()

        # Calcola l'incremento per quadrimestre
        self.incremento_per_quadrimestre()

        # Assegna le etichette
        self.label()

        # Converti le variabili categoriali in dummy variables
        self.get_dummies()

        return self.dataset

    def calculate_age_and_assign_generation(self):
        if 'data_nascita' not in self.dataset.columns:
            raise KeyError("Column 'data_nascita' not found in dataset")

        # Converti 'data_nascita' in datetime e calcola l'età
        self.dataset['data_nascita'] = pd.to_datetime(self.dataset['data_nascita'], errors='coerce')
        self.dataset['età'] = (pd.Timestamp.now() - self.dataset['data_nascita']).dt.days // 365

        if self.dataset['età'].isna().any():
            raise ValueError("Some dates in 'data_nascita' could not be converted to datetime or are missing.")

        anno_corrente = pd.Timestamp.now().year
        self.dataset['anno_nascita'] = anno_corrente - self.dataset['età']

        condizioni = [
            (self.dataset['anno_nascita'] >= 1946) & (self.dataset['anno_nascita'] <= 1964),
            (self.dataset['anno_nascita'] > 1964) & (self.dataset['anno_nascita'] <= 1980),
            (self.dataset['anno_nascita'] > 1980) & (self.dataset['anno_nascita'] <= 2000),
            (self.dataset['anno_nascita'] > 2000)
        ]
        valori = ['Boomer', 'Generazione X', 'Generazione Y', 'Generazione Z']
        self.dataset['generazione'] = np.select(condizioni, valori, default='Unknown')

        self.dataset.drop(columns=['data_nascita', 'anno_nascita'], inplace=True)

    def duration_of_visit(self):
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError("Expected self.dataset to be a DataFrame")
        required_columns = ['ora_inizio_erogazione', 'ora_fine_erogazione']
        if not all(col in self.dataset.columns for col in required_columns):
            raise KeyError(f"Dataset must contain columns: {required_columns}")

        self.dataset['ora_inizio_erogazione'] = pd.to_datetime(self.dataset['ora_inizio_erogazione'], utc=True)
        self.dataset['ora_fine_erogazione'] = pd.to_datetime(self.dataset['ora_fine_erogazione'], utc=True)
        self.dataset['duration_of_visit'] = (self.dataset['ora_fine_erogazione'] - self.dataset['ora_inizio_erogazione']).dt.total_seconds() / 60

    def quadrimesters(self):
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError("Expected self.dataset to be a DataFrame")
        if 'data_erogazione' not in self.dataset.columns:
            raise KeyError("Column 'data_erogazione' not found in dataset")

        self.dataset['data_erogazione'] = pd.to_datetime(self.dataset['data_erogazione'], utc=True, errors='coerce')
        self.dataset['anno'] = self.dataset['data_erogazione'].dt.year
        self.dataset['quadrimestre'] = self.dataset['data_erogazione'].dt.quarter

    def incremento_per_quadrimestre(self):
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError("Expected self.dataset to be a DataFrame")
        required_columns = ['tipologia_servizio', 'id_prenotazione', 'anno', 'quadrimestre']
        if not all(col in self.dataset.columns for col in required_columns):
            raise KeyError(f"Dataset must contain columns: {required_columns}")

        conteggio_per_quadrimestre = self.dataset[self.dataset['tipologia_servizio'] == 'Teleassistenza'].groupby(
            ['generazione', 'codice_regione_erogazione', 'anno', 'quadrimestre']).size().reset_index(name='numero_teleassistenze')

        conteggio_per_quadrimestre['incremento'] = conteggio_per_quadrimestre.groupby(['generazione', 'codice_regione_erogazione', 'quadrimestre'])['numero_teleassistenze'].diff().fillna(0)

        conteggio_per_quadrimestre['incremento_percentuale'] = (conteggio_per_quadrimestre['incremento'] / conteggio_per_quadrimestre.groupby(['generazione', 'codice_regione_erogazione', 'quadrimestre'])['numero_teleassistenze'].shift(1).fillna(1)) * 100

        self.dataset = self.dataset.merge(conteggio_per_quadrimestre[['generazione', 'codice_regione_erogazione', 'anno', 'quadrimestre', 'incremento', 'incremento_percentuale']],
                                          on=['generazione', 'codice_regione_erogazione', 'anno', 'quadrimestre'],
                                          how='left')

    def label(self):
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError("Expected self.dataset to be a DataFrame")
        if 'incremento_percentuale' not in self.dataset.columns:
            raise KeyError("Column 'incremento_percentuale' not found in dataset")

        bins = [-float('inf'), 0, 5, 15, float('inf')]
        labels = ['Decremento', 'Incremento costante', 'Basso incremento', 'Alto incremento']
        self.dataset['label'] = pd.cut(self.dataset['incremento_percentuale'], bins=bins, labels=labels, right=False)
        self.dataset['label'] = self.dataset['label'].astype('object')

    def get_dummies(self, threshold=0.01):
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError("Expected self.dataset to be a DataFrame")
        categories = ['sesso', 'tipologia_servizio', 'codice_provincia_erogazione']
        if not all(col in self.dataset.columns for col in categories):
            raise KeyError(f"Dataset must contain columns: {categories}")

        for column in categories:
            counts = self.dataset[column].value_counts(normalize=True)
            to_keep = counts[counts > threshold].index
            self.dataset[column] = self.dataset[column].apply(lambda x: x if x in to_keep else 'Other')

        self.dataset = pd.get_dummies(self.dataset, columns=categories, drop_first=True)

        return self.dataset
