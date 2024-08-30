import pandas as pd
import numpy as np

class FeatureExtractor:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def calculate_age(self):
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError("Expected self.dataset to be a DataFrame")
        if 'data_nascita' not in self.dataset.columns:
            raise KeyError("Column 'data_nascita' not found in dataset")

        self.dataset['data_nascita'] = pd.to_datetime(self.dataset['data_nascita'])
        self.dataset['età'] = (pd.Timestamp.now() - self.dataset['data_nascita']).dt.days // 365
        self.dataset.drop(columns=['data_nascita'], inplace=True)
        return self.dataset
    
    def duration_of_visit(self):
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError("Expected self.dataset to be a DataFrame")
        required_columns = ['ora_inizio_erogazione', 'ora_fine_erogazione']
        if not all(col in self.dataset.columns for col in required_columns):
            raise KeyError(f"Dataset must contain columns: {required_columns}")

        self.dataset['ora_inizio_erogazione'] = pd.to_datetime(self.dataset['ora_inizio_erogazione'], utc=True)
        self.dataset['ora_fine_erogazione'] = pd.to_datetime(self.dataset['ora_fine_erogazione'], utc=True)
        self.dataset['duration_of_visit'] = (self.dataset['ora_fine_erogazione'] - self.dataset['ora_inizio_erogazione']).dt.total_seconds() / 60
        return self.dataset
    
    def quadrimesters(self):
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError("Expected self.dataset to be a DataFrame")
        if 'data_erogazione' not in self.dataset.columns:
            raise KeyError("Column 'data_erogazione' not found in dataset")

        self.dataset['data_erogazione'] = pd.to_datetime(self.dataset['data_erogazione'], utc=True, errors='coerce')
        self.dataset['anno'] = self.dataset['data_erogazione'].dt.year
        self.dataset['quadrimestre'] = self.dataset['data_erogazione'].dt.quarter
        return self.dataset

    def incremento_per_quadrimestre(self):
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError("Expected self.dataset to be a DataFrame")
        required_columns = ['tipologia_servizio', 'id_prenotazione', 'anno', 'quadrimestre']
        if not all(col in self.dataset.columns for col in required_columns):
            raise KeyError(f"Dataset must contain columns: {required_columns}")

        conteggio_per_quadrimestre = self.dataset[self.dataset['tipologia_servizio'] == 'Teleassistenza'].groupby(
            ['anno', 'quadrimestre']
        )['id_prenotazione'].count().reset_index(name='numero_teleassistenze')

        conteggio_per_quadrimestre['incremento'] = conteggio_per_quadrimestre.groupby('quadrimestre')['numero_teleassistenze'].diff()
        conteggio_per_quadrimestre['incremento'] = conteggio_per_quadrimestre['incremento'].fillna(0)
        conteggio_per_quadrimestre['incremento_percentuale'] = (conteggio_per_quadrimestre['incremento'] / conteggio_per_quadrimestre['numero_teleassistenze']) * 100

        self.dataset = self.dataset.merge(conteggio_per_quadrimestre[['anno', 'quadrimestre', 'incremento', 'incremento_percentuale']], 
                                          on=['anno', 'quadrimestre'], 
                                          how='left')
        self.dataset.drop(columns='incremento', inplace=True)
        return self.dataset

    def label(self):
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError("Expected self.dataset to be a DataFrame")
        if 'incremento_percentuale' not in self.dataset.columns:
            raise KeyError("Column 'incremento_percentuale' not found in dataset")

        bins = [-float('inf'), -60, -30, 0, 30, 60, float('inf')]
        labels = ['grande decremento', 'decremento medio', 'piccolo decremento', 
                  'piccolo incremento', 'incremento medio', 'grande incremento']

        self.dataset['label'] = pd.cut(self.dataset['incremento_percentuale'], bins=bins, labels=labels)
        self.dataset['label'] = self.dataset['label'].astype('object')
        self.dataset.loc[self.dataset['incremento_percentuale'] == 0, 'label'] = 'nessun incremento'
        return self.dataset

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
