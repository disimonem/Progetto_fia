import pandas as pd 
import numpy as np

class FeatureExtractor:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def filter_teleassistenza(self):
        self.dataset = self.dataset[self.dataset['tipologia_servizio'] == 'Teleassistenza']
        return self.dataset
    
    def calculate_quadrimester(self):
        self.dataset['data_erogazione'] = pd.to_datetime(self.dataset['data_erogazione'], utc=True)
        self.dataset['anno'] = self.dataset['data_erogazione'].dt.year
        self.dataset['quadrimestre'] = self.dataset['data_erogazione'].dt.month.apply(lambda x: (x-1) // 4 + 1)
        self.dataset.drop(columns=['data_erogazione'], inplace=True)
        return self.dataset
    
    def calculate_duration_of_visit(self):
        self.dataset['ora_inizio_erogazione'] = pd.to_datetime(self.dataset['ora_inizio_erogazione'], utc=True)
        self.dataset['ora_fine_erogazione'] = pd.to_datetime(self.dataset['ora_fine_erogazione'], utc=True)
        self.dataset['durata_visita'] = (self.dataset['ora_fine_erogazione'] - self.dataset['ora_inizio_erogazione']).dt.total_seconds() / 60
        self.dataset.drop(columns=['ora_inizio_erogazione', 'ora_fine_erogazione'], inplace=True)
        return self.dataset
    
    def calculate_age(self):
        self.dataset['data_nascita'] = pd.to_datetime(self.dataset['data_nascita'], utc=True)
        today = pd.to_datetime('today', utc=True)
        self.dataset['età'] = (today - self.dataset['data_nascita']).dt.days // 365
        self.dataset.drop(columns=['data_nascita'], inplace=True)
        return self.dataset
    
    def calculate_increment_by_quadrimester(self):
        self.filter_teleassistenza()
        self.calculate_quadrimester()
        
        # Ordinamento del dataset
        self.dataset = self.dataset.sort_values(by=['anno', 'quadrimestre'])
        
        # Calcolo del conteggio per ciascun gruppo
        self.dataset['conteggio'] = self.dataset.groupby(['anno', 'quadrimestre', 'età', 'codice_regione_residenza'])['id_prenotazione'].transform('size')
        
        # Calcolo dell'incremento percentuale
        self.dataset['incremento_percentuale'] = self.dataset.groupby(['quadrimestre', 'età', 'codice_regione_residenza'])['conteggio'].pct_change() * 100
        
        # Riempimento dei valori NaN con 0
        self.dataset['incremento_percentuale'] = self.dataset['incremento_percentuale'].fillna(0)
        
        # Rimozione della colonna 'conteggio'
        self.dataset.drop(columns=['conteggio'], inplace=True)
        
        return self.dataset
