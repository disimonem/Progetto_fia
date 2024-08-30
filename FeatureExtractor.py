import pandas as pd 
import numpy as np

class FeatureExtractor:
    def __init__ (self, dataset):
        self.dataset = dataset.copy()
    
    def filter_teleassistenza(self):
        self.dataset = self.dataset[self.dataset['tipologia_servizio'] == 'Teleassistenza']
        return self.dataset
    

    def calculate_quadrimester(self):
        self.dataset['data_erogazione'] = pd.to_datetime(self.dataset['data_erogazione'], utc = True)
        self.dataset['anno'] = self.dataset['data_erogazione'].dt.year
        self.dataset['quadrimestre'] = self.dataset['data_erogazione'].dt.month.apply(lambda x: (x-1) // 4 + 1)
        self.dataset.drop(columns = ['data_erogazione'], inplace = True)
        return self.dataset
    
    

    def calculate_increment_by_quadrimester(self):
        """Calcola l'incremento percentuale per ogni quadrimestre, età e regione."""
        self.filter_teleassistenza()
        self.calculate_quadrimester()
        
        # Ordinamento del dataset
        self.dataset = self.dataset.sort_values(by=['anno', 'quadrimestre'])
        
        # Calcolo del conteggio per ciascun gruppo
        self.dataset['conteggio'] = self.dataset.groupby(['anno', 'quadrimestre', 'età', 'codice_regione_residenza'])['id_prenotazione'].transform('size')
        
        # Visualizza una porzione dei dati per il debug
        print("Dati aggregati e ordinati:")
        print(self.dataset[['anno', 'quadrimestre', 'età', 'codice_regione_residenza', 'conteggio']].head(10))
        
        # Calcolo dell'incremento percentuale
        self.dataset['incremento_percentuale'] = self.dataset.groupby(['quadrimestre', 'età', 'codice_regione_residenza'])['conteggio'].pct_change() * 100
        
        # Verifica se ci sono NaN nell'incremento percentuale
        print("NaN in incremento_percentuale:")
        print(self.dataset['incremento_percentuale'].isna().sum())
        
        # Riempimento dei valori NaN con 0
        self.dataset['incremento_percentuale'] = self.dataset['incremento_percentuale'].fillna(0)
        
        # Rimozione della colonna 'conteggio'
        self.dataset.drop(columns=['conteggio'], inplace=True)
        
        return self.dataset
    
    def extract_features(self):
        self.calculate_increment_by_quadrimester()
        return self.dataset
    



    
