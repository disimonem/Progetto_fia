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
        self.dataset['quadrimestre'] = self.dataset['data_erogazione'].dt.quarter
        self.dataset['anno'] = self.dataset['data_erogazione'].dt.year
        self.dataset.drop(columns = ['data_erogazione'], inplace = True)
        return self.dataset
    
    
    
    
    
    def calculate_increment_by_quadrimester(self):
        self.filter_teleassistenza()
        self.dataset['count'] = self.dataset.groupby(['regione_erogazione', 'quadrimestre', 'età', 'anno'])['tipologia_servizio'].transform('size')
        self.dataset['incremento_percentuale'] = self.dataset.groupby(['regione_erogazione', 'quadrimestre', 'età'])['count'].transform(lambda x: x.diff()/x.shift(1)*100).fillna(0)
        print(self.dataset[['regione_erogazione', 'quadrimestre', 'età', 'anno', 'count', 'incremento_percentuale']].head(200000))
        return self.dataset
    
    def extract_features(self):
        self.calculate_quadrimester()
        self.calculate_increment_by_quadrimester()
        return self.dataset
    



    
