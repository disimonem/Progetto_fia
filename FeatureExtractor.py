import pandas as pd 
import numpy as np

class FeatureExtractor:
    def __init__ (self, dataset):
        self.dataset = dataset.copy()
    
    def calculate_quadrimester(self):
        self.dataset['quadrimestre'] = pd.to_datetime(self.dataset['data_erogazione'], utc = True, errors = 'coerce').dt.quarter
        self.dataset['anno'] = self.dataset['data_erogazione'].dt.year
        self.dataset.drop(columns = ['data_erogazione'], inplace = True)
        return self.dataset
    
    def calculate_age(self):
        self.dataset['data_nascita']= pd.to_datetime(self.dataset['data_nascita'], utc = True)
        self.dataset['età'] = (pd.to_datetime('today', utc = True) - self.dataset['data_nascita']).dt.days//365
        self.dataset.drop(columns = ['data_nascita'], inplace = True)
        return self.dataset
    



    
