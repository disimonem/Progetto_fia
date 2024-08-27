import pandas as pd 
import DataCleaner
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

'''
FeatureSelection class is used to select the features to be used in the model

'''
class FeatureSelection():

    '''
    Constructor
        Parameters:
            dataset: pd.DataFrame
                dataset to be used
    '''
    def __init__ (self, dataset):
        self.dataset = dataset
    
    '''
    This method is used to calculate the duration of the visit
    Parameters:
        None
        Returns:
        dataset: pd.DataFrame
            dataset with the duration of the visit
    '''
    def duration_of_visit(self):
        self.dataset['ora_inizio_erogazione'] = pd.to_datetime(self.dataset['ora_inizio_erogazione'], utc = True)
        self.dataset['ora_fine_erogazione'] = pd.to_datetime(self.dataset['ora_fine_erogazione'], utc = True)
        self.dataset['durata_visita'] = (self.dataset['ora_fine_erogazione'] - self.dataset['ora_inizio_erogazione']).dt.total_seconds()/60
        return self.dataset
    
    '''
    This method is used to drop the columns 'ora_inizio_erogazione' and 'ora_fine_erogazione'
    Parameters:
        None
    Returns:
        dataset: pd.DataFrame
            dataset without the columns 'ora_inizio_erogazione' and 'ora_fine_erogazione'
    '''

    def drop_columns_inizio_e_fine_prestazione(self):
        self.dataset.drop(columns = ['ora_inizio_erogazione', 'ora_fine_erogazione'], inplace = True)
        return self.dataset
    '''
    This method is used to calculate the age of the patient
    Parameters:
        None
        Returns:
            dataset: pd.DataFrame
            dataset with the age of the patient
    '''
    def calculate_age(self):
        self.dataset['data_nascita']= pd.to_datetime(self.dataset['data_nascita'], utc = True)
        self.dataset['età'] = (pd.to_datetime('today', utc = True) - self.dataset['data_nascita']).dt.days//365
        self.dataset.drop(columns = ['data_nascita'], inplace = True)
        return self.dataset
    
    '''
    This method is used to calculate the number of visits per patient
    Parameters:
        None
    Returns:
        dataset: pd.DataFrame
            dataset with the number of visits per patient
    '''

    def cramer_v (x, y):
        contingency = pd.crosstab(x,y)
        chi2, _, _, _ = chi2_contingency(contingency)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape)-1
        return np.sqrt(chi2/(n*min_dim))
    
    '''
    This method is used to compute the correlation matrix
    Parameters:
        None
        Returns:
        correlations: pd.DataFrame
            correlation matrix
'''
    def compute_correlation_matrix(self):
        correlations = pd.DataFrame(index = self.dataset.columns, columns = self.dataset.columns)
        for col1 in self.dataset.columns:
            for col2 in self.dataset.columns:
                if col1 != col2:
                    correlations.loc[col1, col2] = self.cramer_v(self.dataset[col1], self.dataset[col2])
                else:
                    correlations.loc[col1, col2] = 1
        print ("\n\n\n this is the correlation matrix", correlations)
        return correlations
    
    '''
    This method is used to plot the correlation matrix
    Parameters:
        None
        Returns:
        None
    '''
    def plot_corrrelation_matrix(self):
        plt.figure(figsize=(16,12))
        sns.heatmap(self.compute_correlation_matrix(), annot = True, cmap="coolwarm", fmt = '.2f', linewidths=0.5, square=True)
        plt.xticks(rotation = 45, ha = "right", fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
    
    '''
    This method is used to eliminate the highly correlated columns
    Parameters:
        columns_to_exclude: list
            list of columns to exclude
            Returns:
            dataset: pd.DataFrame
                dataset without the highly correlated columns
    '''
    def eliminate_higly_correlated_columns(self, columns_to_exclude):
        print("\n\n\n this is the columns to exclude", columns_to_exclude)
        self.dataset.drop(columns = columns_to_exclude, inplace = True)
        return self.dataset
    
    
