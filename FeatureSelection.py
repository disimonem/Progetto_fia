import pandas as pd 
import DataCleaner
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

'''
FeatureSelection class is used to select the features to be used in the model

'''
'''
    This method is used to calculate the number of visits per patient
    Parameters:
        None
    Returns:
        dataset: pd.DataFrame
            dataset with the number of visits per patient
'''
def cramer_v (x,y):
    contingency = pd.crosstab(x,y) 
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape)-1
    return np.sqrt(chi2/(n*min_dim))

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
                    correlations.loc[col1, col2] = cramer_v(self.dataset[col1], self.dataset[col2])
                else:
                    correlations.loc[col1, col2] = 1
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
    def eliminate_highly_correlated_columns(self):
        categorical_columns = ['codice_tipologia_professionista_sanitario', 'provincia_residenza', 'provincia_erogazione', 'asl_residenza', 'comune_residenza', 'struttura_erogazione', 'regione_erogazione', 'regione_residenza', 'asl_erogazione', 'codice_tipologia_struttura_erogazione']
        correlation_matrix = self.compute_correlation_matrix(self.dataset, categorical_columns)
        self.plot_correlation_matrix(correlation_matrix, "correlation_matrix.png")
 
        high_correlation_threshold = 0.9
        columns_to_exclude = [col for col in correlation_matrix.columns if any(correlation_matrix[col].astype(float) > high_correlation_threshold)]
    #print("\n\n\n this is the columns to exclude", columns_to_exclude)
        self.dataset.drop(columns=columns_to_exclude, inplace=True)
        return self.dataset
