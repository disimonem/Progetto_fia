import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

class FeatureSelection:
    def __init__(self, dataset):
        """
        Initializes the FeatureSelection: with a dataset.

        Parameters:
        dataset (DataFrame): The DataFrame containing the dataset.
        """
        self.dataset = dataset


    def cramer_v(self, x, y):
        """Calculates Cramér's V statistic for categorical-categorical association."""
        contingency = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(contingency)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        return np.sqrt(chi2 / (n * min_dim))

    def compute_correlation_matrix(self, columns):
        """Computes the correlation matrix for categorical columns."""
        correlations = pd.DataFrame(index=columns, columns=columns)
        for col1 in columns:
            for col2 in columns:
                if col1 != col2:
                    correlations.loc[col1, col2] = self.cramer_v(self.dataset[col1], self.dataset[col2])
                else:
                    correlations.loc[col1, col2] = 1.0
        return correlations

    def plot_correlation_matrix(self, correlations, filename):
        """Plots the correlation matrix and saves it as an image file."""
        plt.figure(figsize=(16, 12))
        sns.heatmap(correlations.astype(float), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, square=True)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(filename)

    def eliminate_highly_correlated_columns(self):
        """Eliminates columns that are highly correlated."""
        categorical_columns = [
            'codice_tipologia_professionista_sanitario', 'provincia_residenza', 
            'provincia_erogazione', 'asl_residenza', 'comune_residenza', 
            'struttura_erogazione', 'regione_erogazione', 'regione_residenza', 
            'asl_erogazione', 'codice_tipologia_struttura_erogazione'
        ]
        correlation_matrix = self.compute_correlation_matrix(categorical_columns)
        self.plot_correlation_matrix(correlation_matrix, "correlation_matrix.png")

        high_correlation_threshold = 0.9
        columns_to_exclude = [col for col in correlation_matrix.columns if any(correlation_matrix[col].astype(float) > high_correlation_threshold)]
        self.dataset.drop(columns=columns_to_exclude, inplace=True)

    
    def drop_columns(self):
        """Drops columns that are not needed for the analysis."""
        columns_to_drop = ['id_prenotazione', 'id_paziente', 'data_contatto', 'codice_regione_residenza', 
                           'codice_asl_residenza', 'codice_provincia_residenza', 'codice_comune_residenza', 
                           'descrizione_attivita', 'tipologia_professionista_sanitario', 
                           'tipologia_struttura_erogazione', 'data_erogazione']
        self.dataset.drop(columns=columns_to_drop, inplace=True)

'''def preprocess(self):
        """Executes the full preprocessing pipeline."""
        self.drop_column_id_professionista_sanitario()
        self.drop_visit_cancellation()
        self.delete_column_date_null()
        self.drop_duplicate()
        self.duration_of_visit()
        self.calculate_age()
        self.drop_columns_inio_e_fine_prestazione()
        self.quadrimesters()
        self.incremento_per_quadrimestre()
        self.label()
        self.eliminate_highly_correlated_columns()
        self.drop_columns()
        self.get_dummies()
        self.fill_duration_of_visit()
        return self.dataset

# Load the dataset from a parquet file
dataset = pd.read_parquet('challenge_campus_biomedico_2024.parquet')

# Initialize the FeatureSelection: with the dataset
data_preprocessor = FeatureSelection:(dataset)

# Preprocess the dataset
cleaned_dataset = data_preprocessor.preprocess()'''
    
