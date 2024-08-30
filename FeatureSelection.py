import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

class FeatureSelection:

    def __init__(self, dataset):
        self.dataset = dataset

    def compute_correlation_matrix(self, columns):
        correlations = pd.DataFrame(index=columns, columns=columns)
        for col1 in columns:
            for col2 in columns:
                if col1 != col2:
                    correlations.loc[col1, col2] = self.cramer_v(self.dataset[col1], self.dataset[col2])
                else:
                    correlations.loc[col1, col2] = 1
        return correlations

    def plot_correlation_matrix(self):
        categorical_columns = ['codice_tipologia_professionista_sanitario', 'provincia_residenza',
                               'provincia_erogazione', 'asl_residenza', 'comune_residenza',
                               'struttura_erogazione', 'regione_erogazione', 'regione_residenza',
                               'asl_erogazione', 'codice_tipologia_struttura_erogazione']
        correlation_matrix = self.compute_correlation_matrix(categorical_columns)
        sns.heatmap(correlation_matrix.astype(float), annot=True)
        plt.show()

    def cramer_v(self, x, y):
        contingency = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(contingency)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        return np.sqrt(chi2 / (n * min_dim))

    def eliminate_highly_correlated_columns(self, columns_to_exclude):
        self.dataset.drop(columns=columns_to_exclude, inplace=True)
        return self.dataset

    def remove_columns_with_unique_correlation(self, columns_pairs):
        '''
        This method removes columns with unique correlation
        
        Args:
            columns_pairs: List of tuples containing the code-description column pairs to be compared.

        Returns:
            The DataFrame with removed columns
        '''
        pairs_removed = []

        for code, description in columns_pairs:
            if code in self.dataset.columns and description in self.dataset.columns:
                code_groups = self.dataset.groupby(code)[description].nunique()
                description_groups = self.dataset.groupby(description)[code].nunique()

                # Print details of corrections if needed
                self.print_details_corrections(code, description, code_groups, description_groups)

                unique_correlation_code_description = all(code_groups <= 1)
                unique_correlation_description_code = all(description_groups <= 1)

                if unique_correlation_code_description and unique_correlation_description_code:
                    self.dataset.drop(columns=[code], inplace=True)
                    print(f'Unique correlation between {code} and {description}. Column {code} removed.')
                    pairs_removed.append((code, description))
            else:
                print(f'Columns {code} or {description} not found in the dataframe.')
                pairs_removed.append((code, description))

        # Update the list of columns pairs removing the ones that have been removed
        columns_pairs_updated = [pair for pair in columns_pairs if pair not in pairs_removed]
        return self.dataset, columns_pairs_updated

    def print_details_corrections(self, code, description, code_groups, description_groups):
        # Define this method if you need to print details for debugging
        print(f'Code column {code}:')
        print(code_groups)
        print(f'Description column {description}:')
        print(description_groups)
