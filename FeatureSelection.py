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

    def get_dataset(self):
        return self.dataset


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

      # Compute the correlation matrix
      correlation_matrix = self.compute_correlation_matrix(categorical_columns)

      # Plot and save the correlation matrix
      self.plot_correlation_matrix(correlation_matrix, "correlation_matrix.png")

      # Convert correlation matrix to numeric and replace NaNs with 0
      correlation_matrix = correlation_matrix.astype(float).fillna(0)

      # Define the threshold for high correlation
      high_correlation_threshold = 0.9

      # Initialize list to hold columns to drop
      columns_to_exclude = set()

      # Iterate over each column to find highly correlated columns
      for col in correlation_matrix.columns:
          high_correlation_cols = correlation_matrix.index[
              correlation_matrix[col] > high_correlation_threshold
          ].tolist()

          # Remove self-correlation (1.0) from the list
          high_correlation_cols = [x for x in high_correlation_cols if x != col]

          # Add columns to the exclusion set
          columns_to_exclude.update(high_correlation_cols)

      # Convert set to list
      columns_to_exclude = list(columns_to_exclude)

      print(f"Colonne da escludere per alta correlazione: {columns_to_exclude}")

      # Drop the highly correlated columns from the dataset
      self.dataset.drop(columns=columns_to_exclude, inplace=True)

      print(f"Colonne rimanenti dopo l'esclusione: {self.dataset.columns.tolist()}")