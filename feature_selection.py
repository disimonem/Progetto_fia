import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
dataset = pd.read_parquet('challenge_campus_biomedico_2024.parquet')


def drop_colomun_id_professionista_sanitario(dataset):
    """
    Drops the 'id_professionista_sanitario' column from the dataset, 
    as it is not needed for the analysis.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.

    Returns:
    DataFrame: The DataFrame with the 'id_professionista_sanitario' column removed.
    """
    dataset.drop(columns=['id_professionista_sanitario'], inplace=True)
    return dataset

def drop_visit_cancellation(dataset):
    """
    Removes rows from the dataset where 'data_disdetta' is not null,
    effectively filtering out canceled visits.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.

    Returns:
    DataFrame: The DataFrame with rows where 'data_disdetta' is null.
    """
    dataset = dataset[pd.isna(dataset['data_disdetta'])]
    return dataset

def delete_column_date_null(dataset):
    """
    Deletes the 'data_disdetta' column from the dataset after ensuring 
    it only contains null values, as it is no longer needed.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.

    Returns:
    DataFrame: The DataFrame with the 'data_disdetta' column removed.
    """
    dataset.drop(columns=['data_disdetta'], inplace=True)
    return dataset



def drop_duplicate(dataset):
    """
    Removes duplicate samples from the dataset.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.

    Returns:
    DataFrame: The DataFrame without duplicate samples.
    """
    if dataset.duplicated().any():
        # If there are duplicate samples in the dataset, remove them
        dataset = dataset.drop_duplicates()
    return dataset

def duration_of_visit(dataset):
    """
    Calculates the duration of the visit in minutes.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.

    Returns:
    DataFrame: The DataFrame with the 'duration_of_visit' column added.
    """
    # Convert the 'ora_inizio_erogazione' and 'ora_fine_erogazione' columns to datetime, specifying UTC
    dataset['ora_inizio_erogazione'] = pd.to_datetime(dataset['ora_inizio_erogazione'], utc=True)
    dataset['ora_fine_erogazione'] = pd.to_datetime(dataset['ora_fine_erogazione'], utc=True)

    # Calculate the duration of the visit in minutes
    dataset['duration_of_visit'] = (dataset['ora_fine_erogazione'] - dataset['ora_inizio_erogazione']).dt.total_seconds() / 60
    
    return dataset

def drop_columns_inio_e_fine_prestazione(dataset):
    """
    Drops columns from the dataset that are not needed for the analysis.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.

    Returns:
    DataFrame: The DataFrame with the specified columns removed.
    """
    dataset.drop(columns=['ora_inizio_erogazione', 'ora_fine_erogazione'], inplace=True)
    return dataset

def calculate_age(dataset):
    """
    Calculates the patient's age in years.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.

    Returns:
    DataFrame: The DataFrame with the column 'age' added.
    """
    # Convert the 'data_nascita' column to datetime
    dataset['data_nascita'] = pd.to_datetime(dataset['data_nascita'])

    # Calculate the age of the patient in years
    dataset['età'] = (pd.Timestamp.now() - dataset['data_nascita']).dt.days // 365

    dataset.drop(columns=['data_nascita'], inplace=True)


    return dataset

# Function to calculate Cramér's V
def cramer_v(x, y):
    """
    Calculates Cramér's V statistic for two categorical variables.

    Args:
        x: First categorical variable.
        y: Second categorical variable.

    Returns:
        Cramér's V statistic.
    """
    # Create a contingency table
    contingency = pd.crosstab(x, y)

    # Calculate chi-square test statistic and other values
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1

    # Calculate Cramér's V
    return np.sqrt(chi2 / (n * min_dim))

# Function to compute the correlation matrix using Cramér's V
def compute_correlation_matrix(df, columns):
    """
    Computes the correlation matrix for categorical variables using Cramér's V.

    Args:
        df: DataFrame containing the data.
        columns: List of columns to calculate the correlation matrix for.

    Returns:
        DataFrame containing the correlation matrix.
    """
    correlations = pd.DataFrame(index=columns, columns=columns)
    for col1 in columns:
        for col2 in columns:
            if col1 != col2:
                correlations.loc[col1, col2] = cramer_v(df[col1], df[col2])
            else:
                correlations.loc[col1, col2] = 1.0  # Perfect correlation with itself
    return correlations

# Function to plot and save the correlation matrix as a heatmap
def plot_correlation_matrix(correlations, filename):
    """
    Plots the correlation matrix using a heatmap and saves it to a file.

    Args:
        correlations: DataFrame containing the correlation matrix.
        filename: Filename for saving the heatmap image.

    Returns:
        None
    """
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlations.astype(float), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, square=True)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)

# Function to eliminate columns based on high correlation
def eliminate_highly_correlated_columns(df, columns_to_exclude):
    """
    Removes specified columns from the DataFrame based on correlation analysis.

    Args:
        df: DataFrame containing the data.
        columns_to_exclude: List of columns to be removed due to high correlation.

    Returns:
        DataFrame with the specified columns removed.
    """
    df.drop(columns=columns_to_exclude, inplace=True)
    return df

#dataset = duration_of_visit(dataset)
def dataset_preprocessing(dataset):
    categorical_columns = ['codice_tipologia_professionista_sanitario', 'provincia_residenza', 'provincia_erogazione', 'asl_residenza', 'comune_residenza', 'struttura_erogazione', 'regione_erogazione', 'regione_residenza', 'asl_erogazione', 'codice_tipologia_struttura_erogazione']

    # Compute the correlation matrix
    correlation_matrix = compute_correlation_matrix(dataset, categorical_columns)

    # Plot and save the correlation matrix
    plot_correlation_matrix(correlation_matrix, "correlation_matrix.png")

    # Remove columns that are highly correlated based on a threshold
    high_correlation_threshold = 0.9
    columns_to_exclude = [col for col in correlation_matrix.columns if any(correlation_matrix[col].astype(float) > high_correlation_threshold)]
    dataset = eliminate_highly_correlated_columns(dataset, columns_to_exclude)

    # Proceed with other preprocessing operations
    dataset = drop_colomun_id_professionista_sanitario(dataset)
    dataset = drop_visit_cancellation(dataset)
    dataset = delete_column_date_null(dataset)
    dataset = drop_duplicate(dataset)
    dataset = duration_of_visit(dataset)
    dataset = calculate_age(dataset)
    dataset = drop_columns_inio_e_fine_prestazione(dataset)
    return dataset

dataset = dataset_preprocessing(dataset)