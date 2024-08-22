import pandas as pd
import data_cleaning
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load the dataset from a parquet file
dataset = pd.read_parquet('challenge_campus_biomedico_2024.parquet')

def drop_colomun_id_professionista_sanitario(dataset):
    """
    Drops the 'id_professionista_sanitario' column from the dataset.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.

    Returns:
    DataFrame: The DataFrame with the 'id_professionista_sanitario' column removed.
    """
    dataset.drop(columns=['id_professionista_sanitario'], inplace=True)
    return dataset

def drop_visit_cancellation(dataset):
    """
    Drops rows where 'data_disdetta' is not null (i.e., visit cancellations).

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.

    Returns:
    DataFrame: The DataFrame with visit cancellations removed.
    """
    dataset = dataset[pd.isna(dataset['data_disdetta'])]
    return dataset

def delete_column_date_null(dataset):
    """
    Drops the 'data_disdetta' column from the dataset.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.

    Returns:
    DataFrame: The DataFrame with the 'data_disdetta' column removed.
    """
    dataset.drop(columns=['data_disdetta'], inplace=True)
    return dataset

def drop_duplicate(dataset):
    """
    Drops duplicate rows from the dataset.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.

    Returns:
    DataFrame: The DataFrame with duplicate rows removed.
    """
    if dataset.duplicated().any():
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
    dataset['ora_inizio_erogazione'] = pd.to_datetime(dataset['ora_inizio_erogazione'], utc=True)
    dataset['ora_fine_erogazione'] = pd.to_datetime(dataset['ora_fine_erogazione'], utc=True)
    dataset['duration_of_visit'] = (dataset['ora_fine_erogazione'] - dataset['ora_inizio_erogazione']).dt.total_seconds() / 60
    return dataset

def drop_columns_inio_e_fine_prestazione(dataset):
    """
    Drops the 'ora_inizio_erogazione' and 'ora_fine_erogazione' columns from the dataset.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.

    Returns:
    DataFrame: The DataFrame with the 'ora_inizio_erogazione' and 'ora_fine_erogazione' columns removed.
    """
    dataset.drop(columns=['ora_inizio_erogazione', 'ora_fine_erogazione'], inplace=True)
    return dataset

def calculate_age(dataset):
    """
    Calculates the age of individuals in the dataset.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.

    Returns:
    DataFrame: The DataFrame with the 'età' column added and 'data_nascita' column removed.
    """
    dataset['data_nascita'] = pd.to_datetime(dataset['data_nascita'])
    dataset['età'] = (pd.Timestamp.now() - dataset['data_nascita']).dt.days // 365
    dataset.drop(columns=['data_nascita'], inplace=True)
    return dataset

def cramer_v(x, y):
    """
    Calculates Cramér's V statistic for categorical-categorical association.

    Parameters:
    x (Series): A categorical variable.
    y (Series): Another categorical variable.

    Returns:
    float: The Cramér's V statistic.
    """
    contingency = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

def compute_correlation_matrix(df, columns):
    """
    Computes the correlation matrix for categorical columns.

    Parameters:
    df (DataFrame): The DataFrame containing the dataset.
    columns (list): List of categorical columns to compute the correlation matrix for.

    Returns:
    DataFrame: The correlation matrix.
    """
    correlations = pd.DataFrame(index=columns, columns=columns)
    for col1 in columns:
        for col2 in columns:
            if col1 != col2:
                correlations.loc[col1, col2] = cramer_v(df[col1], df[col2])
            else:
                correlations.loc[col1, col2] = 1.0

    print("\n\n\n this is the correlation matrix", correlations)
    return correlations

def plot_correlation_matrix(correlations, filename):
    """
    Plots the correlation matrix and saves it as an image file.

    Parameters:
    correlations (DataFrame): The correlation matrix.
    filename (str): The filename to save the plot as.

    Returns:
    None
    """
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlations.astype(float), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, square=True)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)

def eliminate_highly_correlated_columns(df, columns_to_exclude):
    """
    Eliminates columns that are highly correlated.

    Parameters:
    df (DataFrame): The DataFrame containing the dataset.
    columns_to_exclude (list): List of columns to exclude.

    Returns:
    DataFrame: The DataFrame with highly correlated columns removed.
    """
    print("\n\n\n this is the columns to exclude", columns_to_exclude)
    df.drop(columns=columns_to_exclude, inplace=True)
    return df

def dataset_preprocessing(dataset):
    """
    Preprocesses the dataset by performing various cleaning and transformation steps.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.

    Returns:
    DataFrame: The preprocessed DataFrame.
    """
    dataset = data_cleaning.data_Cleaning(dataset, data_cleaning.comune_to_codice)  # Corrected function call
    categorical_columns = ['codice_tipologia_professionista_sanitario', 'provincia_residenza', 'provincia_erogazione', 'asl_residenza', 'comune_residenza', 'struttura_erogazione', 'regione_erogazione', 'regione_residenza', 'asl_erogazione', 'codice_tipologia_struttura_erogazione'] 
    correlation_matrix = compute_correlation_matrix(dataset, categorical_columns)
    plot_correlation_matrix(correlation_matrix, "correlation_matrix.png")

    high_correlation_threshold = 0.9
    columns_to_exclude = [col for col in correlation_matrix.columns if any(correlation_matrix[col].astype(float) > high_correlation_threshold)]
    dataset = eliminate_highly_correlated_columns(dataset, columns_to_exclude)

    dataset = drop_colomun_id_professionista_sanitario(dataset)
    dataset = drop_visit_cancellation(dataset)
    dataset = delete_column_date_null(dataset)
    dataset = drop_duplicate(dataset)
    dataset = duration_of_visit(dataset)
    dataset = calculate_age(dataset)
    dataset = drop_columns_inio_e_fine_prestazione(dataset)
    
    return dataset

def quadrimesters(dataset):
    """
    Calculates the quadrimester of the year for each visit.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.

    Returns:
    DataFrame: The DataFrame with the 'quadrimester' column added.
    """
    dataset['data_erogazione'] = pd.to_datetime(dataset['data_erogazione'], utc=True, errors='coerce')
    dataset['anno']= dataset['data_erogazione'].dt.year
    dataset['quadrimestre']= dataset['data_erogazione'].dt.quarter
    return dataset

def teleassistenze_per_quadrimestre(dataset):
    """
    Calculates the number of teleassistenze for each quadrimester and year.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.

    Returns:
    DataFrame: The DataFrame with the number of teleassistenze for each quadrimester and year.
    """
    teleassistenze = dataset[dataset['tipologia_servizio'] == 'Teleassistenza']
    if not teleassistenze.empty:
        teleassistenze_per_quadrimestre = teleassistenze.groupby(['quadrimestre', 'anno']).size().reset_index(name='teleassistenze')
    return teleassistenze_per_quadrimestre

def incremento_per_quadrimestre(dataset):

    """
    Calculates the incremento of teleassistenze for each quadrimester and year.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.

    Returns:
    DataFrame: The DataFrame with the incremento of teleassistenze for each quadrimester and year.

    """
    quadrimestri = teleassistenze_per_quadrimestre['quadrimestre'].unique()
    years = teleassistenze_per_quadrimestre['anno'].unique()

    for year in years[:-1]:
        for quadrimestre in quadrimestri:
            current_year_data = teleassistenze_per_quadrimestre[(teleassistenze_per_quadrimestre['anno']==year) & (teleassistenze_per_quadrimestre['quadrimestre']==quadrimestre)]
            next_year = year + 1
            next_year_data = teleassistenze_per_quadrimestre[(teleassistenze_per_quadrimestre['anno'] == next_year) & (teleassistenze_per_quadrimestre['quadrimestre'] == quadrimestre)]

    if not current_year_data.empty and not next_year_data.empty:
      current_value = current_year_data['teleassistenze'].values[0]
      next_value = next_year_data['teleassistenze'].values[0]

      incremento = next_value - current_value 
    
    dataset['incremento'] = incremento

    return dataset

# Preprocess the dataset    
dataset = dataset_preprocessing(dataset)

'''# Filter the dataset for the year 2019 and quadrimester 1
filtered_dataset = dataset[(dataset['anno'] == 2019) & (dataset['quadrimestre'] == 1) &(dataset['tipologia_servizio'] == 'Teleassistenza')]

# Count the number of rows in the filtered dataset
count_filtered = filtered_dataset.shape[0]

# Display the count
print(f"Number of rows for the year 2019 and quadrimester 1: {count_filtered}")'''