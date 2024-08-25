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

def compute_correlation_matrix(dataset, columns):
    """
    Computes the correlation matrix for categorical columns.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.
    columns (list): List of categorical columns to compute the correlation matrix for.

    Returns:
    DataFrame: The correlation matrix.
    """
    correlations = pd.DataFrame(index=columns, columns=columns)
    for col1 in columns:
        for col2 in columns:
            if col1 != col2:
                correlations.loc[col1, col2] = cramer_v(dataset[col1], dataset[col2])
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

def eliminate_highly_correlated_columns(dataset, columns_to_exclude):
    """
    Eliminates columns that are highly correlated.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.
    columns_to_exclude (list): List of columns to exclude.

    Returns:
    DataFrame: The DataFrame with highly correlated columns removed.
    """
    print("\n\n\n this is the columns to exclude", columns_to_exclude)
    dataset.drop(columns=columns_to_exclude, inplace=True)
    return dataset

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
    dataset = quadrimesters(dataset)
    dataset = incremento_per_quadrimestre(dataset)
    
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



def incremento_per_quadrimestre(dataset):
    """
    Calculates the incremento of teleassistenze for each quadrimester and year

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset

    Returns:
    DataFrame: The DataFrame with the incremento of teleassistenze for each quadrimester and year
    """

    # Step 1: Calcolo della somma delle teleassistenze per quadrimestre e anno
    # Raggruppiamo per anno e quadrimestre, sommando il numero di teleassistenze
    conteggio_per_quadrimestre = dataset[dataset['tipologia_servizio'] == 'Teleassistenza'].groupby(['anno', 'quadrimestre'])['id_prenotazione'].count().reset_index(name='numero_teleassistenze')
    # Step 2: Calcolo della differenza rispetto allo stesso quadrimestre dell'anno precedente
    # Raggruppiamo per quadrimestre e calcoliamo la differenza con l'anno precedente
    conteggio_per_quadrimestre['incremento'] = conteggio_per_quadrimestre.groupby('quadrimestre')['numero_teleassistenze'].diff()
    conteggio_per_quadrimestre['incremento'] = conteggio_per_quadrimestre['incremento'].fillna(0)
    conteggio_per_quadrimestre['incremento_percentuale'] = (conteggio_per_quadrimestre['incremento'] / conteggio_per_quadrimestre['numero_teleassistenze']) * 100

    # Step 3: Merge con il dataset originale per aggiungere l'incremento
    dataset = dataset.merge(conteggio_per_quadrimestre[['anno', 'quadrimestre', 'incremento', 'incremento_percentuale']], 
                        on=['anno', 'quadrimestre'], 
                        how='left')
    return dataset


# Preprocess the dataset    
dataset = dataset_preprocessing(dataset)

def label(dataset):
    """
    Calcola le etichette per l'incremento percentuale delle teleassistenze.

    Parameters:
    dataset (DataFrame): Il DataFrame contenente il dataset.

    Returns:
    DataFrame: Il DataFrame con la colonna 'label' aggiunta.
    """

    # Definire i limiti degli intervalli per il calcolo delle etichette
    bins = [-float('inf'), -60, -30, 0, 30, 60, float('inf')]

    # Definire le etichette per ciascun intervallo
    labels = ['grande decremento', 'decremento medio', 'piccolo decremento', 
            'piccolo incremento', 'incremento medio', 'grande incremento']

    # Creare la colonna 'label' usando pd.cut()
    dataset['label'] = pd.cut(dataset['incremento_percentuale'], bins=bins, labels=labels)

    # Visualizzare il DataFrame con la nuova colonna 'label'
    print(dataset)
    return dataset