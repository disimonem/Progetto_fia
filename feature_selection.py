import pandas as pd

dataset = pd.read_parquet('challenge_campus_biomedico_2024.parquet')

def drop_correlated_columns(dataset):
    """
    Drops columns from the dataset that are considered to be correlated 
    or redundant, to simplify the dataset.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset

    Returns:
    DataFrame: The DataFrame with the specified columns removed.
    """
    dataset.drop(columns=['codice_tipologia_professionista_sanitario','provincia_residenza', 'provincia_erogazione', 'asl_residenza', 'comune_residenza', 'struttura_erogazione','regione_erogazione','regione_residenza', 'asl_erogazione','codice_tipologia_struttura_erogazione'], inplace=True)
    return dataset

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
    
    dataset.drop(columns=['ora_inizio_erogazione', 'ora_fine_erogazione'], inplace=True)

    return dataset

def età(dataset):
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