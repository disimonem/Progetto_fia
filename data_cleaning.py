import pandas as pd
import requests
from bs4 import BeautifulSoup

# Load the dataset
dataset = pd.read_parquet('challenge_campus_biomedico_2024.parquet')

# Fetch data from URL
url = 'https://www1.agenziaentrate.gov.it/servizi/codici/ricerca/VisualizzaTabella.php?ArcName=00T2'
response = requests.get(url)
html_content = response.text
soup = BeautifulSoup(html_content, 'html.parser')

# Parse the table
table = soup.find('table', {'class': 'table table-striped table-hover table-bordered table-header'})
codice_to_comune = {}
comune_to_codice = {}
for row in table.find_all('tr')[1:]:
    cells = row.find_all('td')
    if len(cells) >= 2:
        codice = cells[0].text.strip()
        comune = cells[1].text.strip()
        codice_to_comune[codice] = comune
        comune_to_codice[comune] = codice

# Function to fill in province code
def riempimento_codice_provincia(dataset, comune_to_codice):
    df = dataset.copy()  # Make a copy to avoid modifying the original slice
    df['provincia_residenza_upper'] = df['provincia_residenza'].str.upper()
    mask = df['codice_provincia_residenza'].isnull()
    df.loc[mask, 'codice_provincia_residenza'] = df.loc[mask, 'provincia_residenza_upper'].map(comune_to_codice)
    df.drop(columns=['provincia_residenza_upper'], inplace=True)
    return df

# Function to fill in province code for service
def riempimento_codice_provincia_erogazione(dataset, comune_to_codice):
    df = dataset.copy()  # Make a copy to avoid modifying the original slice
    df['provincia_erogazione_upper'] = df['provincia_erogazione'].str.upper()
    mask = dataset['codice_provincia_erogazione'].isnull()
    df.loc[mask, 'codice_provincia_erogazione'] = df.loc[mask, 'provincia_erogazione_upper'].map(comune_to_codice)
    df.drop(columns=['provincia_erogazione_upper'], inplace=True)
    return df

# Function to remove duplicates
def drop_duplicate(dataset):
    """
    Removes duplicate samples from the dataset.

    Parameters:
    dataset (DataFrame): The DataFrame containing the dataset.

    Returns:
    DataFrame: The DataFrame without duplicate samples.
    """
    dataset = dataset.drop_duplicates()
    return dataset

# Function to remove cancelled visits
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

# Function to fill in the duration of visits
def fill_duration_of_visit(dataset):
    dataset['duration_of_visit'] = dataset['duration_of_visit'].fillna(dataset['duration_of_visit'].mean())
    return dataset

# Main data cleaning function
def data_Cleaning(dataset, comune_to_codice):
    dataset = drop_duplicate(dataset)
    dataset = drop_visit_cancellation(dataset)
    dataset = riempimento_codice_provincia(dataset, comune_to_codice)
    dataset = riempimento_codice_provincia_erogazione(dataset, comune_to_codice)
    return dataset

# Apply the data cleaning function
dataset_cleaned = data_Cleaning(dataset, comune_to_codice)
