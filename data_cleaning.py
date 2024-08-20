import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup
from sklearn.decomposition import PCA


# Carica il dataset
dataset = pd.read_parquet('challenge_campus_biomedico_2024.parquet')

# Recupera i dati da un URL
url = 'https://www1.agenziaentrate.gov.it/servizi/codici/ricerca/VisualizzaTabella.php?ArcName=00T2'
response = requests.get(url)
html_content = response.text
soup = BeautifulSoup(html_content, 'html.parser')

# Parsing della tabella
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

# Funzione per riempire il codice provincia
def riempimento_codice_provincia(dataset, comune_to_codice):
    dataset['provincia_residenza_upper'] = dataset['provincia_residenza'].str.upper()
    mask = dataset['codice_provincia_residenza'].isnull()
    dataset.loc[mask, 'codice_provincia_residenza'] = dataset.loc[mask, 'provincia_residenza_upper'].map(comune_to_codice)
    dataset.drop(columns=['provincia_residenza_upper'], inplace=True)
    return dataset

# Funzione per riempire il codice provincia erogazione
def riempimento_codice_provincia_erogazione(dataset, comune_to_codice):
    dataset['provincia_residenza_upper'] = dataset['provincia_erogazione'].str.upper()
    mask = dataset['codice_provincia_erogazione'].isnull()
    dataset.loc[mask, 'codice_provincia_erogazione'] = dataset.loc[mask, 'provincia_residenza_upper'].map(comune_to_codice)
    dataset.drop(columns=['provincia_residenza_upper'], inplace=True)
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