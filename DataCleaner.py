import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup

"""   
DataCleaner class is used to clean the dataset
"""
class DataCleaner():

    '''
    Constructor
        Parameters:
            dataset_path: str
                path to the dataset
            url: str
                url to the website containing the data
    '''
    def __init__(self, dataset_path, url):
        self.dataset = pd.read_parquet(dataset_path)
        self.url = url
        self.codice_to_comune = {}
        self.comune_to_codice = {}

    '''
    This method is used to retrieve the data from the website
    '''

    def retrieve_data(self):
        response = requests.get(self.url)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')

        table = soup.find('table', {'class': 'table table-striped table-hover table-bordered table-header'})
        for row in table.find_all('tr')[1:]:
            cells = row.find_all('td')
            if len(cells) >= 2:
                codice = cells[0].text.strip()
                comune = cells[1].text.strip()
                self.codice_to_comune[codice] = comune
                self.comune_to_codice[comune] = codice

    '''
    This method is used to fill the missing values in the 'codice_provincia_residenza' column
    Parameters:
        None
    Returns:
        dataset: pd.DataFrame
            cleaned dataset
    '''
    def riempimento_codice_provincia(self):
        self.dataset['provincia_residenza_upper'] = self.dataset['provincia_residenza'].str.upper()
        mask = self.dataset['codice_provincia_residenza'].isnull()
        self.dataset.loc[mask, 'codice_provincia_residenza'] = self.dataset.loc[mask, 'provincia_residenza_upper'].map(self.comune_to_codice)
        self.dataset.drop(columns=['provincia_residenza_upper'], inplace=True)
        return self.dataset
    
    '''
    This method is used to fill the missing values in the 'codice_provincia_erogazione' column
        Parameters:
            None
        Returns:
            dataset: pd.DataFrame
                cleaned dataset         
    '''
    def riempimento_codice_provincia_erogazione(self):
        self.dataset['provincia_residenza_upper']= self.dataset['provincia_erogazione'].str.upper()
        mask = self.dataset['codice_provincia_erogazione'].isnull()
        self.dataset.loc[mask, 'codice_provincia_erogazione'] = self.dataset.loc[mask, 'provincia_residenza_upper'].map(self.comune_to_codice)
        self.dataset.drop(columns = ['provincia_residenza_upper'], inplace  = True)
        return self.dataset
    
    '''
    This method is used to fill the missing values in the 'codice_provincia_residenza' column
    Parameters:
        None
    Returns:
        dataset: pd.DataFrame
            cleaned dataset
    '''

    def drop_duplicate(self):
        if self.dataset.duplicated().any():
            self.dataset = self.dataset.drop_duplicates()
        return self.dataset
    
    '''
    This method is used to drop the rows with missing values in the 'data_disdetta' column
    Parameters:
        None
        Returns:
        dataset: pd.DataFrame
            cleaned dataset
            '''
    def drop_visit_cancellation(self):
        self.dataset =self.dataset[pd.isna(self.dataset['data_disdetta'])]
        return self.dataset
    

    '''
    This method is used to drop the column 'id_professionista_sanitario'
    Parameters:
        None
    Returns:
        dataset: pd.DataFrame
            cleaned dataset
    '''

    def drop_column_id_professionista_sanitario(self):
        self.dataset = self.dataset.drop(columns = ['id_professionista_sanitario'])
        return self.dataset
    
    '''
    This method is used to drop the column 'data_disdetta'
    Parameters:
        None
    Returns:
        dataset: pd.DataFrame
            cleaned dataset
    '''

    def delete_column_date_null(self):
        self.dataset = self.dataset.drop(columns = ['data_disdetta'])
        return self.dataset
    
    
    def dataset(self):
        return self.dataset
    
    def calculate_duration_of_visit(self):
        self.dataset['ora_inizio_erogazione'] = pd.to_datetime(self.dataset['ora_inizio_erogazione'], utc = True)
        self.dataset['ora_fine_erogazione'] = pd.to_datetime(self.dataset['ora_fine_erogazione'], utc = True)
        self.dataset['durata_visita'] = (self.dataset['ora_fine_erogazione'] - self.dataset['ora_inizio_erogazione']).dt.total_seconds()/60
        self.dataset.drop(columns = ['ora_inizio_erogazione', 'ora_fine_erogazione'], inplace = True)
        return self.dataset
    
    def fill_duration_of_visit(self):
        self.dataset['durata_visita'] = self.dataset['durata_visita'].fillna(self.dataset['durata_visita'].mean())
        return self.dataset
    
    def calculate_age(self):
        self.dataset['data_nascita']= pd.to_datetime(self.dataset['data_nascita'], utc = True)
        today = pd.to_datetime('today', utc = True)
        self.dataset['età'] = (today - self.dataset['data_nascita']).dt.days//365
        self.dataset.drop(columns = ['data_nascita'], inplace = True)
        return self.dataset
    



    


