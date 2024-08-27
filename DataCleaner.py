import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup

class DataCleaner():
    def __init__(self, dataset_path, url):
        self.dataset = pd.read_parquet('challenge_campus_biomedico_2024.parquet')
        self.url = url
        self.codice_to_comune = {}
        self.comune_to_codice = {}

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


    def riempimento_codice_provincia(self):
        self.dataset['provincia_residenza_upper'] = self.dataset['provincia_residenza'].str.upper()
        mask = self.dataset['provincia_residenza_upper'].isnull()
        self.dataset.loc[mask, 'provincia_residenza_upper'] = self.dataset.loc[mask, 'provincia_residenza'].map(self.comune_to_codice)
        self.dataset.drop(columns=['provincia_residenza_upper'], inplace=True)
        return self.dataset
    
    def riempimento_codice_provincia_erogazione(self):
        self.dataset['provincia_residenza_upper']= self.dataset['provicia_erogazione'].str.upper()
        mask = self.dataset['codice_provincia_erogazione'].isnull()
        self.dataset.loc[mask, 'codice_provincia_erogazione'] = self.dataset.loc[mask, 'provincia_residenza_upper'].map(self.comune_to_codice)
        self.dataset.drop(columns = ['provincia_residenza_upper'], inplace  = True)
        return self.dataset
    
    def drop_duplicate(self):
        if self.dataset.duplicated().any():
            self.dataset = self.dataset.drop_duplicates()
        return self.dataset
    
    def drop_visit_cancellation(self):
        self.dataset =self.dataset[pd.isna(self.dataset['data_disdetta'])]
        return self.dataset
    
    def drop_column_id_professionista_sanitario(self):
        self.dataset = self.dataset.drop(columns = ['id_professionista_sanitario'])
        return self.dataset
    
    



    

