import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns


"""   
DataCleaner class is used to clean the dataset
"""
class DataCleaner():

    '''
    Constructor
        Parameters:
            dataset_path: str
                path to the dataset
           
    '''
    def __init__(self, dataset):
        self.dataset=dataset
        self.codice_to_comune = {}
        self.comune_to_codice = {}

    '''
    This method is used to retrieve the data from the website
    '''

    def retrieve_data(self):
        url = 'https://www1.agenziaentrate.gov.it/servizi/codici/ricerca/VisualizzaTabella.php?ArcName=00T2'
        response = requests.get(url)
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
    
    def fill_duration_of_visit(self):
        self.dataset['durata_visita'] = self.dataset['durata_visita'].fillna(self.dataset['durata_visita'].mean())
        return self.dataset
   
        
    
    """
    Identifica e rimuove outliers da 'età' e 'duration_of_visit' usando l'Isolation Forest.
    Aggiunge anche le righe con età > 100 come outliers.

    Parameters:
    dataset (DataFrame): Il DataFrame contenente il dataset.
    contamination (float): La proporzione di outliers nel dataset.
    n_estimators (int): Il numero di base estimatori nell'ensemble.
    max_samples (int o float o 'auto'): Il numero di campioni da estrarre da X per allenare ciascun base estimatore.

    Returns:
    DataFrame: Il DataFrame aggiornato senza outliers e con età <= 100.
    """
    
    def update_dataset_with_outliers(self, contamination=0.05, n_estimators=100, max_samples='auto'):
    
        # Seleziona solo le colonne 'età' e 'duration_of_visit'
        numeric_data = self.dataset[['età', 'durata_visita']]
    
        # Modello Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, n_estimators=n_estimators, max_samples=max_samples, random_state=42)
    
        # Fitting del modello
        outliers = iso_forest.fit_predict(numeric_data)
    
        # Aggiungi colonne per i punteggi di anomalie e decisioni
        self.dataset['anomaly_score'] = iso_forest.decision_function(numeric_data)
        self.dataset['outlier'] = outliers

        # Aggiungi un controllo per le righe con età > 100
        self.dataset.loc[self.dataset['età'] > 100, 'outlier'] = -1  # Segna come outlier se età > 100

        # Filtra i dati normali
        dataset_cleaned = self.dataset[self.dataset['outlier'] == 1]
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=self.dataset['età'], y=self.dataset['durata_visita'], hue=self.dataset['outlier'], palette='viridis', legend='full')
        plt.title('Scatter Plot - Dati Originali')
        plt.xlabel('Età')
        plt.ylabel('Durata Visita')

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=dataset_cleaned['età'], y=dataset_cleaned['durata_visita'], hue=dataset_cleaned['outlier'], palette='viridis', legend='full')
        plt.title('Scatter Plot - Dati Puliti')
        plt.xlabel('Età')
        plt.ylabel('Durata Visita')

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(self.dataset['outlier'], bins=3, kde=False)
        plt.title('Distribuzione Classificazione - Dati Originali')
        plt.xlabel('Classe')
        plt.ylabel('Frequenza')
        plt.xticks(ticks=[-1, 1], labels=['Outlier', 'Normale'])

        plt.subplot(1, 2, 2)
        sns.histplot(dataset_cleaned['outlier'], bins=3, kde=False)
        plt.title('Distribuzione Classificazione - Dati Puliti')
        plt.xlabel('Classe')
        plt.ylabel('Frequenza')
        plt.xticks(ticks=[-1, 1], labels=['Outlier', 'Normale'])

        plt.tight_layout()
        plt.show()

        self.dataset = self.dataset.drop(columns=['anomaly_score', 'outlier'])
        return dataset_cleaned


