import pandas as pd
import requests
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

class DataCleaner:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_dataset(self):
        return self.dataset

    def fetch_province_code_data(self, url):
        response = requests.get(url)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')

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

        return codice_to_comune, comune_to_codice

    def riempimento_codice_provincia(self, comune_to_codice):
        df = self.dataset.copy()
        df['provincia_residenza_upper'] = df['provincia_residenza'].str.upper()
        mask = df['codice_provincia_residenza'].isnull()
        df.loc[mask, 'codice_provincia_residenza'] = df.loc[mask, 'provincia_residenza_upper'].map(comune_to_codice)
        df.drop(columns=['provincia_residenza_upper'], inplace=True)
        self.dataset = df

    def riempimento_codice_provincia_erogazione(self, comune_to_codice):
        df = self.dataset.copy()
        df['provincia_erogazione_upper'] = df['provincia_erogazione'].str.upper()
        mask = df['codice_provincia_erogazione'].isnull()
        df.loc[mask, 'codice_provincia_erogazione'] = df.loc[mask, 'provincia_erogazione_upper'].map(comune_to_codice)
        df.drop(columns=['provincia_erogazione_upper'], inplace=True)
        self.dataset = df

    def drop_duplicate(self):
        self.dataset = self.dataset.drop_duplicates()

    def drop_visit_cancellation(self):
        self.dataset = self.dataset[pd.isna(self.dataset['data_disdetta'])]

    def fill_duration_of_visit(self):
        self.dataset['duration_of_visit'] = self.dataset['duration_of_visit'].fillna(self.dataset['duration_of_visit'].mean())

    def drop_column_id_professionista_sanitario(self):
         self.dataset.drop(columns=['id_professionista_sanitario'], inplace=True)

    def delete_column_date_disdetta(self):
        self.dataset.drop('data_disdetta', axis=1, inplace=True)

    def drop_columns_inio_e_fine_prestazione(self):
      """Rimuove le colonne 'ora_inizio_erogazione' e 'ora_fine_erogazione' dal DataFrame esistente."""
      self.dataset.drop(columns=['ora_inizio_erogazione', 'ora_fine_erogazione'], inplace=True)



    def update_dataset_with_outliers(self, contamination=0.05, n_estimators=100, max_samples='auto'):

        numeric_data = self.dataset[['età', 'duration_of_visit']]
        iso_forest = IsolationForest(contamination=contamination, n_estimators=n_estimators, max_samples=max_samples, random_state=42)
        outliers = iso_forest.fit_predict(numeric_data)

        self.dataset['anomaly_score'] = iso_forest.decision_function(numeric_data)
        self.dataset['outlier'] = outliers

        self.dataset.loc[self.dataset['età'] > 100, 'outlier'] = -1

        original_dataset = self.dataset.copy()
        cleaned_dataset = self.dataset[self.dataset['outlier'] == 1].copy()

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=original_dataset['età'], y=original_dataset['duration_of_visit'], hue=original_dataset['outlier'], palette='viridis', legend='full')
        plt.title('Scatter Plot - Dati Originali')
        plt.xlabel('Età')
        plt.ylabel('Durata Visita')

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=cleaned_dataset['età'], y=cleaned_dataset['duration_of_visit'], hue=cleaned_dataset['outlier'], palette='viridis', legend='full')
        plt.title('Scatter Plot - Dati Ripuliti')
        plt.xlabel('Età')
        plt.ylabel('Durata Visita')

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(original_dataset['outlier'], bins=3, kde=False)
        plt.title('Distribuzione Decisioni - Dati Originali')
        plt.xlabel('Classe')
        plt.ylabel('Frequenza')
        plt.xticks(ticks=[-1, 1], labels=['Outlier', 'Normale'])

        plt.subplot(1, 2, 2)
        sns.histplot(cleaned_dataset['outlier'], bins=3, kde=False)
        plt.title('Distribuzione Decisioni - Dati Ripuliti')
        plt.xlabel('Classe')
        plt.ylabel('Frequenza')
        plt.xticks(ticks=[-1, 1], labels=['Outlier', 'Normale'])

        plt.tight_layout()
        plt.show()

        self.dataset = cleaned_dataset.drop(columns=['anomaly_score', 'outlier'])

    def clean_data(self, comune_to_codice):
        self.drop_duplicate()
        self.drop_visit_cancellation()
        self.riempimento_codice_provincia(comune_to_codice)
        self.riempimento_codice_provincia_erogazione(comune_to_codice)
   
    def drop_columns(self):
        """Drops columns that are not needed for the analysis."""
        # Lista delle colonne da eliminare
        columns_to_drop = [
            'id_prenotazione', 'id_paziente', 'data_contatto',
            'codice_regione_residenza', 'codice_asl_residenza',
            'codice_provincia_residenza', 'codice_comune_residenza',
            'descrizione_attivita', 'tipologia_professionista_sanitario',
            'tipologia_struttura_erogazione', 'data_erogazione','id_professionista_sanitario', 'codice_tipologia_professionista_sanitario'
        ]

        # Verifica delle colonne esistenti nel dataset
        existing_columns = [col for col in columns_to_drop if col in self.dataset.columns]
        missing_columns = [col for col in columns_to_drop if col not in self.dataset.columns]

        print(f"Colonne esistenti che verranno eliminate: {existing_columns}")
        print(f"Colonne mancanti che non possono essere eliminate: {missing_columns}")

        # Elimina le colonne presenti nel DataFrame
        self.dataset.drop(columns=existing_columns, inplace=True, errors='ignore')