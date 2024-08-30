import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from DataCleaner import DataCleaner
from FeatureSelection import FeatureSelection
from FeatureExtractor import FeatureExtractor

def main(dataset):
    

    data_cleaner = DataCleaner(dataset)
    url = 'https://www1.agenziaentrate.gov.it/servizi/codici/ricerca/VisualizzaTabella.php?ArcName=00T2'
    codice_to_comune , comune_to_codice = data_cleaner.fetch_province_code_data(url)

    dataset = data_cleaner.clean_data(comune_to_codice)
    print("\n\n ALLA RIGA N 23....", type(dataset))
    # Step 4: Initialize FeatureExtractor and process features
    feature_extractor = FeatureExtractor(dataset)


    dataset=feature_extractor.calculate_age()

    dataset=feature_extractor.duration_of_visit()

    dataset = data_cleaner.fill_duration_of_visit()

    dataset=data_cleaner.update_dataset_with_outliers()
    print("\n\n ....", dataset['età'].max())

    dataset=data_cleaner.drop_columns_inio_e_fine_prestazione()

    dataset=data_cleaner.delete_column_date_disdetta()

    dataset=feature_extractor.quadrimesters()

    dataset=feature_extractor.incremento_per_quadrimestre()

    dataset=feature_extractor.label()


    feature_selector = FeatureSelection(feature_extractor.dataset)

    dataset=feature_selector.eliminate_highly_correlated_columns()

    dataset=feature_selector.drop_columns()
    dataset=data_cleaner.delete_column_date_disdetta()


    print("Data processing complete. Processed dataset saved to 'processed_dataset.csv'.")
    return dataset

if __name__ == "__main__":
    # Carica il dataset
    dataset = pd.read_parquet('challenge_campus_biomedico_2024.parquet')
    # Chiama la funzione main passando il dataset
    dataset = main(dataset)
    dataset.info()