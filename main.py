import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from DataCleaner import DataCleaner
from FeatureSelection import FeatureSelection
from FeatureExtractor import FeatureExtractor

def main():
    dataset = pd.read_parquet('challenge_campus_biomedico_2024.parquet')

    data_cleaner = DataCleaner(dataset)
    url = 'https://www1.agenziaentrate.gov.it/servizi/codici/ricerca/VisualizzaTabella.php?ArcName=00T2'
    codice_to_comune , comune_to_codice = data_cleaner.fetch_province_code_data(url)
    data_cleaner.clean_data(comune_to_codice)

    feature_extractor = FeatureExtractor(dataset)

    feature_extractor.calculate_age_and_assign_generation()
    feature_extractor.duration_of_visit()
    dataset= feature_extractor.get_dataset()
    data_cleaner = DataCleaner(dataset)
    data_cleaner.fill_duration_of_visit()
    data_cleaner.update_dataset_with_outliers()
    data_cleaner.drop_columns_inio_e_fine_prestazione()
    data_cleaner.delete_column_date_disdetta()
    dataset= data_cleaner.get_dataset()
    feature_extractor = FeatureExtractor(dataset)
    feature_extractor.quadrimesters()
    feature_extractor.incremento_per_quadrimestre()
    feature_extractor.label()

    dataset = feature_extractor.get_dataset()
    feature_selector = FeatureSelection(dataset)

    feature_selector.eliminate_highly_correlated_columns()
    dataset = feature_selector.get_dataset()

    data_cleaner = DataCleaner(dataset)
    data_cleaner.drop_columns()

    dataset = data_cleaner.get_dataset()
    return dataset

if __name__ == "__main__":
     dataset = main()
     