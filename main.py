import FeatureSelection
import DataCleaner

def main():
    dataset_path = 'challenge_campus_biomedico_2024.parquet'
    url = 'https://www1.agenziaentrate.gov.it/servizi/codici/ricerca/VisualizzaTabella.php?ArcName=00T2'
    data_cleaner = DataCleaner.DataCleaner(dataset_path, url)
    data_cleaner.retrieve_data()
    data_cleaner.riempimento_codice_provincia()
    dataset = data_cleaner.dataset
    feature_selection = FeatureSelection.FeatureSelection(dataset)
    feature_selection.duration_of_visit()
    feature_selection.drop_columns_inizio_e_fine_prestazione()
    feature_selection.calculate_age()
    feature_selection.dataset

if __name__ == '__main__':
    main()
