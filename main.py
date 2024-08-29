import FeatureSelection
import DataCleaner
import FeatureExtractor

def main():
    dataset_path = 'challenge_campus_biomedico_2024.parquet'
    url = 'https://www1.agenziaentrate.gov.it/servizi/codici/ricerca/VisualizzaTabella.php?ArcName=00T2'
    data_cleaner = DataCleaner.DataCleaner(dataset_path, url)
    data_cleaner.retrieve_data()
    data_cleaner.riempimento_codice_provincia()
    data_cleaner.riempimento_codice_provincia_erogazione()
    data_cleaner.drop_duplicate()
    data_cleaner.drop_visit_cancellation()
    data_cleaner.drop_column_id_professionista_sanitario()
    data_cleaner.delete_column_date_null()
    data_cleaner.calculate_duration_of_visit()
    data_cleaner.fill_duration_of_visit()
    data_cleaner.calculate_age()
    dataset = data_cleaner.dataset
    print (dataset.info())
    
    
    feature_selection = FeatureSelection.FeatureSelection(dataset)

    feature_selection.eliminate_highly_correlated_columns()
    print (dataset.info())
    
    #extractor = FeatureExtractor.FeatureExtractor(dataset)
   # df_with_features = extractor.extract_features()
    
    
    

if __name__ == '__main__':
    main()
