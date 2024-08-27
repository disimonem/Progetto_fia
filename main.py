import FeatureSelection
import DataCleaner
import FeatureExtractor

def main():
    dataset_path = 'challenge_campus_biomedico_2024.parquet'
    url = 'https://www1.agenziaentrate.gov.it/servizi/codici/ricerca/VisualizzaTabella.php?ArcName=00T2'
    data_cleaner = DataCleaner.DataCleaner(dataset_path, url)
    data_cleaner.retrieve_data()
    data_cleaner.riempimento_codice_provincia()
    dataset = data_cleaner.dataset
    feature_selection = FeatureSelection.FeatureSelection(dataset)
    feature_selection.compute_correlation_matrix()
    feature_selection.eliminate_highly_correlated_columns()
    feature_selection.plot_corrrelation_matrix()
    extractor = FeatureExtractor.FeatureExtractor(dataset)
    df_with_features = extractor.extract_features(dataset)
    print(df_with_features.head())
    
    

if __name__ == '__main__':
    main()
