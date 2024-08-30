import pandas as pd
from DataCleaner import DataCleaner
from FeatureSelection import FeatureSelection  # Importa la classe correttamente
import FeatureExtractor

def main():
    dataset_path = 'challenge_campus_biomedico_2024.parquet'
    dataset = pd.read_parquet(dataset_path)

    # Creare un'istanza di DataCleaner
    data_cleaner = DataCleaner(dataset)

    # Recuperare i dati utilizzando il metodo di istanza
    data_cleaner.retrieve_data()
  
    dataset = data_cleaner.riempimento_codice_provincia()
    dataset = data_cleaner.riempimento_codice_provincia_erogazione()
    dataset = data_cleaner.drop_duplicate()
    dataset = data_cleaner.drop_column_id_professionista_sanitario()

    # Creare un'istanza di FeatureSelection
    feature_selector = FeatureSelection(dataset)

    # Utilizza la funzione dell'istanza
    dataset = feature_selector.eliminate_highly_correlated_columns()

    dataset = FeatureExtractor.filter_teleassistenza(dataset)
    dataset = FeatureExtractor.calculate_quadrimester(dataset)
    dataset = FeatureExtractor.calculate_increment_by_quadrimester(dataset)
    dataset = FeatureExtractor.calculate_age(dataset)
    dataset = FeatureExtractor.calculate_duration_of_visit(dataset)

    dataset = data_cleaner.drop_visit_cancellation()
    dataset = data_cleaner.delete_column_date_null()
    dataset = data_cleaner.fill_duration_of_visit()
    dataset = data_cleaner.update_dataset_with_outliers()

    print(dataset.info())

if __name__ == '__main__':
    main()
