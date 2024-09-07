import pandas as pd
from sklearn.pipeline import Pipeline
from  CustomPipeline import CustomPipeline

def main():
    # Carica il dataset
    df = pd.read_parquet('challenge_campus_biomedico_2024.parquet')
    
    # Inizializza la pipeline personalizzata
    pipeline = CustomPipeline()
    
    # Esegui la pipeline sul dataset
    # Se hai delle etichette reali per calcolare il Purity Score, passale come secondo argomento
    pipeline.execute_pipeline(df, y_true_column='label')

if __name__ == "__main__":
    main()
