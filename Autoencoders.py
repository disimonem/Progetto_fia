import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import feature_selection 
dataset = pd.read_parquet('challenge_campus_biomedico_2024.parquet')

df= feature_selection.dataset_preprocessing(dataset)
print("Colonne dopo la pre-elaborazione:", df.columns)

print(df.head())
df = df.drop(['label'], axis=1).fillna(0)
# Dividere in caratteristiche numeriche e categoriali
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['bool', 'int32']).columns

# Normalizzare le caratteristiche numeriche
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Convertire le colonne categoriali in numeri (se necessario)
df[categorical_cols] = df[categorical_cols].astype(float)

# Definire dimensione input
input_dim = df.shape[1]

# Creazione dell'Autoencoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)  # Layer di encoding
encoded = Dense(16, activation='relu')(encoded)      # Ulteriore compressione

latent_dim = 20  # Dimensione del layer latente
encoded = Dense(latent_dim, activation='relu')(encoded)

decoded = Dense(16, activation='relu')(encoded)      # Layer di decoding
decoded = Dense(32, activation='relu')(decoded)      # Ulteriore espansione
decoded = Dense(input_dim, activation='sigmoid')(decoded)  # Ricostruzione finale

# Creare il modello di Autoencoder
autoencoder = Model(inputs=input_layer, outputs=decoded)

# Compilare il modello
autoencoder.compile(optimizer=Adam(), loss='mse')

# Addestramento del modello
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))

# Estrarre il modello Encoder per Feature Extraction
encoder = Model(inputs=input_layer, outputs=encoded)

# Utilizzare l'encoder per ottenere le nuove caratteristiche
encoded_features = encoder.predict(df)
print("Feature estratte tramite autoencoder:")
print(encoded_features)
