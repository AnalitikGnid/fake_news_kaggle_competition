import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU

from pathlib import Path

from helper_functions import train_and_evaluate

# We begin by loading the two datasets
ST_data = pd.read_csv('ST.csv')
Tfidf_data = pd.read_csv('encoded_text_Tfidf.csv')


# We get the ST data ready for training
embeddings = ST_data.iloc[:, :-1].values
print('data shape ', ST_data.shape)
# Reshape for LSTM: (samples, timesteps, features)
X = np.expand_dims(embeddings, axis=1)  # Shape: (samples, 1, 384)
y = ST_data['label'].values  # Shape: (samples,)
print(f'X {X.shape}, y {y.shape}')
# Train/test split
ST_X_train, ST_X_test, ST_y_train, ST_y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train/Validation split
ST_X_train, ST_X_val, ST_y_train, ST_y_val = train_test_split(ST_X_train, ST_y_train, test_size=0.2, random_state=42)

# We get the Tfidf data ready for training
print(Tfidf_data.shape)
TV_X = Tfidf_data.iloc[:, :-1].values
print(TV_X.shape)
labels = Tfidf_data['label_to_predict'].values
y = labels

# Reshape TF-IDF output to simulate sequence for LSTM
TV_X_reshaped = TV_X.reshape(TV_X.shape[0], 25, 10)

# Train/test split
TV_X_train, TV_X_test, TV_y_train, TV_y_test = train_test_split(TV_X_reshaped, y, test_size=0.2, random_state=42)
# Train/Validation split
TV_X_train, TV_X_val, TV_y_train, TV_y_val = train_test_split(TV_X_train, TV_y_train, test_size=0.2, random_state=42)

# Define model architectures as functions for reuse
def build_model_LSTM(X_train, units, dropout, hidden=None):
    model = Sequential()
    model.add(LSTM(units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout))
    if hidden:
        model.add(Dense(hidden, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_model_GRU(X_train, units, dropout, hidden=None):
    model = Sequential()
    model.add(GRU(units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout))
    if hidden:
        model.add(Dense(hidden, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# We know from previous experiments that the best model are:
ST_LSTM_model = build_model_LSTM(ST_X_train, 64, 0.3, 16)
TV_LSTM_model = build_model_LSTM(TV_X_train, 128, 0.3, None)

ST_GRU_model = build_model_GRU(ST_X_train, 128, 0.3, 16)
TV_GRU_model = build_model_GRU(TV_X_train, 64, 0.4, None)

output_dir_plots = Path('best_models_plots')
output_dir_plots.mkdir(parents=True, exist_ok=True)

output_dir_models = Path('best_models_final')
output_dir_models.mkdir(parents=True, exist_ok=True)

# Train and evaluate each model
train_and_evaluate(ST_LSTM_model, ST_X_train, ST_y_train, ST_X_val, ST_y_val, ST_X_test, ST_y_test, "ST_LSTM")
train_and_evaluate(TV_LSTM_model, TV_X_train, TV_y_train, TV_X_val, TV_y_val, TV_X_test, TV_y_test, "TV_LSTM")
train_and_evaluate(ST_GRU_model, ST_X_train, ST_y_train, ST_X_val, ST_y_val, ST_X_test, ST_y_test, "ST_GRU")
train_and_evaluate(TV_GRU_model, TV_X_train, TV_y_train, TV_X_val, TV_y_val, TV_X_test, TV_y_test, "TV_GRU")

