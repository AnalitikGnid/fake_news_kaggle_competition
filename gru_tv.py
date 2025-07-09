import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Load data
df = pd.read_csv('encoded_text_Tfidf.csv')
print(df.shape)

# Features and labels
X = df.iloc[:, :-1].values
print(X.shape)
labels = df['label_to_predict'].values
y = to_categorical(labels)

# Reshape TF-IDF data to simulate a sequence format for GRU
X_reshaped = X.reshape(X.shape[0], 100, 50)  # Ensure this matches TF-IDF shape

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)
print('Data split complete')

# We apply 3-fold cross-validation to find the best model parameters.
# Create three folds for cross-validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True, random_state=42)
# Initialize lists to store results
fold_results = []

# Define model architectures as functions for reuse
def build_model(units, dropout, hidden=None):
    model = Sequential()
    model.add(GRU(units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout))
    if hidden:
        model.add(Dense(hidden, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# List of model configs: (units, dropout, hidden)
model_configs = [
    (32, 0.3, None),
    (64, 0.3, None),
    (128, 0.3, None),
    (32, 0.2, None),
    (64, 0.4, None),
    (128, 0.5, None),
    (32, 0.3, 16),
    (64, 0.3, 16),
    (128, 0.3, 16),
    (32, 0.2, 16),
    (64, 0.4, 16),
    (128, 0.5, 16),
]

# Store average accuracy for each model config
avg_accuracies = []

# Ensure that the output directory exists
output_dir = Path('models')
output_dir.mkdir(parents=True, exist_ok=True)

for idx, (units, dropout, hidden) in enumerate(model_configs):
    fold_accuracies = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model = build_model(units, dropout, hidden)
        model.fit(X_tr, y_tr, epochs=10, batch_size=32, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracies.append(val_acc)
        model.save(output_dir / f'model_{idx+1}_fold_{len(fold_accuracies)}_gru_tv.h5')
    avg_acc = np.mean(fold_accuracies)
    avg_accuracies.append(avg_acc)
    print(f"Model {idx+1}: avg 3-fold accuracy = {avg_acc:.4f}")

# Pick the best model config
best_idx = np.argmax(avg_accuracies)
best_config = model_configs[best_idx]
print(f"\nBest model: {best_idx+1} with config {best_config} (avg accuracy={avg_accuracies[best_idx]:.4f})")

# Retrain best model on full training set and evaluate on test set
best_model = build_model(*best_config)
best_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
y_pred_probs = best_model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

output_dir_best = Path('best_model')
output_dir_best.mkdir(parents=True, exist_ok=True)
best_model.save(output_dir_best / 'best_gru_tv.h5')

from helper_functions import *

# Evaluate the best model
fpr, tpr, thresholds, auc_score, cm = evaluate_classifier(y_test, y_pred_probs.flatten(), threshold=0.5, plot=True, model_name="GRU_TV")
# Plot efficiencies
plot_efficiencies(y_pred_probs.flatten(), y_test, model_name="GRU_TV")
