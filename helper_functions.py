import pandas as pd
import numpy as np
# Download stopwords if not already downloaded
import nltk
#nltk.download('stopwords')
# Import stopwords
from nltk.corpus import stopwords
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns
from pathlib import Path

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: The loaded data as a DataFrame.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the data by removing unnecessary columns and handling missing values.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to preprocess.
    
    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    # Remove unnecessary columns
    df = df.drop(columns=['date', 'text subject'], errors='ignore')
    
    # Handle missing values
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    
    return df

def combine_data(fake_df, true_df):
    """
    Combine fake and true data into a single DataFrame with an additional label column.
    
    Parameters:
    fake_df (pd.DataFrame): The DataFrame containing fake news data.
    true_df (pd.DataFrame): The DataFrame containing true news data.
    
    Returns:
    pd.DataFrame: The combined DataFrame with an additional 'label' column.
    """
    fake_df['label'] = 0  # Label for fake news
    true_df['label'] = 1  # Label for true news
    
    combined_df = pd.concat([fake_df, true_df], ignore_index=True)

    combined_df = combined_df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame
    
    return combined_df

def clean_text(text):
    """
    Clean the text by removing special characters and converting to lowercase.
    Also remove stopwords
    
    Parameters:
    text (str): The text to clean.
    
    Returns:
    str: The cleaned text.
    """
    if pd.isna(text):
        return ''
    
    # Remove special characters and digits
    text = ''.join(char for char in text if char.isalpha() or char.isspace())
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    # Remove extra spaces
    text = ' '.join(text.split())

    # Remove leading and trailing spaces
    text = text.strip()
    # Return the cleaned text

    return text

def preprocess_text_column(df, column_name):
    """
    Preprocess a specific text column in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the text column.
    column_name (str): The name of the text column to preprocess.
    
    Returns:
    pd.DataFrame: The DataFrame with the preprocessed text column.
    """
    df[column_name] = df[column_name].apply(clean_text)
    return df

def save_to_csv(df, file_path):
    """
    Save the DataFrame to a CSV file.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    file_path (str): The path where the CSV file will be saved.
    """
    df.to_csv(file_path, index=False)

def evaluate_classifier(y_true, y_score, threshold=0.5, sample_weight=None, plot=True, model_name = "Classifier"):
    """
    Calculate ROC curve, AUC, and confusion matrix for a classifier, and optionally plot them.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, sample_weight=sample_weight)
    auc_score = auc(fpr, tpr)
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)

    if plot:
        # Plot ROC curve
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()
        plt.savefig(f'{model_name}_roc_curve.png')

        # Plot confusion matrix
        plt.figure(figsize=(4, 4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Fake News', 'True News'])
        plt.yticks(tick_marks, ['Fake News', 'True News'])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        for i in range(2):
            for j in range(2):
                plt.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')
        plt.tight_layout()
        plt.show()
        plt.savefig(f'{model_name}_confusion_matrix.png')

    return fpr, tpr, thresholds, auc_score, cm

def plot_efficiencies(model_outputs, true_labels, model_name="Classifier"):
    """
    Plot the efficiencies of different classifiers.
    
    Parameters:
    efficiencies (dict): A dictionary where keys are classifier names and values are efficiency values.
    """
    thresholds = np.linspace(0, 1, 100)
    signal_efficiencies = []
    background_efficiencies = []
    accuracies_per_threshold = []
    for threshold in thresholds:
        signal_efficiency = np.mean(model_outputs[true_labels == 1] >= threshold)
        background_efficiency = np.mean(model_outputs[true_labels == 0] >= threshold)

        signal_efficiencies.append(signal_efficiency)
        background_efficiencies.append(background_efficiency)

        accuracy_per_threshold = np.mean((model_outputs >= threshold) == true_labels)
        accuracies_per_threshold.append(accuracy_per_threshold)
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, signal_efficiencies, label='True News Efficiency', color='blue')
    plt.plot(thresholds, background_efficiencies, label='Fake News Efficiency', color='red')
    plt.plot(thresholds, accuracies_per_threshold, label='Accuracy', color='green', linestyle='--')
    plt.xlabel('Threshold')
    plt.ylabel('Efficiency/Accuracy')
    plt.title('True and Fake News Efficiencies and Accuracy vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'{model_name}_efficiencies.png')
    

def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    
    # Save the model
    model.save(f'best_models_final/{model_name}.h5')
    
    # Plot training & validation accuracy and loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'best_models_plots/{model_name}_training_plot.png')
    plt.show()

    # Get prediction probabilities on test set
    y_pred_prob = model.predict(X_test)
    # Create histogram of prediction probabilities vs actual labels
    plt.figure(figsize=(8, 6))
    sns.histplot(data=pd.DataFrame({'Probability': y_pred_prob.flatten(), 'Actual': y_test}), 
                 x='Probability', hue='Actual', multiple='stack', bins=30)
    plt.title(f'{model_name} Prediction Probability Histogram')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Count')
    plt.savefig(f'best_models_plots/{model_name}_probability_histogram.png')
    plt.show()
    # Print classification report and confusion matrix
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f'{model_name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'best_models_plots/{model_name}_confusion_matrix.png')
    plt.show()
        
