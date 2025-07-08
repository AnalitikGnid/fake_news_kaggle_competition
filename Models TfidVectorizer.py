import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, GRU
# from keras.preprocessing.text import Tokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

data = load_data('final_cleaned_combined_data.csv')

def plot_length_distribution(data):
    """
    Plot the distribution of text lengths in the dataset comparing real and fake data.
    """
    data['text_length'] = data['text'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x='text_length', hue='label', multiple='stack', bins=30)
    plt.title('Distribution of Text Lengths by Label')
    plt.xlabel('Text Length (in words)')
    plt.ylabel('Frequency')
    plt.show()
    # Save the plot
    plt.savefig('text_length_distribution_from_Tfidf.png')
    print('Length distribution plot saved')

#plot_length_distribution(data)

file_path = 'encoded_text_Tfidf.csv'
retrain = False  # Set to True if you want to retrain the model and re-encode the data

if os.path.exists(file_path) and not retrain:
    encoded_data = pd.read_csv(file_path)
    # Check if the label_to_predict column exists
    if 'label_to_predict' not in encoded_data.columns:
        raise ValueError("The target column is missing in the encoded data.")
    else:
        print(f"Encoded data loaded from {file_path}")
        print(encoded_data.shape)
else:
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # Fit and transform the text data
    X = vectorizer.fit_transform(data['text']).toarray()
    
    # Create a DataFrame with the encoded features
    encoded_data = pd.DataFrame(X, columns=vectorizer.get_feature_names_out())
    
    # Add the label column
    encoded_data['label_to_predict'] = data['label'].values
    
    # Save the encoded data to a CSV file
    encoded_data.to_csv(file_path, index=False)