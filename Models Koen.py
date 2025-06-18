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
from sentence_transformers import SentenceTransformer

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
    plt.savefig('text_length_distribution.png')

plot_length_distribution(data)

# Define the SentenceTransformer model
# We use this one, since it is the one that was used in the Kaggle code, and Hugging Face is european based.
model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')

# Encode the text data using the SentenceTransformer model
def encode_texts(texts):
    """
    Encode texts using the SentenceTransformer model.
    """
    return model.encode(texts, show_progress_bar=True, convert_to_tensor=True)

encoded_texts = encode_texts(data['text'].tolist())
# Convert the encoded texts to a DataFrame
encoded_df = pd.DataFrame(encoded_texts.numpy(), columns=[f'feature_{i}' for i in range(encoded_texts.shape[1])])
# Add the label column to the encoded DataFrame
encoded_df['label'] = data['label'].values
# Save the encoded DataFrame to a CSV file
encoded_df.to_csv('encoded_texts.csv', index=False)