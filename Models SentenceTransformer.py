import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, GRU
# from keras.preprocessing.text import Tokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
import os

def main() -> None:

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
        plt.savefig('text_length_distribution_from_ST.png')
        print('Length distribution plot saved')

    plot_length_distribution(data)

    file_path = 'encoded_text_ST.csv'
    retrain = False  # Set to True if you want to retrain the model and re-encode the data

    # We create this if-else statement to not perform unnecessary encoding if it already happened
    if os.path.exists(file_path) and not retrain:
        encoded_df = pd.read_csv(file_path)
    else:
        # Define the SentenceTransformer model
        # We use this one, since it is the one that was used in the Kaggle code, and Hugging Face is european based.
        model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')

        pool = model.start_multi_process_pool(['cpu', 'cpu'])

        texts = data['text'].tolist()

        emb = model.encode(texts, show_progress_bar=True, pool=pool, chunk_size=50)
        assert emb.shape[0] == len(texts), "Embedding size does not match number of texts"

        emb_normal = model.encode(texts)

        model.stop_multi_process_pool(pool)

        # Convert the encoded texts to a DataFrame
        encoded_df = pd.DataFrame(emb_normal.numpy(), columns=[f'feature_{i}' for i in range(emb_normal.shape[1])])
        # Add the label column to the encoded DataFrame
        encoded_df['label'] = data['label'].values
        # Save the encoded DataFrame to a CSV file
        encoded_df.to_csv(file_path, index=False)

if __name__ == "__main__":
    main()