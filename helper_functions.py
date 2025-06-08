import pandas as pd
# Download stopwords if not already downloaded
import nltk
nltk.download('stopwords')
# Import stopwords
from nltk.corpus import stopwords

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