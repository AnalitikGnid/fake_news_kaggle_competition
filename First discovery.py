import pandas as pd 
import numpy as np
from helper_functions import *
from tqdm import tqdm

fake_data = load_data('Fake.csv')
true_data = load_data('True.csv')

fake_data = preprocess_data(fake_data)
true_data = preprocess_data(true_data)

combined_data = combine_data(fake_data, true_data)
combined_data.to_csv('combined_data.csv', index=False)

clean_rows = []

for index, row in tqdm(combined_data.iterrows()):
    cleaned_text = clean_text(row['text'])
    cleaned_title = clean_text(row['title'])
    clean_rows.append({'title': cleaned_title, 'text': cleaned_text, 'label': row['label']})

clean_data = pd.DataFrame(clean_rows, columns=['title', 'text', 'label'])
save_to_csv(clean_data, 'cleaned_combined_data.csv')
print("Data preprocessing and saving completed successfully.")
