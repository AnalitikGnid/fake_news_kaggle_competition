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

# In the Analysis.py file, we noticed that there were some rows with empty text or text that was just a link.
# We will remove those rows from the cleaned data, as they were just fake data and we believe this to be an artifact of the data collection process.
clean_data = clean_data[~clean_data['text'].isnull() & (clean_data['text'] != '')]
clean_data = clean_data[~clean_data['text'].str.contains('httpswww', na=False)]
save_to_csv(clean_data, 'final_cleaned_combined_data.csv')
# Final cleaned data is saved to 'final_cleaned_combined_data.csv'

print("Data preprocessing and saving completed successfully.")