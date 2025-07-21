import pandas as pd

clean_data = pd.read_csv('cleaned_combined_data.csv')


# I want to print all the rows where the text is empty
empty_rows = clean_data[clean_data['text'].isnull() == True]

# Count the number of labels in each category of the empty rows
label_counts = empty_rows['label'].value_counts()
print("Rows with empty text:")
print(empty_rows)
print("\nLabel counts:")
print(label_counts)

# Print all the rows where the text is just a link, which looks like this: httpswwwyoutubecomwatchvntfnmqpywu
link_rows = clean_data[clean_data['text'].str.contains('httpswww', na=False)]
label_counts_links = link_rows['label'].value_counts()
print("\nRows with text containing links:")
print(link_rows)
print("\nLabel counts for link rows:")
print(label_counts_links)
# Print all the rows where the text is just a link, which looks like this: httpswwwyoutubecomwatchvntfnmqpywu
print(link_rows.head(10))  # Display the first 10 rows with links

# Find the label counts of the entire dataset
label_counts_all = clean_data['label'].value_counts()
print("\nLabel counts for the entire dataset:")
print(label_counts_all)

# Get the label counts after removing the empty and link rows
final_cleaned_data = pd.read_csv('final_cleaned_combined_data.csv')
final_label_counts = final_cleaned_data['label'].value_counts()
print("\nLabel counts after cleaning:")
print(final_label_counts)