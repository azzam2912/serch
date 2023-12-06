import pandas as pd
import os

# Read the CSV file
df = pd.read_csv('documents.csv')

# Create a directory to store the .txt files
os.makedirs('documents_database', exist_ok=True)

# Loop over the DataFrame rows as (index, Series) pairs
for index, row in df.iterrows():
    id = row['id_right']
    content = row['text_right']
    
    # Write the content to a .txt file named by the id
    with open(f'text_files/{id}.txt', 'w', encoding='utf-8') as f:
        f.write(str(content))
