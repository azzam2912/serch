import pandas as pd
import os
import time

# Read the CSV file
df = pd.read_csv('search_engine/wikIR1k/documents.csv')

# Create a directory to store the .txt files
os.makedirs('documents_database', exist_ok=True)

counter = -1
loop_counter = 0

# Loop over the DataFrame rows as (index, Series) pairs
for index, row in df.iterrows():
    if(loop_counter % 10000 == 0):
        counter += 1
        os.makedirs(f'documents_database/{counter}', exist_ok=True)
        print("indexing block number ", loop_counter)

    id = row['id_right']
    content = row['text_right']
    
    # Write the content to a .txt file named by the id
    with open(f'documents_database/{counter}/{id}.txt', 'w+', encoding='utf-8') as f:
        f.write(str(content))
    loop_counter += 1