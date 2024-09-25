import pandas as pd
import numpy as np

# Specify the columns we need to load
usecols = ['subject_id', 'hadm_id', 'charttime', 'specimen_id', 'itemid', 'valuenum']

# Load the dataset in chunks to prevent memory overflow
chunksize = 500000  # Adjust the chunk size based on available memory

# Relevant item IDs
relevant_itemids = [
    50862, 50930, 50976, 50868, 50882, 51006, 50893, 
    50902, 50912, 50931, 50983, 50971
]

# Initialize an empty list to collect filtered chunks
filtered_chunks = []

# Iterate over chunks and filter them
for chunk in pd.read_csv('hosp\labevents.csv',
                         usecols=usecols, chunksize=chunksize):
    
    # Filter relevant rows in the current chunk
    chunk_filtered = chunk[
        (chunk['itemid'].isin(relevant_itemids)) & 
        ((chunk['valuenum'] > 0) | (chunk['itemid'] == 50868))
    ]
    
    # Append the filtered chunk to the list
    filtered_chunks.append(chunk_filtered)

# Concatenate all the filtered chunks into a single DataFrame
df_filtered = pd.concat(filtered_chunks)

# Pivot the DataFrame
df_pivot = df_filtered.pivot_table(
    index=['specimen_id'],
    values='valuenum',
    columns='itemid',
    aggfunc='max'
)

df_pivot = df_pivot.reset_index()

# Apply conditions to create new columns with unit limits
df_pivot['albumin'] = np.where(df_pivot[50862] <= 10, df_pivot[50862], np.nan)
df_pivot['globulin'] = np.where(df_pivot[50930] <= 10, df_pivot[50930], np.nan)
df_pivot['total_protein'] = np.where(df_pivot[50976] <= 20, df_pivot[50976], np.nan)
df_pivot['aniongap'] = np.where(df_pivot[50868] <= 10000, df_pivot[50868], np.nan)
df_pivot['bicarbonate'] = np.where(df_pivot[50882] <= 10000, df_pivot[50882], np.nan)
df_pivot['bun'] = np.where(df_pivot[51006] <= 300, df_pivot[51006], np.nan)
df_pivot['calcium'] = np.where(df_pivot[50893] <= 10000, df_pivot[50893], np.nan)
df_pivot['chloride'] = np.where(df_pivot[50902] <= 10000, df_pivot[50902], np.nan)
df_pivot['creatinine'] = np.where(df_pivot[50912] <= 150, df_pivot[50912], np.nan)
df_pivot['glucose'] = np.where(df_pivot[50931] <= 10000, df_pivot[50931], np.nan)
df_pivot['sodium'] = np.where(df_pivot[50983] <= 200, df_pivot[50983], np.nan)
df_pivot['potassium'] = np.where(df_pivot[50971] <= 30, df_pivot[50971], np.nan)

# Reduce memory usage by downcasting numeric columns
df_filtered['subject_id'] = df_filtered['subject_id'].astype('int32')
df_filtered['hadm_id'] = df_filtered['hadm_id'].astype('int32')
df_filtered['specimen_id'] = df_filtered['specimen_id'].astype('int32')

# Group by specimen_id to aggregate the max values of subject_id, hadm_id, and charttime
df_max_columns = df_filtered.groupby('specimen_id').agg({
    'subject_id': 'max',
    'hadm_id': 'max',
    'charttime': 'max'
}).reset_index()

# Merge the max columns with the pivoted DataFrame
blood_chemistry = pd.merge(df_max_columns, df_pivot, on='specimen_id', how='left')

# Display the result
print(blood_chemistry.head())
