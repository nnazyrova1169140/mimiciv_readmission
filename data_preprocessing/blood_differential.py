import pandas as pd
import numpy as np

# Define relevant item IDs
relevant_itemids = [
    51146, 52069, 51199, 51200, 52073, 51244, 51245, 51133, 52769, 
    51253, 51254, 52074, 51256, 52075, 51143, 51144, 51218, 52135, 
    51251, 51257, 51300, 51301, 51755
]

# Define data types for more efficient memory usage
dtype_dict = {
    'subject_id': 'int32',
    'hadm_id': 'int32',
    'specimen_id': 'int32',
    'itemid': 'int32',
    'valuenum': 'float32'
}

# Load dataset in chunks (adjust chunk size based on memory limits)
chunksize = 10 ** 6  # 1 million rows per chunk

# Initialize an empty list to store processed chunks
processed_chunks = []

# Process each chunk
for chunk in pd.read_csv('hosp\labevents.csv', 
                         usecols=['subject_id', 'hadm_id', 'specimen_id', 'itemid', 'valuenum', 'charttime'], 
                         dtype=dtype_dict, 
                         chunksize=chunksize):
    
    # Filter for relevant itemids and non-negative valuenum early
    chunk_filtered = chunk[(chunk['itemid'].isin(relevant_itemids)) & (chunk['valuenum'] >= 0)]
    
    # Append to the list of processed chunks
    processed_chunks.append(chunk_filtered)

# Concatenate the processed chunks
df_filtered = pd.concat(processed_chunks)

# Free up memory from intermediate chunks
del processed_chunks

# Reduce the memory of the 'charttime' column if applicable
df_filtered['charttime'] = pd.to_datetime(df_filtered['charttime'], errors='coerce')

# Group the filtered data by specimen_id, performing necessary aggregation
df_grouped = df_filtered.groupby('specimen_id').agg(
    subject_id=('subject_id', 'max'),
    hadm_id=('hadm_id', 'max'),
    charttime=('charttime', 'max'),
    # WBC related columns
    wbc=('valuenum', lambda x: max(x[df_filtered['itemid'].isin([51300, 51301, 51755])], default=np.nan)),
    basophils_abs=('valuenum', lambda x: max(x[df_filtered['itemid'] == 52069], default=np.nan)),
    eosinophils_abs=('valuenum', lambda x: max(x.apply(lambda y: y / 1000.0 if df_filtered['itemid'] == 51199 else y if df_filtered['itemid'] == 52073 else np.nan), default=np.nan)),
    lymphocytes_abs=('valuenum', lambda x: max(x.apply(lambda y: y / 1000.0 if df_filtered['itemid'] == 52769 else y if df_filtered['itemid'] == 51133 else np.nan), default=np.nan)),
    monocytes_abs=('valuenum', lambda x: max(x.apply(lambda y: y / 1000.0 if df_filtered['itemid'] == 51253 else y if df_filtered['itemid'] == 52074 else np.nan), default=np.nan)),
    neutrophils_abs=('valuenum', lambda x: max(x[df_filtered['itemid'] == 52075], default=np.nan)),
    granulocytes_abs=('valuenum', lambda x: max(x.apply(lambda y: y / 1000.0 if df_filtered['itemid'] == 51218 else np.nan), default=np.nan)),
    # Percentages
    basophils=('valuenum', lambda x: max(x[df_filtered['itemid'] == 51146], default=np.nan)),
    eosinophils=('valuenum', lambda x: max(x[df_filtered['itemid'] == 51200], default=np.nan)),
    lymphocytes=('valuenum', lambda x: max(x[df_filtered['itemid'].isin([51244, 51245])], default=np.nan)),
    monocytes=('valuenum', lambda x: max(x[df_filtered['itemid'] == 51254], default=np.nan)),
    neutrophils=('valuenum', lambda x: max(x[df_filtered['itemid'] == 51256], default=np.nan)),
    atypical_lymphocytes=('valuenum', lambda x: max(x[df_filtered['itemid'] == 51143], default=np.nan)),
    bands=('valuenum', lambda x: max(x[df_filtered['itemid'] == 51144], default=np.nan)),
    immature_granulocytes=('valuenum', lambda x: max(x[df_filtered['itemid'] == 52135], default=np.nan)),
    metamyelocytes=('valuenum', lambda x: max(x[df_filtered['itemid'] == 51251], default=np.nan)),
    nrbc=('valuenum', lambda x: max(x[df_filtered['itemid'] == 51257], default=np.nan))
).reset_index()

# Imputation logic for absolute counts based on percentages
df_grouped['basophils_abs'] = np.where(df_grouped['basophils_abs'].isnull() & df_grouped['basophils'].notnull() & (df_grouped['wbc'] > 0), df_grouped['basophils'] * df_grouped['wbc'] / 100, df_grouped['basophils_abs'])
df_grouped['eosinophils_abs'] = np.where(df_grouped['eosinophils_abs'].isnull() & df_grouped['eosinophils'].notnull() & (df_grouped['wbc'] > 0), df_grouped['eosinophils'] * df_grouped['wbc'] / 100, df_grouped['eosinophils_abs'])
df_grouped['lymphocytes_abs'] = np.where(df_grouped['lymphocytes_abs'].isnull() & df_grouped['lymphocytes'].notnull() & (df_grouped['wbc'] > 0), df_grouped['lymphocytes'] * df_grouped['wbc'] / 100, df_grouped['lymphocytes_abs'])
df_grouped['monocytes_abs'] = np.where(df_grouped['monocytes_abs'].isnull() & df_grouped['monocytes'].notnull() & (df_grouped['wbc'] > 0), df_grouped['monocytes'] * df_grouped['wbc'] / 100, df_grouped['monocytes_abs'])
df_grouped['neutrophils_abs'] = np.where(df_grouped['neutrophils_abs'].isnull() & df_grouped['neutrophils'].notnull() & (df_grouped['wbc'] > 0), df_grouped['neutrophils'] * df_grouped['wbc'] / 100, df_grouped['neutrophils_abs'])

# Round to 4 decimal places
df_grouped[['basophils_abs', 'eosinophils_abs', 'lymphocytes_abs', 'monocytes_abs', 'neutrophils_abs']] = df_grouped[['basophils_abs', 'eosinophils_abs', 'lymphocytes_abs', 'monocytes_abs', 'neutrophils_abs']].round(4)

# Display the results
print(df_grouped.head())
