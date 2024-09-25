
import pandas as pd
import numpy as np

# Step 1: Define the chunk size and relevant itemids
chunk_size = 500_000
relevant_itemids = {
    220045, 225309, 225310, 225312, 220050, 220051, 220052,
    220179, 220180, 220181, 220210, 224690, 220277, 225664, 
    220621, 226537, 223762, 223761, 224642
}

# Step 2: Function to optimize data types and filter data while reading CSV in chunks
def load_and_filter_csv(file_path, chunk_size, relevant_itemids):
    for chunk in pd.read_csv(
        file_path, 
        chunksize=chunk_size,
        dtype={
            'subject_id': 'int32',
            'hadm_id': 'int32',
            'stay_id': 'int32',
            'charttime': 'str',  # Keeping as string to parse later if needed
            'itemid': 'int32',
            'valuenum': 'float32'
        },
        usecols=['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'valuenum']  # Only load relevant columns
    ):
        # Filter chunk to only include relevant itemids
        chunk_filtered = chunk[chunk['itemid'].isin(relevant_itemids)]
        yield chunk_filtered

# Step 3: Optimize pivot and vital signs calculation functions
def pivot_vital_signs(df):
    return df.pivot_table(
        index=['subject_id', 'hadm_id', 'stay_id', 'charttime'],
        columns='itemid',
        values='valuenum',
        aggfunc='mean'  # Mean aggregation is fine here
    ).reset_index()

def calculate_vital_signs(df_pivot):
    # Check for the existence of columns before accessing them
    def get_column(df, col):
        return df[col] if col in df else np.nan

    # Heart rate
    df_pivot['heart_rate'] = get_column(df_pivot, 220045).where((get_column(df_pivot, 220045) > 0) & (get_column(df_pivot, 220045) < 300))

    # Systolic Blood Pressure (SBP)
    df_pivot['sbp'] = df_pivot[[col for col in [220179, 220050, 225309] if col in df_pivot]].where(lambda x: (x > 0) & (x < 400)).mean(axis=1)

    # Diastolic Blood Pressure (DBP)
    df_pivot['dbp'] = df_pivot[[col for col in [220180, 220051, 225310] if col in df_pivot]].where(lambda x: (x > 0) & (x < 300)).mean(axis=1)

    # Mean Blood Pressure (MBP)
    df_pivot['mbp'] = df_pivot[[col for col in [220052, 220181, 225312] if col in df_pivot]].where(lambda x: (x > 0) & (x < 300)).mean(axis=1)

    # Non-invasive Blood Pressure
    df_pivot['sbp_ni'] = get_column(df_pivot, 220179).where((get_column(df_pivot, 220179) > 0) & (get_column(df_pivot, 220179) < 400))
    df_pivot['dbp_ni'] = get_column(df_pivot, 220180).where((get_column(df_pivot, 220180) > 0) & (get_column(df_pivot, 220180) < 300))
    df_pivot['mbp_ni'] = get_column(df_pivot, 220181).where((get_column(df_pivot, 220181) > 0) & (get_column(df_pivot, 220181) < 300))

    # Respiratory Rate
    df_pivot['resp_rate'] = df_pivot[[col for col in [220210, 224690] if col in df_pivot]].where(lambda x: (x > 0) & (x < 70)).mean(axis=1)

    # Temperature
    df_pivot['temperature'] = np.where(
        (get_column(df_pivot, 223761) > 70) & (get_column(df_pivot, 223761) < 120),
        (get_column(df_pivot, 223761) - 32) / 1.8,
        get_column(df_pivot, 223762).where((get_column(df_pivot, 223762) > 10) & (get_column(df_pivot, 223762) < 50))
    )

    # Temperature site
    df_pivot['temperature_site'] = get_column(df_pivot, 224642)

    # SPO2
    df_pivot['spo2'] = get_column(df_pivot, 220277).where((get_column(df_pivot, 220277) > 0) & (get_column(df_pivot, 220277) <= 100))

    # Glucose
    df_pivot['glucose'] = df_pivot[[col for col in [225664, 220621, 226537] if col in df_pivot]].where(lambda x: x > 0).mean(axis=1)

    return df_pivot

# Step 4: Process CSV in chunks
def process_vital_signs_in_chunks(file_path, chunk_size):
    processed_chunks = []
    for chunk in load_and_filter_csv(file_path, chunk_size, relevant_itemids):
        df_pivot = pivot_vital_signs(chunk)
        df_vitals = calculate_vital_signs(df_pivot)
        processed_chunks.append(df_vitals)

    df_final = pd.concat(processed_chunks, ignore_index=True)
    return df_final

# Step 5: Run the processing function and print the result
df_vitals_final = process_vital_signs_in_chunks('chartevents.csv', chunk_size)

print(df_vitals_final.head())
