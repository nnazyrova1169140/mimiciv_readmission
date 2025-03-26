import pandas as pd
from scipy.sparse import csr_matrix
from itertools import combinations
import numpy as np
from multiprocessing import Pool
from scipy.stats import hmean

# dataframe has hadm_id (admission id), readmission (readmission status, which is used as an indicator of adverse drug event, other adverse clinical events can be used, such as mortality), gsn (Generic Sequence Number of drugs, identifies a drug's clinical formulation)
baseline_readmission_rate = 0.13 #this is the average readmission rate in many hospital worldwide, higher than baseline rates considered to be exsessive readmissions

# Step 1: Map each unique hadm_id and drug to integer indices
unique_admissions = df_drugs['hadm_id'].unique()
admission_index = {hadm_id: idx for idx, hadm_id in enumerate(unique_admissions)}
df_drugs['admission_index'] = df_drugs['hadm_id'].map(admission_index)

# Map each unique drug to a column index
unique_drugs = df_drugs['gsn'].unique()
drug_index = {drug: idx for idx, drug in enumerate(unique_drugs)}
df_drugs['drug_index'] = df_drugs['gsn'].map(drug_index)

# Create a sparse matrix for drug prescriptions
admission_drug_matrix = csr_matrix(
    (np.ones(len(df_drugs)), 
     (df_drugs['admission_index'], df_drugs['drug_index'])),
    shape=(len(unique_admissions), len(unique_drugs))
)

# Step 2: Calculate co-occurrences with matrix multiplication
cooccurrence_matrix = (admission_drug_matrix.T @ admission_drug_matrix).tocoo()

# Step 3: Filter based on support thresholds
min_support, max_support = 100, 15000
valid_pairs = []
for i, j, count in zip(cooccurrence_matrix.row, cooccurrence_matrix.col, cooccurrence_matrix.data):
    if i < j and min_support <= count <= max_support:
        valid_pairs.append((unique_drugs[i], unique_drugs[j], count))

# Step 4: Calculate readmission rates and LIFT for each valid pair
def calculate_combination_metrics(pair):
    drug1, drug2 = pair[0], pair[1]
    
    # Patients who took only drug1, only drug2, and both drugs
    patients_drug1 = df_drugs[df_drugs['gsn'] == drug1]
    patients_drug2 = df_drugs[df_drugs['gsn'] == drug2]
    patients_both = df_drugs[df_drugs['gsn'].isin([drug1, drug2])].groupby('hadm_id').filter(lambda x: set([drug1, drug2]).issubset(set(x['gsn'])))
    
    # Frequencies (support) of drug1, drug2, and both drugs
    freq_drug1 = patients_drug1['hadm_id'].nunique()
    freq_drug2 = patients_drug2['hadm_id'].nunique()
    freq_both = patients_both['hadm_id'].nunique()
    
    # Readmission rates for each case
    readmission_rate_drug1 = patients_drug1['readmission'].mean()
    readmission_rate_drug2 = patients_drug2['readmission'].mean()
    readmission_rate_both = patients_both['readmission'].mean()
    
    # Readmission rates for drug1-only and drug2-only patients
    patients_drug1_only = patients_drug1[~patients_drug1['hadm_id'].isin(patients_both['hadm_id'])]
    patients_drug2_only = patients_drug2[~patients_drug2['hadm_id'].isin(patients_both['hadm_id'])]
    readmission_rate_drug1_only = patients_drug1_only['readmission'].mean()
    readmission_rate_drug2_only = patients_drug2_only['readmission'].mean()
    
    # Calculate LIFT for drug combination
    lift_drugs = freq_both / (freq_drug1 * freq_drug2)
    
    # Calculate LIFT for readmission rates
    if readmission_rate_drug1 * readmission_rate_drug2 > 0:
        lift_readmission = readmission_rate_both / (readmission_rate_drug1 * readmission_rate_drug2)
    else:
        lift_readmission = np.nan  # Handle cases where either rate is zero
    
    # Calculate Harmonic Mean of LIFTs
    if np.isnan(lift_readmission) or lift_readmission == 0:
        harmonic_mean_lift = np.nan
    else:
        harmonic_mean_lift = hmean([lift_drugs, lift_readmission])
    
    return {
        "Drug 1": drug1,
        "Drug 2": drug2,
        "Freq Drug 1": freq_drug1,
        "Freq Drug 2": freq_drug2,
        "Readmission Rate Drug 1 Only": readmission_rate_drug1_only,
        "Readmission Rate Drug 2 Only": readmission_rate_drug2_only,
        "Freq Both": freq_both,
        "Readmission Rate Both": readmission_rate_both,
        "LIFT Drugs": lift_drugs,
        "LIFT Readmission": lift_readmission,
        "Harmonic Mean LIFT": harmonic_mean_lift
    }

with Pool() as pool:
    combination_metrics = pool.map(calculate_combination_metrics, valid_pairs)

df_combination_metrics = pd.DataFrame(combination_metrics)
df_combination_metrics = df_combination_metrics.dropna(subset=['Harmonic Mean LIFT']).query("`Harmonic Mean LIFT` > 0")
