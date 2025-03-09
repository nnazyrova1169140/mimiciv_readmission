df_drugs = pd.read_csv('hosp/prescriptions.csv')
df_drugs = df_drugs[df_drugs['hadm_id'].isin(df_elderly['hadm_id'])]


# Calculate each component of MCI
unique_drug_count = df_drugs.groupby('hadm_id')['gsn'].nunique()
dose_frequency_sum = df_drugs.groupby('hadm_id')['dose_val_rx'].count()
route_count = df_drugs.groupby('hadm_id')['route'].nunique()
dosage_adjustments = df_drugs.groupby(['hadm_id', 'drug'])['dose_val_rx'].nunique() - 1
dosage_adjustments_sum = dosage_adjustments.groupby('hadm_id').sum()
df_drugs['duration'] = (pd.to_datetime(df_drugs['stoptime']) - pd.to_datetime(df_drugs['starttime'])).dt.days
duration_sum = df_drugs.groupby('hadm_id')['duration'].sum()

# Combine components (example with normalized components)
unique_drug_count_norm = unique_drug_count / unique_drug_count.max()
dose_frequency_sum_norm = dose_frequency_sum / dose_frequency_sum.max()
route_count_norm = route_count / route_count.max()
dosage_adjustments_sum_norm = dosage_adjustments_sum / dosage_adjustments_sum.max()
duration_sum_norm = duration_sum / duration_sum.max()

# Calculate MCI
mci = (unique_drug_count_norm + dose_frequency_sum_norm + route_count_norm +
       dosage_adjustments_sum_norm + duration_sum_norm)

# Create a final dataframe with MCI
mci_df = pd.DataFrame({
    'hadm_id': mci.index,
    'Medication Complexity Index': mci.values
})
