import pandas as pd

# Function to group ICD-10-PCS codes into broader procedure categories
def map_icd10_procedure_to_group(icd_code):
    if icd_code.startswith('0'):
        return "Medical and Surgical Procedures"  # Leave Medical and Surgical Procedures as is
    elif icd_code.startswith('B'):
        return "Imaging Procedures"  # Diagnostic Imaging Procedures
    elif icd_code.startswith(('C', 'D')):
        return "Diagnostic Procedures"  # Diagnostic procedures, measurement, monitoring, etc.
    elif icd_code.startswith('F'):
        return "Therapeutic Procedures"  # Physical rehabilitation, therapeutic procedures
    elif icd_code.startswith('X'):
        return "New Technology Procedures"  # New technology procedures
    else:
        return "Other Procedures"  # Default for unknown or other categories

# Function to group ICD-9-PCS codes into broader procedure categories
def map_icd9_procedure_to_group(icd_code):
    try:
        icd_numeric = int(icd_code[:3])  # Take the first 3 digits for ICD-9-PCS codes
    except (ValueError, TypeError):  # Handle invalid or missing codes
        return "Unknown"

    if 0 <= icd_numeric <= 39:
        return "Medical and Surgical Procedures"
    elif 40 <= icd_numeric <= 49:
        return "Diagnostic Procedures"
    elif 50 <= icd_numeric <= 59:
        return "Therapeutic Procedures"
    else:
        return "Other Procedures"

# Function to map both ICD-9 and ICD-10 procedure codes based on version
def map_procedure_to_group(row):
    icd_code = row['icd_code']
    icd_version = row['icd_version']
    
    if icd_version == 9:
        return map_icd9_procedure_to_group(icd_code)
    elif icd_version == 10:
        return map_icd10_procedure_to_group(icd_code)
    else:
        return "Unknown"  # Handle missing or invalid versions

# Load the procedures data
df_procedures = pd.read_csv('hosp/procedures_icd.csv')

# Apply the mapping to the procedure codes
df_procedures['procedure_group'] = df_procedures.apply(map_procedure_to_group, axis=1)

# Count the occurrences of each procedure group for each patient and admission
df_procedures_grouped = df_procedures.groupby(['subject_id', 'hadm_id', 'procedure_group']).size().reset_index(name='procedure_count')

# Pivot the table to create columns for each procedure group, with the sum of procedures in each group
df_procedures_pivot = df_procedures_grouped.pivot_table(
    index=['subject_id', 'hadm_id'], 
    columns='procedure_group', 
    values='procedure_count', 
    aggfunc='sum', 
    fill_value=0
).reset_index()

# Rename pivoted columns to make them clear
df_procedures_pivot.columns = [f"procedures_{col}" if col not in ['subject_id', 'hadm_id'] else col for col in df_procedures_pivot.columns]

# Merge the procedure group data with the main patient data (assuming df_elderly is the patient data)
df_elderly = pd.merge(df_elderly, df_procedures_pivot, on=['subject_id', 'hadm_id'], how='left')

# Fill NaN values with 0 in the newly created procedure columns
procedure_columns = [col for col in df_procedures_pivot.columns if col.startswith('procedures_')]
df_elderly[procedure_columns] = df_elderly[procedure_columns].fillna(0)

# Check the result
print(df_elderly[procedure_columns].head())
