######################################################################################################################################################################################################
# this code is an adapted version of SQL code for charlson comorbidity code from MIT/LCP/mimic_code: https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/comorbidity/charlson.sql
######################################################################################################################################################################################################
import pandas as pd
import numpy as np

# Step 1: Load the data
diagnoses_icd =  pd.read_csv('mimic-iv-3.0/mimic-iv-3.0/hosp/diagnoses_icd.csv')
admissions =df_final     # Admissions data
#age_df = df_final['age']
# Step 2: Separate ICD-9 and ICD-10 codes
def extract_icd_codes(df):
    df['icd9_code'] = df['icd_code'].where(df['icd_version'] == 9, None)
    df['icd10_code'] = df['icd_code'].where(df['icd_version'] == 10, None)
    return df

diagnoses_icd = extract_icd_codes(diagnoses_icd)

# Step 3: Define comorbidities based on ICD-9 and ICD-10
def calculate_comorbidities(df):
    # Myocardial Infarction
    df['myocardial_infarct'] = np.where(
        (df['icd9_code'].str[:3].isin(['410', '412'])) |
        (df['icd10_code'].str[:3].isin(['I21', 'I22'])) |
        (df['icd10_code'].str[:4] == 'I252'), 1, 0
    )
    
    # Congestive Heart Failure
    df['congestive_heart_failure'] = np.where(
        (df['icd9_code'].str[:3] == '428') |
        (df['icd9_code'].str[:5].isin(['39891', '40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493'])) |
        (df['icd9_code'].str[:4].between('4254', '4259')) |
        (df['icd10_code'].str[:3].isin(['I43', 'I50'])) |
        (df['icd10_code'].str[:4].isin(['I099', 'I110', 'I130', 'I132', 'I255', 'I420', 'I425', 'I426', 'I427', 'I428', 'I429', 'P290'])), 
        1, 0
    )
    
    # Peripheral Vascular Disease
    df['peripheral_vascular_disease'] = np.where(
        (df['icd9_code'].str[:3].isin(['440', '441'])) |
        (df['icd9_code'].str[:4].isin(['0930', '4373', '4471', '5571', '5579', 'V434'])) |
        (df['icd9_code'].str[:4].between('4431', '4439')) |
        (df['icd10_code'].str[:3].isin(['I70', 'I71'])) |
        (df['icd10_code'].str[:4].isin(['I731', 'I738', 'I739', 'I771', 'I790', 'I792', 'K551', 'K558', 'K559', 'Z958', 'Z959'])), 
        1, 0
    )
    
    # Cerebrovascular Disease
    df['cerebrovascular_disease'] = np.where(
        (df['icd9_code'].str[:3].isin(['430', '431', '432', '433', '434', '435', '436', '437', '438'])) |
        (df['icd10_code'].str[:3].isin(['I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69'])), 
        1, 0
    )
    
    # Dementia
    df['dementia'] = np.where(
        (df['icd9_code'].str[:3].isin(['290'])) |
        (df['icd10_code'].str[:3].isin(['F00', 'F01', 'F02', 'F03'])), 
        1, 0
    )
    
    # Chronic Pulmonary Disease
    df['chronic_pulmonary_disease'] = np.where(
        (df['icd9_code'].str[:3].isin(['490', '491', '492', '493', '494', '495', '496'])) |
        (df['icd10_code'].str[:3].isin(['J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J47', 'J60', 'J61', 'J62', 'J63', 'J64', 'J65', 'J66', 'J67'])), 
        1, 0
    )
    
    # Rheumatic Disease
    df['rheumatic_disease'] = np.where(
        (df['icd9_code'].str[:3].isin(['710', '714', '725'])) |
        (df['icd10_code'].str[:3].isin(['M05', 'M06', 'M32', 'M34', 'M35', 'M45'])), 
        1, 0
    )
    
    # Peptic Ulcer Disease
    df['peptic_ulcer_disease'] = np.where(
        (df['icd9_code'].str[:3].isin(['531', '532', '533', '534'])) |
        (df['icd10_code'].str[:3].isin(['K25', 'K26', 'K27', 'K28'])), 
        1, 0
    )
    
    # Mild Liver Disease
    df['mild_liver_disease'] = np.where(
        (df['icd9_code'].str[:3].isin(['570', '571'])) |
        (df['icd10_code'].str[:3].isin(['K70', 'K73', 'K74'])), 
        1, 0
    )
    
    # Diabetes without Complications
    df['diabetes_without_complications'] = np.where(
        (df['icd9_code'].str[:3].isin(['250'])) & (df['icd9_code'].str[3] == '0'),
        1, 0
    )
    
    # Diabetes with Complications
    df['diabetes_with_complications'] = np.where(
        (df['icd9_code'].str[:3].isin(['250'])) & (df['icd9_code'].str[3] != '0'),
        1, 0
    )
    
    # Paraplegia and Hemiplegia
    df['paraplegia_hemiplegia'] = np.where(
        (df['icd9_code'].str[:3].isin(['342', '344'])) |
        (df['icd10_code'].str[:3].isin(['G81', 'G82'])), 
        1, 0
    )
    
    # Renal Disease
    df['renal_disease'] = np.where(
        (df['icd9_code'].str[:3].isin(['582', '583', '585', '586', '588'])) |
        (df['icd10_code'].str[:3].isin(['N18', 'N19'])), 
        1, 0
    )
    
    # Cancer (Any malignancy, including lymphoma and leukemia)
    df['cancer'] = np.where(
        (df['icd9_code'].str[:3].isin(['140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '170', '171', '172', '173', '174', '175', '176', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199'])) |
        (df['icd10_code'].str[:3].isin(['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38', 'C39', 'C40', 'C41', 'C43', 'C44', 'C45', 'C46', 'C47', 'C48', 'C49', 'C50', 'C51', 'C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58', 'C60', 'C61', 'C62', 'C63', 'C64', 'C65', 'C66', 'C67', 'C68', 'C69', 'C70', 'C71', 'C72', 'C73', 'C74', 'C75', 'C76', 'C77', 'C78', 'C79', 'C80'])),
        1, 0
    )
    
    # Metastatic Solid Tumor
    df['metastatic_solid_tumor'] = np.where(
        (df['icd9_code'].str[:3].isin(['196', '197', '198', '199'])) |
        (df['icd10_code'].str[:3].isin(['C77', 'C78', 'C79', 'C80'])), 
        1, 0
    )
    
    # Severe Liver Disease
    df['severe_liver_disease'] = np.where(
        (df['icd9_code'].str[:3].isin(['572'])) |
        (df['icd10_code'].str[:3].isin(['K72', 'K76'])), 
        1, 0
    )
    
    # AIDS/HIV
    df['aids_hiv'] = np.where(
        (df['icd9_code'].str[:3].isin(['042'])) |
        (df['icd10_code'].str[:3].isin(['B20'])), 
        1, 0
    )
    
    return df

comorbidities = calculate_comorbidities(diagnoses_icd)

# Step 4: Aggregate the comorbidities for each patient (hadm_id)
comorbidities_agg = comorbidities.groupby('hadm_id').agg(
    myocardial_infarct=('myocardial_infarct', 'max'),
    congestive_heart_failure=('congestive_heart_failure', 'max'),
    peripheral_vascular_disease=('peripheral_vascular_disease', 'max'),
    cerebrovascular_disease=('cerebrovascular_disease', 'max'),
    dementia=('dementia', 'max'),
    chronic_pulmonary_disease=('chronic_pulmonary_disease', 'max'),
    rheumatic_disease=('rheumatic_disease', 'max'),
    peptic_ulcer_disease=('peptic_ulcer_disease', 'max'),
    mild_liver_disease=('mild_liver_disease', 'max'),
    diabetes_without_complications=('diabetes_without_complications', 'max'),
    diabetes_with_complications=('diabetes_with_complications', 'max'),
    paraplegia_hemiplegia=('paraplegia_hemiplegia', 'max'),
    renal_disease=('renal_disease', 'max'),
    cancer=('cancer', 'max'),
    metastatic_solid_tumor=('metastatic_solid_tumor', 'max'),
    severe_liver_disease=('severe_liver_disease', 'max'),
    aids_hiv=('aids_hiv', 'max')
).reset_index()

# Step 5: Age-based Charlson score
def calculate_age_score(age):
    if age <= 50:
        return 0
    elif age <= 60:
        return 1
    elif age <= 70:
        return 2
    elif age <= 80:
        return 3
    else:
        return 4

admissions['age_score'] = admissions['age'].apply(calculate_age_score)

# Step 6: Merge comorbidities with admissions and age score
merged_df = pd.merge(admissions, comorbidities_agg, on='hadm_id', how='left')
#merged_df = pd.merge(merged_df, age_df[['hadm_id', 'age_score']], on='hadm_id', how='left')

# Step 7: Calculate Charlson Comorbidity Index
merged_df['charlson_comorbidity_index'] = (
    merged_df['age_score'] +
    merged_df['myocardial_infarct'] + 
    merged_df['congestive_heart_failure'] +
    merged_df['peripheral_vascular_disease'] +
    merged_df['cerebrovascular_disease'] +
    merged_df['dementia'] +
    merged_df['chronic_pulmonary_disease'] +
    merged_df['rheumatic_disease'] +
    merged_df['peptic_ulcer_disease'] +
    merged_df['mild_liver_disease'] +
    merged_df['diabetes_without_complications'] +
    merged_df['diabetes_with_complications'] +
    merged_df['paraplegia_hemiplegia'] +
    merged_df['renal_disease'] +
    merged_df['cancer'] +
    merged_df['metastatic_solid_tumor'] * 6 +  # Weight for metastatic cancer
    merged_df['severe_liver_disease'] * 3 +   # Weight for severe liver disease
    merged_df['aids_hiv'] * 6                 # Weight for AIDS/HIV
)

# Final result
print(merged_df[['subject_id', 'hadm_id', 'age_score', 'charlson_comorbidity_index']].head())
