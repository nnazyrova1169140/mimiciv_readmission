import pandas as pd

# Load the dataset
df_diag = pd.read_csv('hosp/diagnoses_icd.csv')
df_diag = df_diag[df_diag['hadm_id'].isin(df_elderly['hadm_id'])]
# Convert 'icd_code' to string for consistent processing
df_diag['icd_code'] = df_diag['icd_code'].astype(str)

# Define detailed groupings for ICD-9 and ICD-10 codes related to SDOH
sdoh_categories = {
    "Homelessness": ["Z59", "Z590", "Z591", "Z592", "Z593", "Z594", "V600", "V601", "V609"],  # Homelessness, inadequate housing
    "Tobacco_Use": ["Z720", "3051", "V1582", "F17", "Z87891"],  # Tobacco use and nicotine dependence
    "Substance_Use": ["305", "304", "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F19"],  # Alcohol, opioids, and other substances
    "Violence_Abuse": ["T740", "T741", "T742", "T743", "T744", "T745", "T76", "Y09", "Z622", "V7181"],  # Physical, sexual, emotional abuse
    "Mental_Health": ["296", "300", "311", "F32", "F33", "F34", "F41", "F43.1", "F43.8"],  # Depression, anxiety, PTSD
    "Nutrition_Obesity": ["E66", "E40", "E46", "E643", "278", "V653"],  # Malnutrition, obesity, dietary issues
    "Economic_Struggles": ["Z595", "Z596", "Z597", "Z598", "Z599", "Z560", "Z561", "Z563"],  # Low income, unemployment
    "Family_Social_Support": ["Z630", "Z631", "Z632", "Z633", "Z634", "Z635", "Z636", "Z637", "Z638", "V6110", "V6111", "V6120"],  # Family and social discord
    "Need_for_Assistance": ["Z740", "Z741", "Z742", "Z743", "Z748", "Z748", "R531", "R532", "Z9981", "Z9989", "V4984", "V5881", "V602", "V667", "V6089", "V609"],  # Problems related to dependency
    "Life_Related_Problems": ["Z730", "Z731", "Z732", "Z733", "Z734", "Z735", "Z736", "Z738", "Z739", "V620", "V621", "V622", "V690"],  # Burnout, stress, lack of relaxation
    #"Legal_Issues": ["Z650", "Z651", "Z652", "Z653", "V625"],  # Imprisonment, legal challenges
    #"Education": ["Z550", "Z551", "Z552", "Z553", "Z554", "V623"],  # Literacy, underachievement, school problems
    #"Employment_Problems": ["Z560", "Z561", "Z562", "Z563", "Z564", "Z565", "Z566", "Z567", "Z579"],  # Job loss, occupational hazards
    "Physical_Activities": ["Y930", "Y931", "Y932", "Y933", "Y934", "Y935", "Y936", "Y937", "Y938", "Y939"],  # Activities: sports, yoga, running, etc.
    "Frailty": ["E40", "E41", "E42", "E43", "E44", "E45", "E46", "R54", "E643", "R54",
               "R531", "R532", "R634", "R636", "R627", "Z741", "Z742", "Z743",
               "M6281", "M6284", "260", "261", "262", "263", "7994", "7832", "V4984", "V667"],  # Sarcopenia, malnutrition, frailty
    "Dependency_on_Assistive_Devices": ["Z99"],  # Wheelchair dependency
    "Polypharmacy": ["Z9112"],  # ≥5 medications or medication burden
    "High_Risk_Medications": ["T36", "T37", "T38", "T39", "T40", "T41", "T42", "960", "961", "962", "963", "965", "967", "968"],  # Anticoagulants, opioids, psychotropics
    "Nonadherence_to_Medication": ["Z911"],  # Patient intentional underdosing
}

# Create boolean columns for each SDOH category
for category, codes in sdoh_categories.items():
    df_diag[category] = df_diag['icd_code'].apply(
        lambda code: any(code.startswith(c) for c in codes)
    )

# Display a sample of the updated dataframe
print(df_diag.head())
