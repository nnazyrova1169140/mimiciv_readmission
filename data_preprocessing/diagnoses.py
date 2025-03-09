
# Define rule-based function for ICD-10
def map_icd10_to_group(icd_code):
    if icd_code.startswith(('A', 'B')):
        return "Infectious"
    elif icd_code.startswith(('C', 'D0', 'D1', 'D2', 'D3', 'D4')):
        return "Neoplasms"
    elif icd_code.startswith(('D5', 'D6', 'D7', 'D8')):
        return "Blood_Immune"
    elif icd_code.startswith(('E')):
        return "Endocrine"
    elif icd_code.startswith(('F')):
        return "Mental_Behavioural"
    elif icd_code.startswith(('G')):
        return "Nervous_System"
    elif icd_code.startswith(('H0', 'H1', 'H2', 'H3', 'H4', 'H5')):
        return "Eye_Adnexa"
    elif icd_code.startswith(('H6', 'H7', 'H8', 'H9')):
        return "Ear_Mastoid"
    elif icd_code.startswith(('I')):
        return "Circulatory"
    elif icd_code.startswith(('J')):
        return "Respiratory"
    elif icd_code.startswith(('K')):
        return "Digestive"
    elif icd_code.startswith(('L')):
        return "Skin_Subcutaneous"
    elif icd_code.startswith(('M')):
        return "Musculoskeletal"
    elif icd_code.startswith(('N')):
        return "Genitourinary"
    elif icd_code.startswith(('O')):
        return "Pregnancy_Childbirth"
    elif icd_code.startswith(('P')):
        return "Perinatal"
    elif icd_code.startswith(('Q')):
        return "Congenital"
    elif icd_code.startswith(('R')):
        return "Symptoms_Signs"
    elif icd_code.startswith(('S', 'T')):
        return "Injury_Poisoning"
    elif icd_code.startswith(('V', 'W', 'X', 'Y')):
        return "External_Causes"
    elif icd_code.startswith(('Z')):
        return "Factors_Influencing_Health"
    elif icd_code.startswith(('U0', 'U1', 'U2', 'U3', 'U4')):
        return "Uncertain"
    elif icd_code.startswith(('U8')):
        return "Resistance_to_drugs"
    return None

# Define rule-based function for ICD-9
def map_icd9_to_group(icd_code):
    try:
        if icd_code.startswith('E'):
            return "External_Causes"
        elif icd_code.startswith('V'):
            return "Factors_Influencing_Health"
        
        # Convert the first three digits to an integer for numeric codes
        icd_numeric = int(icd_code[:3])
        
    except ValueError:
        return None  # Return None if conversion fails

    # Map based on ICD-9 code ranges
    if 1 <= icd_numeric <= 139:
        return "Infectious"
    elif 140 <= icd_numeric <= 239:
        return "Neoplasms"
    elif 240 <= icd_numeric <= 279:
        return "Endocrine"
    elif 280 <= icd_numeric <= 289:
        return "Blood_Immune"
    elif 290 <= icd_numeric <= 319:
        return "Mental_Behavioural"
    elif 320 <= icd_numeric <= 359:
        return "Nervous_System"
    elif 360 <= icd_numeric <= 389:
        return "Eye_Adnexa"
    elif 390 <= icd_numeric <= 459:
        return "Circulatory"
    elif 460 <= icd_numeric <= 519:
        return "Respiratory"
    elif 520 <= icd_numeric <= 579:
        return "Digestive"
    elif 580 <= icd_numeric <= 629:
        return "Genitourinary"
    elif 630 <= icd_numeric <= 679:
        return "Pregnancy_Childbirth"
    elif 680 <= icd_numeric <= 709:
        return "Skin_Subcutaneous"
    elif 710 <= icd_numeric <= 739:
        return "Musculoskeletal"
    elif 740 <= icd_numeric <= 759:
        return "Congenital"
    elif 760 <= icd_numeric <= 779:
        return "Perinatal"
    elif 780 <= icd_numeric <= 799:
        return "Symptoms_Signs"
    elif 800 <= icd_numeric <= 999:
        return "Injury_Poisoning"
    return None  # Return None for any code that doesn't fit into these categories

# Function to map both ICD-9 and ICD-10 codes based on version
def map_icd_to_group(row):
    icd_code = row['icd_code']
    icd_version = row['icd_version']
    
    if icd_version == 9:
        return map_icd9_to_group(icd_code)
    elif icd_version == 10:
        return map_icd10_to_group(icd_code)
    return None

