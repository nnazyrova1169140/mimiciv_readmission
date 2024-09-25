import pandas as pd
import numpy as np

df = pd.read_csv('hosp\labevents.csv')

relevant_itemids = [
    51221, 51222, 51248, 51249, 51250, 51265, 
    51279, 51277, 52159, 51301
]

df_filtered = df[(df['itemid'].isin(relevant_itemids)) & (df['valuenum'] > 0)]

df_pivot = df_filtered.pivot_table(
    index=['specimen_id'],
    values='valuenum',
    columns='itemid',
    aggfunc='max'
)

df_pivot = df_pivot.reset_index()

df_max_columns = df_filtered.groupby('specimen_id').agg({
    'subject_id': 'max',
    'hadm_id': 'max',
    'charttime': 'max'
}).reset_index()

df_final = pd.merge(df_max_columns, df_pivot, on='specimen_id', how='left')

df_final.rename(columns={
    51221: 'hematocrit',
    51222: 'hemoglobin',
    51248: 'mch',
    51249: 'mchc',
    51250: 'mcv',
    51265: 'platelet',
    51279: 'rbc',
    51277: 'rdw',
    52159: 'rdwsd',
    51301: 'wbc'
}, inplace=True)

print(df_final.head())
