import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

#combined data file
original_data = pd.read_csv('data/new_combined.csv')
descriptor = pd.read_csv('data/material_descriptors.csv')

original_data
original_data=original_data.drop(['index'], axis=1)
descriptor = descriptor.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

"""combine with material descriptor"""
new_one = pd.merge(original_data, descriptor, on='material' , how='inner')

"""combine with cell descriptor"""
cell_info = pd.read_csv('data/cell_line_descriptors.csv')
new =new_one.rename(columns={'Cell type':'cell line'})
cell = cell_info.drop([ 'BSL'], axis = 1)
# combined = pd.merge(new, cell, on = 'cell line', how='inner')

def descriptor_addition(data):
    new_one = pd.merge(data, descriptor, on='material', how='inner')
    new = new_one.rename(columns={'Cell type': 'cell line'})
    combined = pd.merge(new, cell, on='cell line', how='inner')
    combined.to_csv('combined_all_descriptors.csv')
    return combined

df_with_descriptors = descriptor_addition(original_data)

""" remove outliers
removing 1% of the data from the end where data are widely spread
details of the process and visualization is present is outliers_removal_visualization.py file"""

def outlier_remove(data_with_descriptor):
    df = df_with_descriptors.drop(['CID', 'Canonical_smiles'], axis=1)
    df['Valance_electron'] = df['Valance_electron'].astype(float)
    df['time (hr)'] = df['time (hr)'].astype(float)
    df2 = df[df['concentration (ug/ml)'] < 1001]
    df3 = df2[df2['viability (%)'] < 126.185]
    df4 = df3[df3['Hydrodynamic diameter (nm)'] < 600]
    df5 = df4[df4['Zeta potential (mV)'] > -63.5]
    return df5
# print(outlier_remove(descriptor_addition(original_data)))


def variance_threshold(df,th):
    var_thres=VarianceThreshold(threshold=th)
    var_thres.fit(df)
    new_cols = var_thres.get_support()
    return df.iloc[:,new_cols]

def corr(df, val):
  corr_matrix = df.corr().abs()
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
  to_drop = [column for column in upper.columns if any(upper[column] > val)]
  return df.drop(to_drop, axis=1, inplace=True)

def remove_correlation(data_with_outlier_removed):
    df5 = data_with_outlier_removed
    dff5 = df5.select_dtypes(include=['float64'])
    dff5o = df5.select_dtypes(include=['object'])
    df6 = variance_threshold(dff5, 0)
    corr(df6, 0.90)
    df6_all = pd.merge(dff5o, df6, left_index=True, right_index=True)
    return df6_all

preprocessed = remove_correlation(outlier_remove(descriptor_addition(original_data)))
print(preprocessed)
preprocessed.to_csv('data/preprocessed_all.csv')
