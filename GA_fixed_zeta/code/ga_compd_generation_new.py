from sklearn.linear_model import LinearRegression
import pandas as pd
import models
import random
import transform
from sklearn.ensemble import  ExtraTreesRegressor

# population_size = 100

# test_compound =pd.read_csv('test_generation_TiO2_Sasha_final.csv')
df = pd.read_csv('data/preprocessed/original_preprocessed.csv')
# df = pd.read_csv('data/preprocessed/original_preprocessed.csv')

X = df.drop([ 'Unnamed: 0', 'viability (%)'], axis=1)

# it might be better not to choose random generation from unique set, we can just choose from all data ( higher number of data have higher chance to pick, this will imporve the predictibility of the material as model predict best for those who have higher number of data)
''' generate dataframe with unique cell lines'''
uniq_cell_data = X.drop_duplicates('cell line')

# materiall = df.loc[df['material'] == 2]
# Y = materiall.drop([ 'Unnamed: 0', 'viability (%)'], axis=1)
"""uniq value datasets"""
uniq = [] # stores all the unique characters available in the dataset, it helps to make a new population with random parameters
for a in range(len(X.columns)):
  uni = pd.unique(X.iloc[:, a])
  uniq.append(uni)
# uniq[0]


"""create individual with values that are picked from the uniq array above"""

def individuals():
  indv = []
  for a in range(len(X.columns)):
    uniqas = random.choice(uniq[a])
    indv.append(uniqas)
  return indv
# individuals()

"""generate population with specific population size"""
#population with specific material descriptors were generated but cell line were still random
def population(size):
  pops = []
  for indv in range(2*size):
    single = individuals()
    pops.append(single)
  new_one = pd.DataFrame(data=pops, columns=X.columns)
  neww = new_one[(new_one['concentration (ug/ml)'] > 5) & (new_one['Hydrodynamic diameter (nm)']> 8)]
  new = neww.reset_index(drop=True)
  new = new.head(size)
  material_uniq = X.iloc[[random.randrange(0, len(X)) for _ in range(len(new))]]
  # material_uniq = X.loc[X['material']=='ZnO'] #to specify the name of the material
  material_uniq = material_uniq.reset_index(drop=True)
  # print(new.columns)
  new[['material','mcd', 'electronegativity', 'rox', 'radii',
       'Valance_electron', 'amw', 'lipinskiHBA', 'lipinskiHBD',
       'NumRotatableBonds', 'CrippenClogP', 'chi0v', 'chi1v', 'chi2v',
       'hallKierAlpha', 'kappa1']] =  material_uniq[['material','mcd', 'electronegativity', 'rox', 'radii',
       'Valance_electron', 'amw', 'lipinskiHBA', 'lipinskiHBD',
       'NumRotatableBonds', 'CrippenClogP', 'chi0v', 'chi1v', 'chi2v',
       'hallKierAlpha', 'kappa1']]
  return new

# dff = population(population_size)
# print(dff.columns)
# print(dff.loc[3])

"""change cell type into cancer and normal cell line"""
# cell = pd.read_csv('Data/cell_line/cell_decode.csv')
# canc =cell.loc[cell['Cell type'] =='SKOV-3', 'oe'].values.tolist()
# norm = cell.loc[cell['Cell type'] =='CHO-K1', 'oe'].values.tolist()

"""selecting SKOV-3 as cancer cell line and  CHO-K1 as normal cell line (both of them are ovary cell line """
popn = []
popc = []
#ovary normal- CHOK1: 14, canc - SKOV3: 61
# lung normal- BEAS 2B:9, canc- H1299:18, normal:HFL-1: 26
# liver normal - L02: 39, canc - HepG2: 33
def cell_lines(dat):
  #making same normal and cancer dataframe
  # single_norm = uniq_cell_data.loc[df['cell line'] == norm[0]]
  # single_canc = uniq_cell_data.loc[df['cell line'] == canc[0]]
  single_norm = uniq_cell_data.loc[df['cell line'] == 'SKOV-3'] #'L-02'] #'BEAS-2B'] #'CHO-K1'] # 'Hepatocytes'] #"'L-02']
  single_canc = uniq_cell_data.loc[df['cell line'] == 'CHO-K1'] #'A549'] #'SKOV-3'] #'HepG2']
  pop_norm =pd.concat([single_norm]*len(dat), ignore_index=True)
  pop_canc = pd.concat([single_canc] * len(dat), ignore_index=True)
  df_norm= dat.copy()
  df_canc = dat.copy()
  #replaced random data for cell descriptors with normal and cancer cell line
  df_norm[['cell line','organism','cell type', 'morphology', 'tissue', 'disease']] = pop_norm[['cell line','organism','cell type', 'morphology', 'tissue', 'disease']]
  df_canc[['cell line','organism','cell type', 'morphology', 'tissue', 'disease']] = pop_canc[['cell line','organism','cell type', 'morphology', 'tissue', 'disease']]
  # print('norm',df_norm.columns)
  return df_norm, df_canc

def zeta_potential(df):
  zeta = models.xtra_tree(df)
  df['Zeta potential (mV)'] = zeta
  return df

# print('here', zeta_potential(dff))



def fitness(df):
  norm, canc = cell_lines(df)
  nc = transform.transform(norm)
  cc = transform.transform(canc)
  nc_zeta = zeta_potential(nc)
  cc_zeta = zeta_potential(cc)
  # print('norm',norm.columns, canc)
  norm_viability = models.lgbm_predict(nc_zeta)
  canc_viability = models.lgbm_predict(cc_zeta)
  fitness = []
  norm_v = []
  canc_v = []
  for a in range(len(norm_viability)):
    n = norm_viability[a]
    c = canc_viability[a]
    fit = n - c
    fitnn = fit.tolist()
    norm_v.append(n)
    canc_v.append(c)
    fitness.append(fitnn)
  copy = norm.assign(norm_v=norm_v)
  copy2 = copy.assign(canc_v=canc_v)
  copy3 = copy2.assign(Fitness = fitness)
  copy3 = copy3.sort_values('Fitness', ascending=False)
  return copy3


# print(fitness(dff))

# def result_evaluation(df):
#   out = decoder.decode_transformed(df)
#   return out

# def test_df(df):
#   new = fitness(df)
#   return new
#
# testt = fitness(test_compound)
# testt.to_csv('Exp_test_fitness_TiO2.csv')
