import pandas as pd
import models
import random
import encoder_transformation

population_size = 100
df = pd.read_csv('data/original_preprocessed.csv')
X = df.drop([ 'Unnamed: 0', 'viability (%)'], axis=1)

''' generate dataframe with unique cell lines'''
uniq_cell_data = X.drop_duplicates('cell line')

"""uniq value datasets"""
uniq = [] # stores all the unique characters available in the dataset, it helps to make a new population with randomized parameters
for a in range(len(X.columns)):
  uni = pd.unique(X.iloc[:, a])
  uniq.append(uni)
# print(uniq[0])


"""create individual with values that are picked from the uniq array above"""

def individuals():
  indv = []
  for a in range(len(X.columns)):
    uniqas = random.choice(uniq[a])
    indv.append(uniqas)
  return indv
# print(individuals())

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
  material_uniq = X.loc[X['material']=='Fe3O4'] #to specify the name of the material
  # material_uniq = X.iloc[[random.randrange(0, len(X)) for _ in range(len(new))]]
  material_uniq = material_uniq.reset_index(drop=True)
  new[['material','mcd', 'electronegativity', 'rox', 'radii',
       'Valance_electron', 'amw', 'lipinskiHBA', 'lipinskiHBD',
       'NumRotatableBonds', 'CrippenClogP', 'chi0v', 'chi1v', 'chi2v',
       'hallKierAlpha', 'kappa1']] =  material_uniq[['material','mcd', 'electronegativity', 'rox', 'radii',
       'Valance_electron', 'amw', 'lipinskiHBA', 'lipinskiHBD',
       'NumRotatableBonds', 'CrippenClogP', 'chi0v', 'chi1v', 'chi2v',
       'hallKierAlpha', 'kappa1']]
  return new
# print(population(population_size))

"""selecting HepG2 as cancer cell line and  Hepatocytes as normal cell line (both of them are liver cell line """
def cell_lines(dat):
  single_norm = uniq_cell_data.loc[df['cell line'] == 'SKOV-3'] #'BEAS-2B'] #'CHO-K1'] # 'Hepatocytes'] #"'L-02'] # cho-k1 cell line
  single_canc = uniq_cell_data.loc[df['cell line'] == 'CHO-K1'] #'A549'] #'SKOV-3'] #'HepG2'] # skov-3 cell line
  pop_norm =pd.concat([single_norm]*len(dat), ignore_index=True)
  pop_canc = pd.concat([single_canc] * len(dat), ignore_index=True)
  df_norm= dat.copy()
  df_canc = dat.copy()
  #replaced random data for cell descriptors with normal and cancer cell line
  df_norm[['cell line','organism','cell type', 'morphology', 'tissue', 'disease']] = pop_norm[['cell line','organism','cell type', 'morphology', 'tissue', 'disease']]
  df_canc[['cell line','organism','cell type', 'morphology', 'tissue', 'disease']] = pop_canc[['cell line','organism','cell type', 'morphology', 'tissue', 'disease']]
  return df_norm, df_canc



def fitness(df):
  norm, canc = cell_lines(df)
  nc = encoder_transformation.data_encode(norm)
  cc = encoder_transformation.data_encode(canc)
  norm_viability = models.lgbm_predict(nc)
  canc_viability = models.lgbm_predict(cc)
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

# print(fitness(population(population_size)))


def test_df(df):
  new = fitness(df)
  return new


# test_compound =pd.read_csv('test_generation_zno.csv')
# testt = fitness(test_compound)
# testt.to_csv('test_fitness_zno.csv')
