# import ga_compd_generation_new
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import StandardScaler

original = pd.read_csv('C:/Users/user/Desktop/project/v2/preprocessed/original_preprocessed.csv')
additional = pd.read_csv('C:/Users/user/Desktop/project/v2/preprocessed/additional_preprocessed.csv')
all = pd.concat([original,additional], axis=0)
all = all.reset_index(drop=True)

# for transformation of data
cat_col = ['cell line', 'test', 'organism', 'cell type', 'morphology', 'tissue', 'disease', ]
num_col = ['time (hr)', 'concentration (ug/ml)','Hydrodynamic diameter (nm)', 'Zeta potential (mV)', 'mcd',
           'electronegativity', 'rox', 'radii', 'Valance_electron', 'amw',
           'lipinskiHBA', 'lipinskiHBD', 'NumRotatableBonds', 'CrippenClogP',
           'chi0v', 'chi1v', 'chi2v', 'hallKierAlpha', 'kappa1']

# standard scaler and binary encoding
def transform(data):
    be = ce.BinaryEncoder()
    Xc = be.fit_transform(all[cat_col])
    Xct = be.transform(data[cat_col])  # put anything you want to transform

    sc = StandardScaler()
    X_all = sc.fit_transform(all[num_col])
    X_ss = sc.transform(data[num_col])  # put anything you want to transform
    X_sc = pd.DataFrame(X_ss, columns=num_col)
    join = pd.concat([Xct, X_sc], axis=1)
    return join

print(transform(all))

# dfn, dfc = ga_compd_generation_new.cell_lines(ga_compd_generation_new.population(50))
# print('n',transform(dfn))