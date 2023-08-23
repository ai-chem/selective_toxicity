import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import StandardScaler

original = pd.read_csv('data/original_preprocessed.csv')
additional = pd.read_csv('data/additional_preprocessed.csv')
all = pd.concat([original,additional], axis=0)
all = all.reset_index(drop=True)

# for transformation of data
cat_col = ['cell line', 'test', 'organism', 'cell type', 'morphology', 'tissue', 'disease', ]
num_col = ['time (hr)', 'concentration (ug/ml)','Hydrodynamic diameter (nm)', 'Zeta potential (mV)', 'mcd',
           'electronegativity', 'rox', 'radii', 'Valance_electron', 'amw',
           'lipinskiHBA', 'lipinskiHBD', 'NumRotatableBonds', 'CrippenClogP',
           'chi0v', 'chi1v', 'chi2v', 'hallKierAlpha', 'kappa1']

# standard scaler and binary encoding
def data_encode(data):
    be = ce.BinaryEncoder()
    Xc = be.fit_transform(all[cat_col])
    Xct = be.transform(data[cat_col])  # put anything you want to transform

    sc = StandardScaler()
    X_all = sc.fit_transform(all[num_col])
    X_ss = sc.transform(data[num_col])  # put anything you want to transform
    X_sc = pd.DataFrame(X_ss, columns=num_col)
    join = pd.concat([Xct, X_sc], axis=1)
    return join

# print(data_encode(additional))
