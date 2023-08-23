import pickle
import warnings
from sklearn.ensemble import  ExtraTreesRegressor

warnings.filterwarnings('ignore')

model_lgbm = 'Models/lgbm_model_final.pkl'
model_xtree = 'Models/model_xtra.pkl'
def model_lg():
    with open(model_lgbm, 'rb') as f:
        rfr = pickle.load(f)
        return rfr

lgbm = model_lg()
def lgbm_predict(input):
    # x_input = input.drop(['material'], axis=1)
    y_input_predict = lgbm.predict(input)
    return y_input_predict

def xtra_tree(df):
    with open(model_xtree, 'rb') as f:
        xt = pickle.load(f)
    x = df[['Hydrodynamic diameter (nm)', 'mcd',
       'electronegativity', 'rox', 'radii', 'Valance_electron', 'amw',
       'lipinskiHBA', 'lipinskiHBD', 'NumRotatableBonds', 'CrippenClogP',
       'chi0v', 'chi1v', 'chi2v', 'hallKierAlpha', 'kappa1']]

    y = xt.predict(x)
    return y
