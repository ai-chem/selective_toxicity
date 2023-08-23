import pickle
import warnings
warnings.filterwarnings('ignore')

model_lgbm = 'Models/lgbm_model_final.pkl'

def model_lg():
    with open(model_lgbm, 'rb') as f:
        rfr = pickle.load(f)
        return rfr

lgbm = model_lg()
def lgbm_predict(input):
    # x_input = input.drop(['material'], axis=1)
    y_input_predict = lgbm.predict(input)
    return y_input_predict