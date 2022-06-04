import pandas as pd
from sklearn.feature_selection import RFE
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import *
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from Decompose import Decompose
import pickle

obj = Decompose()

model_nm = 'ResNet50V2'
resultFile = model_nm + '_pickle10_sigmoid_results.csv'

#toPath = 'pickle1/'

#Actual_x = pd.read_pickle(model_nm + '_selected_top_features.pickle')
#Actual_x = pd.read_pickle(model_nm + '_features_sigmoid_'+ '800'+ '.pkl')
Actual_x = pd.read_pickle('ResNet50V2_Tuning_features__850.pkl')
y = pd.read_pickle('Corona_Pne_normal_class_data.pickle')

x = Actual_x
print(x.shape)

for i in range(5):
    model = RandomForestClassifier()
    cs = cross_val_score(model, x, y, cv=10).mean()
    print(cs)

'''for n_comp in range(870, 951, 10):
    #kernel pca approach
    x = obj.kernel_pca(x, n_comp, 'sigmoid')
    print('Kpca approach Features: ',x.shape)

    if n_comp%10 == 0:
        features = model_nm + '_Tuning_features_' + '_' + str(n_comp) + ".pkl"
        fo = open(features, "wb")
        pickle.dump(x, fo)
        fo.close()

    model = RandomForestClassifier()
    cs = cross_val_score(model, x,y,cv=10).mean()
    print(n_comp,cs)
    res = str(n_comp) + "," + str(cs) + "\n"
    fp = open(resultFile, "a")
    fp.write(res)
    fp.close()
'''
