import pandas as pd
from sklearn.feature_selection import RFE
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import *
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from Decompose import Decompose
import pickle

obj = Decompose()
resultFile = 'cvs_kpca_increment.csv'

toPath = 'C:/Users/Raj/MyProg/gait_research/pickle1/'



for degree in range(2,3):
  x = pd.read_pickle('ResNet50V2_selected_top_features.pickle')
  y = pd.read_pickle('Corona_Pne_normal_class_data.pickle')
  print(x.shape)
  for n_comp in range(400,1301,10):
    x = obj.kernel_pca(x,n_comp,'poly',degree)

    print(x.shape)
    if n_comp%100 ==0:
        features = toPath + 'features_' + str(degree) + '_' + str(n_comp) + ".pkl"
        fo = open(features, "wb")
        pickle.dump(x, fo)
        fo.close()

    model = xgb.XGBClassifier(use_label_encoder = False,eval_metric='logloss')
    cs = cross_val_score(model, x,y,cv=10).mean()
    print(degree,n_comp,cs)
    res = str(degree) + "," + str(n_comp) + "," + str(cs) + "\n"
    fp = open(resultFile, "a")
    fp.write(res)
    fp.close()
