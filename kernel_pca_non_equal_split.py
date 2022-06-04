import pandas as pd
from sklearn.feature_selection import RFE
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import *
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from Decompose import Decompose
x = pd.read_pickle('ResNet50V2_selected_top_features.pickle')
y = pd.read_pickle('Corona_Pne_normal_class_data.pickle')

print(x.shape)

obj = Decompose()

'''#part 1
for degree in range(2,5):
  for n_comp in range(100,201,10):
    #fo = open('/content/drive/MyDrive/gait_research/vgg_kernel_pca_poly.csv',mode='a')

    x = obj.kernel_pca(x,n_comp,'poly',degree)
    print(x.shape)
    model = xgb.XGBClassifier(use_label_encoder = False,eval_metric='logloss')
    cs = cross_val_score(model, x,y,cv=10).mean()
    print(degree,n_comp,cs)
    #fo.write(str(i) + ',' + str(cs)+ '\n')

    #fo.close()'''


#part 2
resultFile = 'non_equal_split_kpca.csv'
for degree in range(2,10):
  X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
  for n_comp in range(200, 600, 10):
    avg_acc = 0
    for i in range(10):
      X_train = obj.kernel_pca(X_train,n_comp,'poly',degree)
      X_test = obj.kernel_pca(X_test, n_comp, 'poly', degree)
      print('X_train: ',X_train.shape,'X_test: ',X_test.shape)
      model = xgb.XGBClassifier(use_label_encoder = False,eval_metric='logloss')
      model.fit(X_train,Y_train)
      Y_pred = model.predict(X_test)
      p, r, f1, s = precision_recall_fscore_support(Y_test, Y_pred)
      acc = accuracy_score(Y_test, Y_pred)
      print( p, r, f1, s, acc)
      print(degree,n_comp)
      avg_acc += acc
    acc = avg_acc / 10
    res = str(degree) + "," + str(n_comp) + "," + str(acc) + "\n"
    fp = open(resultFile, "a")
    fp.write(res)
    fp.close()
    print("Average Accuracy : ", acc)


