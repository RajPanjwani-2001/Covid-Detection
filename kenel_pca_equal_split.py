import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet101V2, EfficientNetB0, EfficientNetB1, EfficientNetB2, \
    EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.applications import ResNet50V2, ResNet152V2, MobileNetV3Large, MobileNetV3Small, InceptionResNetV2, \
    Xception, NASNetLarge, DenseNet121, InceptionV3, VGG16, VGG19, NASNetMobile, DenseNet169, MobileNet

import pickle
import numpy as np
import pandas as pd
from config import config
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import *
from sklearn.preprocessing import MinMaxScaler
from Decompose import TransferLearningFeatureSelection
from Decompose import Decompose
if __name__ == "__main__":

    features = pd.read_pickle('ResNet50V2_selected_top_features.pickle')
    cls_labels = pd.read_pickle('Corona_Pne_normal_class_data.pickle')
    print(features.shape)

    covid = []
    covid_cls = []
    normal = []
    normal_cls = []
    pne = []
    pne_cls = []
    for i in range(cls_labels.shape[0]):
        if cls_labels[i] == 0:
            covid.append(features[i,:])
            covid_cls.append(0)

        if cls_labels[i] == 1:
            normal.append(features[i, :])
            normal_cls.append(1)

        if cls_labels[i] == 2:
            pne.append(features[i, :])
            pne_cls.append(2)

    covid = np.array(covid)
    covid_cls = np.array(covid_cls)

    normal = np.array(normal)
    normal_cls = np.array(normal_cls)

    pne = np.array(pne)
    pne_cls = np.array(pne_cls)

    print(covid.shape, normal.shape, pne.shape)

    n_folds = 10
    cv = KFold(n_splits=n_folds, shuffle=True)

    covid_train_index = []
    normal_train_index = []
    pne_train_index = []

    for train_index, test_index in cv.split(covid):
        covid_train_index.append([train_index, test_index])

    for train_index, test_index in cv.split(normal):
        normal_train_index.append([train_index, test_index])

    for train_index, test_index in cv.split(pne):
        pne_train_index.append([train_index, test_index])

    resultFile = 'equal_split_kpca.csv'
    obj = Decompose()
    for degree in range(2, 10):
        for n_comp in range(200, 600, 10):

            avg_acc = 0
            for i in range(n_folds):
                X_train, X_test = covid[covid_train_index[i][0], :], covid[covid_train_index[i][1], :]
                Y_train, Y_test = covid_cls[covid_train_index[i][0]], covid_cls[covid_train_index[i][1]]

                train, test = normal[normal_train_index[i][0], :], normal[normal_train_index[i][1], :]
                X_train = np.append(X_train, train, axis=0)
                X_test = np.append(X_test, test, axis=0)
                train, test = normal_cls[normal_train_index[i][0]], normal_cls[normal_train_index[i][1]]
                Y_train = np.append(Y_train, train, axis=0)
                Y_test = np.append(Y_test, test, axis=0)

                train, test = pne[pne_train_index[i][0], :], pne[pne_train_index[i][1], :]
                X_train = np.append(X_train, train, axis=0)
                X_test = np.append(X_test, test, axis=0)
                train, test = pne_cls[pne_train_index[i][0]], pne_cls[pne_train_index[i][1]]
                Y_train = np.append(Y_train, train, axis=0)
                Y_test = np.append(Y_test, test, axis=0)

                X_train = obj.kernel_pca(X_train, n_comp, 'poly', degree)
                X_test = obj.kernel_pca(X_test, n_comp, 'poly', degree)
                print('X_train: ', X_train.shape, 'X_test: ', X_test.shape)

                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                model.fit(X_train, Y_train)
                Y_pred = model.predict(X_test)
                p, r, f1, s = precision_recall_fscore_support(Y_test, Y_pred)
                acc = accuracy_score(Y_test, Y_pred)
                avg_acc += acc
                print(p, r, f1, s, acc)

            acc = avg_acc/n_folds
            res = str(degree) + "," + str(n_comp) + "," + str(acc) + "\n"
            fp = open(resultFile, "a")
            fp.write(res)
            fp.close()
            print("Average Accuracy : ", acc)

