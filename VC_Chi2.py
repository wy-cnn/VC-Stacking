# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:41:40 2019

@author: hengyue_yu
"""

from sklearn import neighbors, preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from scipy.io import loadmat
import xgboost as xgb
import pandas as pd
import lightgbm as lgb
from sklearn import svm,metrics
from sklearn.metrics import accuracy_score  


VC_data_train = loadmat("VC_data.mat")
virus_train = VC_data_train["VC_data"]
list1 = [i for i in range(1,238)]
virus_train[np.isnan(virus_train)] = 0
X = virus_train[:,list1]
y = virus_train[:,0]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.25,random_state=3)


scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



#=======================RF===========================

model_RF=RandomForestClassifier(bootstrap=True, 
                                class_weight=None, 
                                criterion='gini',
                                max_depth=4, 
                                max_features='auto', 
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                min_impurity_split=None,
                                min_samples_leaf=1, 
                                min_samples_split=2,
                                min_weight_fraction_leaf=0.0, 
                                n_estimators=18, 
                                n_jobs=-1,
                                oob_score=False, 
                                random_state=1, 
                                verbose=0,
                                warm_start=False)


model_RF.fit(X_train,y_train)


y_RF_pred = model_RF.predict(X_test)
y_RF_prob = model_RF.predict_proba(X_test)


print('The RF accuracy is:',accuracy_score(y_test,y_RF_pred))
print('The RF precision is:',metrics.precision_score(y_test,y_RF_pred,average='macro'))
print('The RF recall is::',metrics.recall_score(y_test,y_RF_pred,average='macro'))
print('The RF f1 score is:',metrics.f1_score(y_test,y_RF_pred,average='macro'))

#=======================RF END=========================



#==========================GBDT==============================


model_gbdt =  GradientBoostingClassifier(criterion='mse', 
                                         init=None,
                                         learning_rate=0.2, 
                                         loss='deviance', 
                                         max_depth=3,
                                         max_features='auto', 
                                         max_leaf_nodes=None,
                                         min_impurity_decrease=0.0, 
                                         min_impurity_split=None,
                                         min_samples_leaf=1, 
                                         min_samples_split=2,
                                         min_weight_fraction_leaf=0.0, 
                                         n_estimators=50, 
                                         presort='auto',
                                         random_state=1, 
                                         subsample=0.8, 
                                         verbose=0, 
                                         warm_start=False)


model_gbdt.fit(X_train,y_train)


y_GBDT_pred = model_gbdt.predict(X_test)
y_GBDT_prob = model_gbdt.predict_proba(X_test)


print('The GBDT accuracy is:',accuracy_score(y_test,y_GBDT_pred))
print('The GBDT precision is:',metrics.precision_score(y_test,y_GBDT_pred,average='macro'))
print('The GBDT recall is::',metrics.recall_score(y_test,y_GBDT_pred,average='macro'))
print('The GBDT f1 score is:',metrics.f1_score(y_test,y_GBDT_pred,average='macro'))


#========================GBDT END=========================



#========================XGBoost=========================


xg_train = xgb.DMatrix(X_train, y_train)
xg_test = xgb.DMatrix(X_test, y_test)


param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.1
param['max_depth'] = 7
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 7
watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 50


model_xgb = xgb.train(param, xg_train, num_round, watchlist )


y_xgb_prob = model_xgb.predict(xg_test)
y_xgb_pred = np.argmax(y_xgb_prob, axis=1)


print('The Xgboost accuracy is:',accuracy_score(y_test,y_xgb_pred))
print('The Xgboost precision is:',metrics.precision_score(y_test,y_xgb_pred,average='macro'))
print('The Xgboost recall is::',metrics.recall_score(y_test,y_xgb_pred,average='macro'))
print('The Xgboost f1 score is:',metrics.f1_score(y_test,y_xgb_pred,average='macro'))
#========================XGBoost END=========================



#========================Adaboost=========================


model_Adb = AdaBoostClassifier(
                                algorithm='SAMME',#或SAMME.R
                                base_estimator=None,
                                learning_rate=1.0,#0--1
                                n_estimators=45,#0--100
                                random_state=None,
                                )


model_Adb.fit(X_train,y_train)


y_Adb_pred = model_Adb.predict(X_test)
y_Adb_prob = model_Adb.predict_proba(X_test)


print('The Adaboost accuracy is:',accuracy_score(y_test,y_Adb_pred))
print('The Adaboost precision is:',metrics.precision_score(y_test,y_Adb_pred,average='macro'))
print('The Adaboost recall is::',metrics.recall_score(y_test,y_Adb_pred,average='macro'))
print('The Adaboost f1 score is:',metrics.f1_score(y_test,y_Adb_pred,average='macro'))


#========================Adaboost END=========================



#========================KNN=============================

model_knn = neighbors.KNeighborsClassifier()

model_knn = neighbors.KNeighborsClassifier(
                                           algorithm='auto', 
                                           leaf_size=30, 
                                           metric='minkowski',
                                           n_neighbors=3,
                                           p=2,
                                           weights='distance')
model_knn.fit(X_train,y_train)

y_KNN_pred = model_knn.predict(X_test)
y_KNN_prob = model_knn.predict_proba(X_test)
print('KNN算法完成')
#========================KNN END=========================

#========================LightGBM=========================

params = {
                        'boosting_type': 'gbdt',  
                        'objective': 'multiclass',  
                        'num_class': 7,  
                        'metric': 'multi_error',  
                        'num_leaves': 120,  
                        'min_data_in_leaf': 100,  
                        'learning_rate': 0.06,  
                        'feature_fraction': 0.8,  
                        'bagging_fraction': 0.8,  
                        'bagging_freq': 5,  
                        'lambda_l1': 0.4,  
                        'lambda_l2': 0.5,  
                        'min_gain_to_split': 0.2,  
                        'verbose': -1,
	}

trn_data = lgb.Dataset(X_train, y_train)
val_data = lgb.Dataset(X_test, y_test)
model_Lgb = lgb.train( params, 
                trn_data, 
                num_boost_round = 1000,
                valid_sets = [val_data], 
                verbose_eval = 100, 
                early_stopping_rounds = 100
)



y_Lgb_prob = model_Lgb.predict(X_test,num_iteration=model_Lgb.best_iteration)
y_Lgb_pred=[list(x).index(max(x)) for x in y_Lgb_prob]

print('The LightGBM accuracy is:',accuracy_score(y_test,y_Lgb_pred))
print('The LightGBM precision is:',metrics.precision_score(y_test,y_Lgb_pred,average='macro'))
print('The LightGBM recall is::',metrics.recall_score(y_test,y_Lgb_pred,average='macro'))
print('The LightGBM f1 score is:',metrics.f1_score(y_test,y_Lgb_pred,average='macro'))

#========================LightGBM END=========================


#========================SVM=========================
model_SVM=svm.SVC(C=1.0, 
                  cache_size=200, 
                  class_weight=None, 
                  coef0=0.0,
                  decision_function_shape='ovo', 
                  degree=3, 
                  gamma='auto', 
                  kernel='rbf',
                  max_iter=-1, 
                  probability=False, 
                  random_state=None, 
                  shrinking=True,
                  tol=0.001, 
                  verbose=False)

model_SVM.fit(X_train,y_train)


y_SVM_pred = model_SVM.predict(X_test)

print('The SVM accuracy is:',accuracy_score(y_test,y_SVM_pred))
print('The SVM precision is:',metrics.precision_score(y_test,y_SVM_pred,average='macro'))
print('The SVM recall is::',metrics.recall_score(y_test,y_SVM_pred,average='macro'))
print('The SVM f1 score is:',metrics.f1_score(y_test,y_SVM_pred,average='macro'))

#========================SVM END=========================


Pdf = pd.DataFrame({'RF':y_RF_pred,'GBDT':y_GBDT_pred,'Xgboost':y_xgb_pred,'Adboost':y_Adb_pred,'LightGBM':y_Lgb_pred,'SVM':y_SVM_pred})
P_spearman = Pdf.corr('spearman')

P_SVM = [1,0.9210,0.9800,0.9835,0.8389,0.9873]
P_RF = [0.9210,1,0.9253,0.9272,0.8309,0.9259]
P_GBDT = [0.9800,0.9253,1,0.9871,0.8452,0.9880]
P_Xgb = [0.9835,0.9272,0.9871,1,0.8432,0.9928]
P_Adb = [0.8389,0.8309,0.8452,0.8432,1,0.8434]
P_lgb = [0.9873,0.9259,0.9880,0.9928,0.8434,1]
P_spearman = pd.DataFrame({'SVM':P_SVM,'RF':P_RF,'GBDT':P_GBDT,'XGBoost':P_Xgb,'Adaboost':P_Adb,'LightGBM':P_lgb})
P_Value = np.vstack((P_SVM,P_RF,P_GBDT,P_Xgb,P_Adb,P_lgb))
P_spearman = pd.DataFrame(P_Value, columns = ['SVM','RF','GBDT','XGBoost','Adaboost','LightGBM'],index = ['SVM','RF','GBDT','XGBoost','Adaboost','LightGBM'])









