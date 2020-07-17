# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:17:25 2019

@author: hengyue_yu
"""


from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import time 
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from scipy.io import loadmat
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm,metrics

starttime=time.time()


VC_data_train = loadmat("train.mat")
virus_train = VC_data_train["train"]
list1 = [i for i in range(1,238)]
virus_train[np.isnan(virus_train)] = 0
X_train = virus_train[:,list1]
y_train = virus_train[:,0]

VC_data_test = loadmat("test.mat")
virus_test = VC_data_test["test"]
list1 = [i for i in range(1,238)]
virus_test[np.isnan(virus_test)] = 0
X_test = virus_test[:,list1]
y_test = virus_test[:,0]


scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test) 



#=======================RF===========================

model_RF=RandomForestClassifier(bootstrap=True, 
                                class_weight=None, 
                                criterion='entropy',
                                max_depth=3, 
                                max_features='log2', 
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                min_impurity_split=None,
                                min_samples_leaf=1, 
                                min_samples_split=2,
                                min_weight_fraction_leaf=0.0, 
                                n_estimators=12, 
                                n_jobs=-1,
                                oob_score=False, 
                                random_state=1, 
                                verbose=0,
                                warm_start=False)


model_RF.fit(X_train,y_train)


y_RF_pred = model_RF.predict(X_test)
y_RF_prob = model_RF.predict_proba(X_test)


accuracy_score(y_test,y_RF_pred)
print('The RF accuracy is:',accuracy_score(y_test,y_RF_pred))
print('The RF precision is:',metrics.precision_score(y_test,y_RF_pred,average='macro'))
print('The RF recall is::',metrics.recall_score(y_test,y_RF_pred,average='macro'))
print('The RF f1 score is:',metrics.f1_score(y_test,y_RF_pred,average='macro'))



print('The RF confusion matrix is:\n',confusion_matrix(y_test,y_RF_pred,labels=[1,2,3,4,5,6]))
print('The RF precision, recall, f1 score are:\n',classification_report(y_test,y_RF_pred))

#=======================RF END=========================



#==========================SVM==============================


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


accuracy_score(y_test,y_SVM_pred)
print('The SVM accuracy is:',accuracy_score(y_test,y_SVM_pred))
print('The SVM precision is:',metrics.precision_score(y_test,y_SVM_pred,average='macro'))
print('The SVM recall is::',metrics.recall_score(y_test,y_SVM_pred,average='macro'))
print('The SVM f1 score is:',metrics.f1_score(y_test,y_SVM_pred,average='macro'))



print('The SVM confusion matrix is:\n',confusion_matrix(y_test,y_SVM_pred,labels=[1,2,3,4,5,6]))
print('The SVM precision, recall, f1 score are:\n',classification_report(y_test,y_SVM_pred))
#========================SVM END=========================



#========================Adaboost=========================


model_Adb = AdaBoostClassifier(
                                algorithm='SAMME',
                                base_estimator=None,
                                learning_rate=1.0,
                                n_estimators=45,
                                random_state=None,
                                )


model_Adb.fit(X_train,y_train)


y_Adb_pred = model_Adb.predict(X_test)
y_Adb_prob = model_Adb.predict_proba(X_test)


Adb_accuracy = accuracy_score(y_test,y_Adb_pred)  
print('The Adb accuracy is:',Adb_accuracy)
print('The Adb precision is:',metrics.precision_score(y_test,y_Adb_pred,average='macro'))
print('The Adb recall is::',metrics.recall_score(y_test,y_Adb_pred,average='macro'))
print('The Adb f1 score is:',metrics.f1_score(y_test,y_Adb_pred,average='macro'))



print('The Adb confusion matrix is:\n',confusion_matrix(y_test,y_Adb_pred,labels=[1,2,3,4,5,6]))
print('The Adb precision, recall, f1 score are:\n',classification_report(y_test,y_Adb_pred))


#========================Adaboost END=========================



#========================Stacking=========================
model_log = LogisticRegression(penalty='l2',
                         C=10,
                         multi_class='multinomial',
                         class_weight='balanced',
                         solver='newton-cg',
                         )

model_sta = StackingClassifier(
        classifiers=[model_RF,model_SVM, model_Adb], 
        meta_classifier=model_log,
        )

model_sta.fit(X_train,y_train)


y_sta_pred = model_sta.predict(X_test)


accuracy_score(y_test,y_sta_pred)
print('The sta accuracy is:',accuracy_score(y_test,y_sta_pred))
print('The sta precision is:',metrics.precision_score(y_test,y_sta_pred,average='macro'))
print('The sta recall is::',metrics.recall_score(y_test,y_sta_pred,average='macro'))
print('The sta f1 score is:',metrics.f1_score(y_test,y_sta_pred,average='macro'))



print('The sta confusion matrix is:\n',confusion_matrix(y_test,y_sta_pred,labels=[1,2,3,4,5,6]))
print('The sta precision, recall, f1 score are:\n',classification_report(y_test,y_sta_pred))



#========================Stacking END=========================




print('=============================================================\n')
print('The RF accuracy is:',accuracy_score(y_test,y_RF_pred))
print('The SVM accuracy is:',accuracy_score(y_test,y_SVM_pred))
print('The Adb accuracy is:',accuracy_score(y_test,y_Adb_pred))
print('The Stacking accuracy is:',accuracy_score(y_test,y_sta_pred))






