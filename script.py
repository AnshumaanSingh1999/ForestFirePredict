import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
import math
import pickle



df = pd.read_csv('forestfires.csv') 
X=df[['temp','RH','wind','rain']]
y=df['area']
y=y.astype('int')
X=X.astype('int')
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
    
    
pickle.dump(log_reg, open('model.pkl','wb'))
#model = pickle.load(open('model.pkl','rb'))
    #print(model.predict_proba([[2, 9, 6]]))
    
#a=model.predict_proba(inp)
    #a=log_reg.predict_proba(inp)
#if(a[0][1]>=0.99):
    #print()
#else:
    #print()
    