import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset_df = pd.read_csv('C:/users/DELL/Desktop/mj/dataset.csv')
dataset_df.head()
dataset_df.size
dataset_df.shape
dataset_df.count()
dataset_df['Connectivity'].value_counts()
f4g_df = dataset_df[dataset_df['Connectivity']==4]
t3g_df = dataset_df[dataset_df['Connectivity']==3]
t2g_df = dataset_df[dataset_df['Connectivity']==2]
axes=f4g_df.plot(kind='scatter',x='DL (in Mbps)',y='UL (in Mbps)',color='red',label='4g')
t3g_df.plot(kind='scatter',x='DL (in Mbps)',y='UL (in Mbps)',color='blue',label='3g',ax=axes)
t2g_df.plot(kind='scatter',x='DL (in Mbps)',y='UL (in Mbps)',color='black',label='2g',ax=axes)
dataset_df = dataset_df[pd.to_numeric(dataset_df['UL (in Mbps)'],errors='coerce').notnull()]
dataset_df['UL (in Mbps)']=dataset_df['UL (in Mbps)'].astype('float')
dataset_df.dtypes
dataset_df.columns
feature_df = dataset_df[['PING (in ms)', 'JITTER (in ms)', 'DL (in Mbps)',
       'UL (in Mbps)']]
X=np.asarray(feature_df)
y=np.asarray(dataset_df['Connectivity'])
y.shape
y[0:5]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
y_test.shape
X_train.shape
from sklearn import svm
classifier=svm.SVC(kernel = 'linear',gamma='auto',C=2)
classifier.fit(X_train,y_train)
y_predict = classifier.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_predict))
import pickle
with open('model.pkl','wb') as f:
    pickle.dump(classifier, f)
with open('model.pkl','rb') as f:
    classifier_loaded = pickle.load(f)