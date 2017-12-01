import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_recall_fscore_support

flights = pd.read_csv('all_features_boolean.csv')
orig_airports = flights['ORIGIN_AIRPORT'].tolist()
dest_airports = flights['DESTINATION_AIRPORT'].tolist()
manufacts = flights['Manu'].tolist()
models = flights['Model'].tolist()
arr_sky = flights['arr_sky'].tolist()
dept_sky = flights['dept_sky'].tolist()
airlines = flights['AIRLINE'].tolist()

le_airports = preprocessing.LabelEncoder()
le_manu = preprocessing.LabelEncoder()
le_model = preprocessing.LabelEncoder()
le_sky = preprocessing.LabelEncoder()
le_airline = preprocessing.LabelEncoder()

le_airports.fit(orig_airports)
le_manu.fit(manufacts)
le_model.fit(models)
le_sky.fit(arr_sky)
le_airline.fit(airlines)

enc_orig_airports = le_airports.transform(orig_airports)
enc_dest_airports = le_airports.transform(dest_airports)
enc_manufacts = le_manu.transform(manufacts)
enc_models = le_model.transform(models)
enc_arr_sky = le_sky.transform(arr_sky)
enc_dept_sky = le_sky.transform(dept_sky)
enc_airlines = le_airline.transform(airlines)

flights['ORIGIN_AIRPORT'] = enc_orig_airports
flights['DESTINATION_AIRPORT'] = enc_dest_airports
flights['Model'] = enc_models
flights['Manu'] = enc_manufacts
flights['dept_sky'] = enc_dept_sky
flights['arr_sky'] = enc_arr_sky
flights['AIRLINE'] = enc_airlines
features = [x for x in list(flights) if x != 'ARRIVAL_DELAY' and x != 'BOOL_DELAY']
X = flights[features]
y = flights['BOOL_DELAY']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
M = 10
reg = RandomForestClassifier(20, max_depth=M)
reg.fit(X_train, y_train)
y_pred_test  = reg.predict(X_test)
precision, recall, x, y = precision_recall_fscore_support(y_test, y_pred_test, average='binary')
print('Max Depth:', M)
print('Random Forests Classifier\n---')
print('Precision:', precision)
print('Recall:', recall)
print('Score:', reg.score(X_test, y_test))
print(reg.feature_importances_)