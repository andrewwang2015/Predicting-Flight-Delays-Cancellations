import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn import preprocessing
#import h5py
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#from sklearn.ensemble import AdaBoostRegressor
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

def main():
	flights = pd.read_csv('DelaysOnlyRegressorTop10.csv')
	print("Loaded in all data...")

	orig_airports = flights['ORIGIN_AIRPORT'].tolist()
	dest_airports = flights['DESTINATION_AIRPORT'].tolist()
	airlines = flights['AIRLINE'].tolist()

	le_airports = preprocessing.LabelEncoder()
	le_airline = preprocessing.LabelEncoder()

	le_airports.fit(orig_airports)
	le_airline.fit(airlines)

	enc_orig_airports = le_airports.transform(orig_airports)
	enc_dest_airports = le_airports.transform(dest_airports)
	enc_airlines = le_airline.transform(airlines)

	flights['ORIGIN_AIRPORT'] = enc_orig_airports
	flights['DESTINATION_AIRPORT'] = enc_dest_airports
	flights['AIRLINE'] = enc_airlines

	features = [i for i in list(flights) if i != "ARRIVAL_DELAY"]

	X = flights[features]
	y = flights["ARRIVAL_DELAY"]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
	assert(len(X_train) != len(X_test))
	assert(len(X_train) == len(y_train))
	print("Split inputs into training and testing ...")
	trainErrors = []
	testErrors = []
	
	for v in [1e-5, 1e-4, 1e-3]:
		reg = GradientBoostingRegressor(min_samples_split=v, n_estimators=150)
		reg.fit(X_train, y_train)
		y_pred_train = reg.predict(X_train)
		y_pred_test  = reg.predict(X_test)
		print((v, mean_squared_error(y_train, y_pred_train)), (v, mean_squared_error(y_test, y_pred_test)))
		trainErrors.append((v, mean_absolute_error(y_train, y_pred_train)))
		testErrors.append((v, mean_absolute_error(y_test, y_pred_test)))
		
	
	print('Training Errors):', trainErrors)
	print('Test Errors:', testErrors)
	trainErrors = np.asarray(trainErrors)
	testErrors = np.asarray(testErrors)
	#np.savetxt("TrainingMinSamples.txt", trainErrors)
	#np.savetxt("TestingMinSamples.txt", testErrors)
	print()
	#trainErrors = []
	#testErrors = []

	#for v in [1e-6, 5e-05, 1e-05, 5e-04]:
		#reg = GradientBoostingRegressor(min_samples_leaf=v, n_estimators=150)
		#reg.fit(X_train, y_train)
		#y_pred_train = reg.predict(X_train)
		#y_pred_test  = reg.predict(X_test)
		#print((v, mean_squared_error(y_train, y_pred_train)), (v, mean_squared_error(y_test, y_pred_test)))
		#trainErrors.append((v, mean_absolute_error(y_train, y_pred_train)))
		#testErrors.append((v, mean_absolute_error(y_test, y_pred_test)))
		
	
	#print('Training Errors):', trainErrors)
	#print('Test Errors:', testErrors)
	#trainErrors = np.asarray(trainErrors)
	#testErrors = np.asarray(testErrors)
	#np.savetxt("TrainingMinLeaf.txt", trainErrors)
	#np.savetxt("TestingMinLeaf.txt", testErrors)	
if __name__ == '__main__':
	main()