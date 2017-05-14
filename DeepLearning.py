import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn import preprocessing
#import h5py
#from sqlalchemy import create_engine
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
#from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#from sklearn.ensemble import AdaBoostRegressor
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
#from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def main():
	flights = pd.read_csv('BigData_Pruned_Top10.csv')
	print("Loaded in all data...")

	airlines = flights['AIRLINE'].tolist()

	le_airline = preprocessing.LabelEncoder()


	le_airline.fit(airlines)


	enc_airlines = le_airline.transform(airlines)

	flights['AIRLINE'] = enc_airlines

	features = [i for i in list(flights) if i != "ARRIVAL_DELAY"]

	X = flights[features]
	y = flights["ARRIVAL_DELAY"]
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
	#assert(len(X_train) != len(X_test))
	#assert(len(X_train) == len(y_train))
	#print("Split inputs into training and testing ...")
	trainErrors = []
	testErrors = []

	estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=1)
	
	
	kfold = KFold(n_splits=10, random_state=seed)
	results = cross_val_score(estimator, X, Y, cv=kfold, scoring = 'neg_mean_absolute_error')
	print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

	
	#print('Training Errors):', trainErrors)
	#print('Test Errors:', testErrors)
	#trainErrors = np.asarray(trainErrors)
	#testErrors = np.asarray(testErrors)
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