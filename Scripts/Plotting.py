import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import scipy
from sklearn.tree import DecisionTreeClassifier

def pltGraphs(title, xRange, trainingError, testingError, labelX):

    fig = plt.figure()
    plt.title(title, fontsize = 22)    
    plt.plot(xRange, trainingError, xRange, testingError, marker = '.', linewidth = 2)
    plt.legend(('Training', 'Testing'), loc = 'best', fontsize = 14)
    plt.xlabel(labelX, fontsize = 18)
    plt.ylabel('Mean Absolute Error', fontsize = 18)
    plt.margins(y=0.02)
    fig.savefig(title+".png", bbox_inches='tight')

# DecisionTreeRegressor, min_samples_split
title = 'MAE vs. min_samples_split'
labelX = 'min_samples_split'
x = [1e-5, 0.0001, 0.001, 0.01, 0.1, 0.15]
Training= [16.669, 19.31,19.915, 20.244, 20.632, 20.7175]
Testing=[21.7026, 19.94398, 19.928, 20.1965, 20.589, 20.67878]
assert(len(x) == len(Training))
assert(len(x) == len(Testing))
pltGraphs(title, x, Training, Testing, labelX)

# DecisionTreeRegressor, min_samples_leaf
title = 'MAE vs. min_samples_leaf' 
labelX = 'min_samples_leaf'
x = [1e-7, 1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 0.2]
Training = [3.66e-05, 12.444, 18.44, 19.601, 20.059, 20.4011,  20.8401, 20.952]
Testing =  [27.37, 23.600, 20.1608, 19.60185, 20.059, 20.4011, 20.840, 20.9629]
assert(len(x) == len(Training))
assert(len(x) == len(Testing))
pltGraphs(title, x, Training, Testing, labelX)

# RandomForestRegressor, 100 trees, min_samples_leaf
x = [1e-6, 1e-5, 1e-4, 1e-3]
Training = [13.54, 18.15, 19.522, 20.04067]
Testing = [18.964, 19.0498, 19.617, 20.0391]
title = 'MAE vs. min_samples_leaf (RF)' 
labelX = 'min_samples_leaf'
assert(len(x) == len(Training))
assert(len(x) == len(Testing))
pltGraphs(title, x, Training, Testing, labelX)

# GradientBoostedRegressor, 150 trees,min_impurity_split
x=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 2e-1]
Training=  [2.001298125209856948e+01, 2.001298125209854106e+01, 
           2.001298125209855883e+01, 2.001298125209855883e+01, 
           2.001298125209854106e+01, 2.001298125209856238e+01]
Testing= [2.015904564379610520e+01, 2.015904564379608388e+01, 
          2.015904564379609099e+01, 2.015904564379610520e+01, 
          2.015904564379609099e+01, 2.015904564379609809e+01]
assert(len(x) == len(Training))
assert(len(x) == len(Testing))
title = 'MAE vs. min_impurity_split (Boosted)' 
labelX = 'min_impurity_split'
pltGraphs(title, x, Training, Testing, labelX)

#For GradientBooster(n_estimators = 150, minSamplesSplit)
x = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 2e-1]
Training= [1.997886194872963728e+01, 1.997886194872960530e+01,
           1.997384610230804824e+01, 1.998555289701618420e+01,
           2.007073902783057662e+01, 2.013350320521946557e+01]
Testing= [1.999647180976635852e+01, 1.999647180976633010e+01,
          1.999147692190578596e+01, 2.000421189361799890e+01,
          2.008679893551159523e+01, 2.015168629369095044e+01]
assert(len(x) == len(Training))
assert(len(x) == len(Testing))
title = 'MAE vs. min_samples_split (Boosted)' 
labelX = 'min_samples_split'
pltGraphs(title, x, Training, Testing, labelX)

#For GradientBooster(n_estimators = 150, minLeafSplit)
x = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 2e-1]
Training: [1.997473351476456571e+01, 1.997475050961516629e+01,
           1.998449319863669160e+01, 2.003122149000803276e+01,
           2.034426051331836760e+01, 2.049700625672175036e+01]
Testing: [1.999208415168982711e+01, 1.999268427755358246e+01,
          2.000070032785654561e+01, 2.004538930689321674e+01,
          2.036222309794497320e+01, 2.051358015672627033e+01]
title = 'MAE vs. min_samples_leaf (Boosted)' 
labelX = 'min_samples_leaf'
pltGraphs(title, x, Training, Testing, labelX)

#For KNN: WEIGHTED
x =[10, 20, 30]
Training= [5.695241923309865388e-03, 5.695241923309865388e-03,
           5.695241923309865388e-03]
Testing= [2.061291120905593033e+01, 2.009372873168479856e+01,
          1.992154612122467228e+01]
assert(len(x) == len(Training))
assert(len(x) == len(Testing))
title = 'MAE vs. Number of Neighbors (Weighted)' 
labelX = 'Number of Neighbors'
pltGraphs(title, x, Training, Testing, labelX)

# For KNN: Unweighted
x = [10, 15, 20, 25, 30, 35, 40]
Training= [1.865004769716800581e+01, 1.899704079009671176e+01,
           1.916408893180170026e+01, 1.926632978249361727e+01,
           1.933833969447137946e+01,  1.938747908093020200e+01,
           1.942644375439914839e+01]
Testing= [2.066056918720280322e+01, 2.033695535931076037e+01,
          2.017161184452829303e+01, 2.006558293706980933e+01,
          1.997440505907804464e+01, 1.992634496732652494e+01,
          1.989311396634556317e+01]  
assert(len(x) == len(Training))
assert(len(x) == len(Testing))
title = 'MAE vs. Number of Neighbors (Unweighted)' 
labelX = 'Number of Neighbors'
pltGraphs(title, x, Training, Testing, labelX)