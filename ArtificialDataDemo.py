import numpy as np
import pyximport; pyximport.install(setup_args={
                              "include_dirs":np.get_include()},
                            reload_support=True)

import RegressionTsetlinMachine

#hyper parameters
T = 3
s = 2
number_of_clauses = 3
states = 100
epochs = 100

#import data
df = np.loadtxt("2inNoNoise.txt").astype(dtype=np.float32)

#train with 80% of data and test with 20%
NOofTestingSamples = df.shape[0]*20//100
NOofTrainingSamples = df.shape[0]-NOofTestingSamples

#training data
X_train = df[0:NOofTrainingSamples,0:df.shape[1]-1].astype(dtype=np.int32)
y_train = df[0:NOofTrainingSamples,df.shape[1]-1:df.shape[1]].flatten()
rows, number_of_features = X_train.shape

max_target = (max(y_train))
min_target = (min(y_train))

#testing data
X_test = df[NOofTrainingSamples:df.shape[0],0:df.shape[1]-1].astype(dtype=np.int32)
y_test = df[NOofTrainingSamples:df.shape[0],df.shape[1]-1:df.shape[1]].flatten()

#call and train the regression tsetlin machine
tsetlin_machine = RegressionTsetlinMachine.TsetlinMachine(number_of_clauses, number_of_features, states, s, T, max_target, min_target)
tsetlin_machine.fit(X_train, y_train, y_train.shape[0], epochs=epochs)

#test the setlin machine
print ("Average Absolute Error on Training Data:", tsetlin_machine.evaluate(X_train, y_train, y_train.shape[0]))
print ("Average Absolute Error on Test Data:", tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0]))
