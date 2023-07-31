# importing necessary modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data collection and processing
# Loading data set to pandas data frame
sonar_data = pd.read_csv('./sonar_all-data.csv')

# sonar_data["Label"].value_counts()

# groups the data based on the value in that column
# sonar_data.groupby("Label").mean()

# Droppping the "Labels"(to use it for test or in unsupervised learning)
# Seperating data and labels
X = sonar_data.drop(columns="Label",axis=1)
y = sonar_data["Label"]

# Splitting Testing and Training data
# Note that to do this via the function, it is mandatory to maintain the order
X_test,X_train,Y_test,Y_train = train_test_split(X,y,test_size=0.1,stratify=y,random_state=1)
#'random state' is kind-a-like the seeding for the splitting
# statify is used to split the test and train sets based on the count of types available

# Model Training
model = LogisticRegression()
#Training the Logistic Regression model
model.fit(X_train,Y_train)

# Accuracy in training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train) # Comparing two 'answers'
print(training_data_accuracy)

# Accuracy in test data
# Accuracy in training data
X_train_prediction_test = model.predict(X_test)
training_data_accuracy_test = accuracy_score(X_train_prediction_test,Y_test) # Comparing two 'answers'
print(training_data_accuracy_test)

# Making a predictive system
input_data = (0.0228,0.0853,0.1,0.0428,0.1117,0.1651,0.1597,0.2116,0.3295,0.3517,0.333,0.3643,0.402,0.4731,0.5196,0.6573,0.8426,0.8476,0.8344,0.8453,0.7999,0.8537,0.9642,1.0,0.9357,0.9409,0.907,0.7104,0.632,0.5667,0.3501,0.2447,0.1698,0.329,0.3674,0.2331,0.2413,0.2556,0.1892,0.194,0.3074,0.2785,0.0308,0.1238,0.1854,0.1753,0.1079,0.0728,0.0242,0.0191,0.0159,0.0172,0.0191,0.026,0.014,0.0125,0.0116,0.0093,0.0012,0.0036)
#changing input data into a numpy array (to improve performance)
input_data_as_numpy_array = np.asarray(input_data)

# Reshaping the numpy array as we are predicting for one instance(else model will get confused by the number of data points)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# Note:- '-1' is to make all the data in a 1D either as rows or columns (depending on the position of -1) and the other one signifies that it is an addditional column or row added to make the array a 2D ONE
# Here (1,-1) creates a 2D array of the form (1,60) and (-1,1) creates it of the form (60,1)
# LESSER NUMBER OF COLUMS AND MORE NUMBER OF ROWS IS THE IDEAL SITUATION

#Prediction
prediction = model.predict(input_data_reshaped)
# print(prediction) will actually give us a list

#Final output in style
if prediction[0] == 'R':
  print("Chill! That's a rock")
else:
  print("Oh Crap! It's a mine.\nAbort!")

