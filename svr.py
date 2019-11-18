#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values


#splitting the data set into dataset and training set
"""from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)"""


#feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

#fitting the SVR to the dataset
#create your regressor here
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,Y)

#Predicting a new result with Linear Regression
Y_pred=regressor.predict([[6.5]])

#Visualising the SVR results
#Visualising Polynomial Regressio n Results
#X_grid=np.arange(min(X),0.1)
#X_grid=X_grid((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title("True of Bluff( SVR Result)")
plt.xlabel("PositionLevel")
plt.ylabel("Salary")
plt.show()
