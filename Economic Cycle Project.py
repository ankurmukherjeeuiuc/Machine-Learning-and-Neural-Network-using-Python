#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import the `pandas` library as `pd`
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load in the data with `read_csv()`
data = pd.read_csv(r'C:\Users\ankur\OneDrive\Desktop\Machine Learning\Group Project\MLF_GP2_EconCycle.csv')
data.head()

#data.info()


# In[55]:


data.describe()


# In[5]:


#Preprocessing - Handling 0 and missing data
import numpy as np
from sklearn.impute import SimpleImputer
#changing zero value to NAN
data.PCT9MOFWD.replace(0, np.nan, inplace=True)
#Drop the NA value since its only one row
data = data.dropna()
data.shape


# In[22]:


#Exploratory Data Analysis
#Histogram to plot the USPHCI index over the years
sns.set()
plt.hist(data['USPHCI'])
plt.xlabel('USPHCI Index range')
plt.ylabel('Frequency')
plt.show()


# In[62]:


corr = data.corr()
print(corr)


# In[63]:


import numpy as np
cm = np.corrcoef(data.values.T)
sns.set(font_scale=1.3)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.1f',annot_kws={'size': 8},yticklabels=True,xticklabels=True)
plt.show()


# In[64]:


#Scatter plot for the above variables which show strong +ve and -ve correlation
import matplotlib.pyplot as plt
import seaborn as sns

cols = ['PCT3MOFWD','PCT6MOFWD', 'PCT9MOFWD', 'CP1M', 'CP3M','CP6M']
sns.pairplot(data[cols], size=3.5)
plt.tight_layout()
plt.show()


# In[58]:


#Bee Swarm plot 
sns.swarmplot(x='PCT3MOFWD', y='CP3M', data=data)
plt.xlabel('CP3M')
plt.ylabel('PCT3MOFWD')
plt.show()


# In[17]:


#Train Test split
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler 

X = data.iloc[:, 6:12]
#print(X)
y3month = data.iloc[:, 12].values
y6month = data.iloc[:, 13].values
y9month = data.iloc[:, 14].values
#print(y6month)
X_train, X_test, y_train, y_test = train_test_split(X, y3month, test_size=0.10,random_state=42)
#print( X_train.shape, y_train.shape)
# performing preprocessing part 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



# In[11]:


##########################Models for 3 Month forward##########################
# I. Linear regression model 
import statsmodels.api as sm

X = X_train
y = y_train

# Note the difference in argument order
model_train = sm.OLS(y, X).fit()
predictions = model_train.predict(X) # make the predictions by the model

# Print out the statistics
model_train.summary()
#For test set
X = X_test
y = y_test

model_test = sm.OLS(y, X).fit()
predictions = model_test.predict(X) # make the predictions by the model

# Print out the statistics
model_test.summary()


# In[13]:


# II. Regression Tree Model
from sklearn.tree import DecisionTreeRegressor
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE
# Instantiate a DecisionTreeRegressor 'dt'
dt = DecisionTreeRegressor(max_depth=4,min_samples_leaf=0.1,random_state=3)
# Fit 'dt' to the training-set
dt.fit(X_train, y_train)
# Predict test-set labels
y_pred = dt.predict(X_test)
# Compute test-set MSE
mse_dt = MSE(y_test, y_pred)
# Compute test-set RMSE
rmse_dt = mse_dt**(1/2)
# Print rmse_dt
print(rmse_dt)
#Print
print('The regression Tree model fits extremely well with a very low rmse')


# In[14]:


#Cross Validation
from sklearn.model_selection import cross_val_score
# Evaluate the list of MSE ontained by 10-fold CV
# Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV = - cross_val_score(dt, X_train, y_train, cv= 10,scoring='neg_mean_squared_error',n_jobs = -1)
# Fit 'dt' to the training set
dt.fit(X_train, y_train)
# Predict the labels of training set
y_predict_train = dt.predict(X_train)
# Predict the labels of test set
y_predict_test = dt.predict(X_test)
# Training set MSE
print('Train MSE: {:.7f}'.format(MSE(y_train, y_predict_train)))
# Test set MSE
print('Test MSE: {:.8f}'.format(MSE(y_test, y_predict_test)))


# In[15]:


#III. SVD Regression
from sklearn.svm import SVR
svm=SVR(kernel='linear',degree=1,gamma=0.5,C=1.0)
svm.fit(X_train,y_train)
print("R-square:" + str(svm.score(X_train,y_train)))


# In[152]:


print("Clearly the Decision Regression Tree is the best model that describes the Forward USHCPI index change wrt to CP and the spreads")


# In[16]:


#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
# Instantiate a random forests regressor 'rf' 400 estimators
rf = RandomForestRegressor(n_estimators=400,min_samples_leaf=0.12,random_state=1)
# Fit 'rf' to the training set
rf.fit(X_train, y_train)
# Predict the test set labels 'y_pred'
y_pred = rf.predict(X_test)
# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)
# Print the test set RMSE
print('Test set RMSE of rf: {:.6f}'.format(rmse_test))
print("The Random Forest Regressor does a very good job in training the individual trees and introduces further randomization")


# In[26]:


#####################6 Month Forward#################################
#I. Linear regression model 

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler 

X = data.iloc[:, :-1]
#print(X)
#y3month = data.iloc[:, 6:12].values
y6month = data.iloc[:, 13].values
#y9month = data.iloc[:, 14].values
#print(y6month)
X_train6, X_test6, y_train6, y_test6 = train_test_split(X, y6month, test_size=0.10,random_state=42)
#print( X_train.shape, y_train.shape)
# performing preprocessing part 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train6)
X_train6 = scaler.transform(X_train6)
X_test6 = scaler.transform(X_test6)


# In[25]:


import statsmodels.api as sm

X = X_train6
y = y_train6

# Note the difference in argument order
model_train = sm.OLS(y, X).fit()
predictions = model_train.predict(X) # make the predictions by the model

# Print out the statistics
model_train.summary()
#For test set
X = X_test6
y = y_test6

model_test = sm.OLS(y, X).fit()
predictions = model_test.predict(X) # make the predictions by the model

# Print out the statistics
model_test.summary()


# In[28]:


# II. Regression Tree Model for 6month
from sklearn.tree import DecisionTreeRegressor
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE
# Instantiate a DecisionTreeRegressor 'dt'
dt = DecisionTreeRegressor(max_depth=4,min_samples_leaf=0.1,random_state=3)
# Fit 'dt' to the training-set
dt.fit(X_train6, y_train6)
# Predict test-set labels
y_pred6 = dt.predict(X_test6)
# Compute test-set MSE
mse_dt = MSE(y_test6, y_pred6)
# Compute test-set RMSE
rmse_dt = mse_dt**(1/2)
# Print rmse_dt
print(rmse_dt)
#Print
print('The regression Tree model for the 6month forward also fits extremely well with a very low rmse')


# In[29]:


#III. SVD Regression for 6month
from sklearn.svm import SVR

svm=SVR(kernel='linear',degree=1,gamma=0.5,C=1.0)
svm.fit(X_train6,y_train6)
print("R-square:" + str(svm.score(X_train6,y_train6)))


# In[36]:


#Random Forest Regressor for 6month prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
# Instantiate a random forests regressor 'rf' 400 estimators
rf = RandomForestRegressor(n_estimators=400,min_samples_leaf=0.12,random_state=1)
# Fit 'rf' to the training set
rf.fit(X_train6, y_train6)
# Predict the test set labels 'y_pred'
y_pred6 = rf.predict(X_test6)
# Evaluate the test set RMSE
rmse_test = MSE(y_test6, y_pred6)**(1/2)
# Print the test set RMSE
print('Test set RMSE of rf: {:.6f}'.format(rmse_test))
print("The Random Forest Regressor for 6 month also does a very good job in training the individual trees and introduces further randomization")


# In[41]:


#Train Test split
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler 

X = data.iloc[:, 6:12]
#print(X)
y9month = data.iloc[:, 14].values
#print(y6month)
X_train9, X_test9, y_train9, y_test9 = train_test_split(X, y9month, test_size=0.10,random_state=42)
#print( X_train.shape, y_train.shape)
# performing preprocessing part 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train9)
X_train = scaler.transform(X_train9)
X_test = scaler.transform(X_test9)



# In[42]:


#for 9month
import statsmodels.api as sm

X = X_train9
y = y_train9

# Note the difference in argument order
model_train = sm.OLS(y, X).fit()
predictions = model_train.predict(X) # make the predictions by the model

# Print out the statistics
model_train.summary()
#For test set
X = X_test9
y = y_test9

model_test = sm.OLS(y, X).fit()
predictions = model_test.predict(X) # make the predictions by the model

# Print out the statistics
model_test.summary()


# In[43]:


# II. Regression Tree Model for 9month
from sklearn.tree import DecisionTreeRegressor
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE
# Instantiate a DecisionTreeRegressor 'dt'
dt = DecisionTreeRegressor(max_depth=4,min_samples_leaf=0.1,random_state=3)
# Fit 'dt' to the training-set
dt.fit(X_train9, y_train9)
# Predict test-set labels
y_pred9 = dt.predict(X_test9)
# Compute test-set MSE
mse_dt = MSE(y_test9, y_pred9)
# Compute test-set RMSE
rmse_dt = mse_dt**(1/2)
# Print rmse_dt
print(rmse_dt)
#Print
print('The regression Tree model for the 6month forward also fits extremely well with a very low rmse')


# In[44]:


#III. SVD Regression for 9month
from sklearn.svm import SVR

svm=SVR(kernel='linear',degree=1,gamma=0.5,C=1.0)
svm.fit(X_train9,y_train9)
print("R-square:" + str(svm.score(X_train9,y_train9)))


# In[46]:


#Random Forest Regressor for 9month prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
# Instantiate a random forests regressor 'rf' 400 estimators
rf = RandomForestRegressor(n_estimators=400,min_samples_leaf=0.12,random_state=1)
# Fit 'rf' to the training set
rf.fit(X_train9, y_train9)
# Predict the test set labels 'y_pred'
y_pred9 = rf.predict(X_test9)
# Evaluate the test set RMSE
rmse_test = MSE(y_test9, y_pred9)**(1/2)
# Print the test set RMSE
print('Test set RMSE of rf: {:.6f}'.format(rmse_test))
print("The Random Forest Regressor for 9 month also does a very good job in training the individual trees and introduces further randomization")


# In[ ]:




