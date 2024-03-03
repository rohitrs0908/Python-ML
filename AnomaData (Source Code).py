#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Load the CSV file into a pandas DataFrame
data = pd.read_excel("C:/Users/Noush/Downloads/AnomaData.xlsx")


# In[3]:


data.head()


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


# Check for missing values
print(data.isnull().sum())


# In[6]:


# Summary statistics
print(data.describe())


# In[7]:


# Visualize the distribution of 'y' (target variable)
plt.hist(data['y'])
plt.xlabel('Anomaly')
plt.ylabel('Frequency')
plt.title('Distribution of Anomalies')
plt.show()


# In[8]:


# Handle missing values
data.fillna(data.mean(), inplace=True)


# In[9]:


#Get the correct datatype for date

data['time'] = pd.to_datetime(data['time'])
print(data.dtypes)


# In[10]:


#Feature Engineering:
#Time-Based Features
data.loc[:, 'hour'] = data['time'].dt.hour
data.loc[:, 'day_of_week'] = data['time'].dt.dayofweek
data.loc[:, 'month'] = data['time'].dt.month


# In[11]:


#Transform Existing Features
import numpy as np

data['log_x1'] = np.where(data['x1'] > 0, np.log(data['x1']), np.nan)
data['sqrt_x2'] = np.where(data['x2'] >= 0, np.sqrt(data['x2']), np.nan)


# In[12]:


#Encode Categorical Features
print("No categorical feature to encode. Skipping one-hot encoding step.")


# In[13]:


#Feature Selection:
#Correlation Analysis
threshold = 3
correlation_matrix = data.corr()
correlation_with_target = correlation_matrix['y'].abs().sort_values(ascending=False)
relevant_features = correlation_with_target[correlation_with_target > threshold].index

print("Relevant features based on correlation analysis:")
print(relevant_features)


# In[14]:


#Train/Test Split
from sklearn.model_selection import train_test_split


# In[15]:


X = data.drop(columns=['time', 'y'])
y = data['y']


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


# In[18]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# In[19]:


model = RandomForestClassifier()


# In[20]:


X_clean = X.dropna(subset=['log_x1', 'sqrt_x2'])
y_clean = y.loc[X_clean.index] 


# In[21]:


num_folds = 5
scores = cross_val_score(model, X_clean, y_clean, cv=num_folds, scoring='accuracy')
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())


# In[22]:


#Model Selection, Training, and Evaluation

from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# In[23]:


# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)


# In[24]:


# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


# In[25]:


# Model evaluation
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy:", accuracy_score(y_test, y_pred_test))
print("Training Precision:", precision_score(y_train, y_pred_train, zero_division=1))
print("Testing Precision:", precision_score(y_test, y_pred_test, zero_division=1))
print("Training Recall:", recall_score(y_train, y_pred_train, zero_division=1))
print("Testing Recall:", recall_score(y_test, y_pred_test, zero_division=1))
print("Training F1-score:", f1_score(y_train, y_pred_train, zero_division=1))
print("Testing F1-score:", f1_score(y_test, y_pred_test, zero_division=1))


# In[26]:


#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[27]:


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# In[28]:


model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')


# In[29]:


grid_search.fit(X_train, y_train)


# In[30]:


best_params = grid_search.best_params_
print("Best Parameters:", best_params)


# In[31]:


best_model = grid_search.best_estimator_
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)


# In[32]:


train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)


# In[33]:


#Model deployment plan
import joblib

#Save the model to a file
joblib.dump(best_model, 'model.pkl')


# In[34]:


#Load the model in a production environment
loaded_model = joblib.load('model.pkl')
loaded_model


# In[ ]:





# In[ ]:




