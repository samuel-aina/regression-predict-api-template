"""
    Simple file to create a Sklearn model for deployment in our API
    Author: Explore Data Science Academy
    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.
"""

# Dependencies
import pandas as pd
import numpy as np
import pickle

from math import sin, cos, sqrt, atan2
from scipy.stats import norm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor

# Fetch training data and preprocess for modeling
test_df = pd.read_csv('data/test_data.csv')
train_df = pd.read_csv('data/train_data.csv')
riders_df = pd.read_csv('data/riders.csv')

# Merge datasets
testcols = test_df.columns # getting a list of the test columns
newtrain = train_df[testcols] # Refining the train columns to align to test columns
y = np.array(train_df["Time from Pickup to Arrival"]).reshape(-1,1)

# Combine train and test datasets
df = pd.concat([newtrain, test_df])

# Merging delivery and rider datasets
df = df.merge(riders_df, how = 'left', on = 'Rider Id')

df.columns = [col.replace(" ","_") for col in df.columns]
df.drop(['Precipitation_in_millimeters'], axis = 1, inplace = True)
df['Temperature'] = df['Temperature'].fillna(df['Temperature'].mean())


# Feature engineering

'''-----Time Conversion-----'''
df['Pickup_-_Time'] = pd.to_datetime(df['Pickup_-_Time'])
df['Confirmation_-_Time'] = pd.to_datetime(df['Confirmation_-_Time'])
df['Arrival_at_Pickup_-_Time'] = pd.to_datetime(df['Arrival_at_Pickup_-_Time'])
df['Placement_-_Time'] = pd.to_datetime(df['Placement_-_Time'])

# Calculating difference in time
df['Time_from_Placement_to_Confirmation'] = (df['Confirmation_-_Time'] - df['Placement_-_Time']).dt.total_seconds().astype(int)
df['Time_from_Confirmation_to_Arrival_at_Pickup'] = (df['Arrival_at_Pickup_-_Time'] - df['Confirmation_-_Time']).dt.total_seconds().astype(int)
df['Time_from_Arrival_at_Pickup_to_Pickup_-_Time'] = (df['Pickup_-_Time'] - df['Arrival_at_Pickup_-_Time']).dt.total_seconds().astype(int)

# Calculating Performance of Riders
df['Performance']=df['Age'] / df['No_Of_Orders']

# Creating Time Features
#df = df[['Pickup_-_Time']]
df['Pickup_Hour'] = df['Pickup_-_Time'].dt.hour
df['Pickup_Minute'] = df['Pickup_-_Time'].dt.minute
df['Pickup_Second'] = df['Pickup_-_Time'].dt.second

df.rename(columns={'Pickup_-_Day_of_Month':'Pickup_Day_of_Month',
                   'Pickup_-_Weekday_(Mo_=_1)':'Pickup_Day_of_Weekday'}, 
                 inplace=True)

df['Pickup_Hour_sin'] = np.sin(df.Pickup_Hour*(2.*np.pi/24))
df['Pickup_Hour_cos'] = np.cos(df.Pickup_Hour*(2.*np.pi/24))
df['Pickup_Month_sin'] = np.sin((df.Pickup_Day_of_Month)*(2.*np.pi/31))
df['Pickup_Month_cos'] = np.cos((df.Pickup_Day_of_Month)*(2.*np.pi/31))
df['Pickup_Day_sin'] = np.sin((df.Pickup_Day_of_Weekday)*(2.*np.pi/7))
df['Pickup_Day_cos'] = np.cos((df.Pickup_Day_of_Weekday)*(2.*np.pi/7))


# Dropping co-linear and unwanted variables
df = df.drop(['Placement_-_Day_of_Month','Placement_-_Weekday_(Mo_=_1)','Placement_-_Time','Confirmation_-_Day_of_Month',
              'Confirmation_-_Weekday_(Mo_=_1)','Confirmation_-_Time','Arrival_at_Pickup_-_Day_of_Month',
              'Arrival_at_Pickup_-_Weekday_(Mo_=_1)','Arrival_at_Pickup_-_Time', 'Pickup_Day_of_Month',
              'Pickup_Day_of_Weekday','Pickup_-_Time'], axis=1)

df_final = df.drop(['Order_No','User_Id','Vehicle_Type','Rider_Id'], axis=1)


# Dummying out the last categorical column
df_final = pd.get_dummies(df_final, drop_first=True)

# Scaling of dataset
scaler=StandardScaler()
x_scaled=scaler.fit_transform(df_final)
df_final=pd.DataFrame(x_scaled,columns=df_final.columns)

# Splitting our dataset again into Train and Validation Test
finaltrain = df_final[:len(train_df)]
finaltest = df_final[len(train_df):]

x_train, x_val, y_train, y_val = train_test_split(finaltrain, y, test_size = 0.3)




#lr = LinearRegression()
#print ("Training Model...")
#lr.fit(x_train, y_train)

# Fit model
params = {
    'n_estimators':[300],
    'max_depth': [4],
    'learning_rate': [0.15],
    'colsample_bylevel':[0.5],
    'l2_leaf_reg':[6]}

cbr = GridSearchCV(CatBoostRegressor(random_state=50), cv=2,scoring='neg_mean_squared_error', param_grid=params)
#print ("Training Model...")
cbrm = cbr.fit(x_train, y_train)

# Pickle model for use within our API
save_path = 'team1_sendy_catboost.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(cbr, open(save_path,'wb'))