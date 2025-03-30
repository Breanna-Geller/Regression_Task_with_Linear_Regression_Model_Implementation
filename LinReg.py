from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import r_regression
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np

  
# fetch dataset 
auto_mpg = fetch_ucirepo(id=9) 
  
# data (as pandas dataframes) 
X = auto_mpg.data.features
y = auto_mpg.data.targets 


################# Preprocessing ########################################
# Missing Values
X_dropped = X.dropna(how='any') #any NaN and inplace modifies orig. dataframe
# Find indices of dropped rows
dropped_indices = X.index.difference(X_dropped.index)
# Drop corresponding indices in y
y.drop(index=dropped_indices, inplace=True)



# Normalization - used standardScalar; mean = 0, stdev = 1
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X_dropped)
scaled_df = pd.DataFrame(scaled_data,columns=X_dropped.columns)


################# Model Training Straight LinReg ##################################
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train.values.ravel())

# Evaluate the model
y_prediction = model.predict(X_test)
# Evaluate where y_i is ground truth X__i_b is the prediction from lin regression - said no restriction on external libraries
rmse = root_mean_squared_error(y_test, y_prediction)
#Had this print but not sure if assignment just wants the table RMSE...


################# 10-Fold Cross Validation #########################
# Done for each feature then RMSE
KFold = KFold(n_splits=10, shuffle=True) # set random state for reproducibility

# Table row 1 - feature names
print("\t", end=" ")
for feature in scaled_df.columns:
    print(f"{feature}", end="\t\t")
print("RMSE")
fold_count = 1
for train_index, test_index in KFold.split(scaled_df):
    print(f"Fold {fold_count}", end="\t")
    fold_count+=1
    X_train, X_test = scaled_df.iloc[train_index], scaled_df.iloc[test_index]
    y_train, y_test = y.iloc[train_index].values.ravel(), y.iloc[test_index].values.ravel()

    # Feature Selection - r_regression
    # Calculate the correlation coefficients
    correlation = r_regression(X_train, y_train)
    # Select features with a correlation coefficient greater than 0.1
    for corr in  correlation:
        print(f"{corr:.5f}", end="\t\t")
    # Perform linear regression on the Fold
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_prediction = model.predict(X_test)
    #RMSE for the fold
    rmse = root_mean_squared_error(y_test, y_prediction)
    print(f"{rmse:.5f}")
    r2 = r2_score(y_test, y_prediction)
    mae = mean_absolute_error(y_test, y_prediction)


########### Ridge Regression ###############
# Perform Ridge regression on the Fold
X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2)
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid={'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}, cv=10, n_jobs=-1)
ridge_cv.fit(X_train, y_train)
y_prediction = ridge_cv.predict(X_test)
#RMSE for the fold
rmse = root_mean_squared_error(y_test, y_prediction)
print(f"RMSE for Ridge Regression on Fold: {rmse:.5f}")
r2 = r2_score(y_test, y_prediction)
mae = mean_absolute_error(y_test, y_prediction)
print(f"R2 for Ridge Regression on Fold: {r2:.5f}")
print(f"MAE for Ridge Regression on Fold: {mae:.5f}")
print(f"Best Alpha: {ridge_cv.best_estimator_.alpha}")
# print(ridge_cv.best_estimator_.intercept_)
print("\t", end=" ")
for feature in scaled_df.columns:
    print(f"{feature}", end=" ")
print("\n")
print(f"Best Coefficients: {ridge_cv.best_estimator_.coef_}")
