# Dataset Information
**Usage**: Predicting the attribute "mpg", 8 of the original instances were removed because they had unknown values for the "mpg" attribute. The data concerns city-cycle fuel consumption in miles per gallon, to be predicted in terms of 3 multivalued discrete and 5 continuous attributes.

- **Requires preprocessing?** YES - determined by: 'preprocessing_description': None
- **Missing Values?** YES - determined by: 'has_missing_values': 'yes' ; 'missing_values_symbol': 'NaN' 
 
## Variable Information:
```
           name     role         type demographic description units missing_values
0  displacement  Feature   Continuous        None        None  None             no
1           mpg   Target   Continuous        None        None  None             no
2     cylinders  Feature      Integer        None        None  None             no
3    horsepower  Feature   Continuous        None        None  None            yes
4        weight  Feature   Continuous        None        None  None             no
5  acceleration  Feature   Continuous        None        None  None             no
6    model_year  Feature      Integer        None        None  None             no
7        origin  Feature      Integer        None        None  None             no
8      car_name       ID  Categorical        None        None  None             no
```
# Data Preprocessing
- **Missing Values**: Addressed missing data by removing samples with any missing values. I had to match the indicies in y (the targets) that I removed to those in X that were NaN in features.
- **Normalization**: I standardized the data with StandardScalar.
- **Feature Selection**: Did not perform any feature selection. If I wanted to select features, I could probably remove those that don't have much to do with mpg? Weight might be important to how many mpg you get, as well as acceleration. Don't seem to need to do this anyways.


# Model Training
**Coefficients**: Compute the optimal set of coefficients for the model that minimizes
prediction error (use RMSE defined below).



# Libraries Used
The assignment stated there was no limitation on libraries we could use.
## RMSE
```python
from sklearn.metrics import root_mean_squared_error
```
From the sklearn.metrics library, there exists an implementation of the root mean sqaured error formula- implemented like what is required for RSME in the pdf. The following is the parameters for the imported library. 
Found at: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html
```python
sklearn.metrics.root_mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
```
The two variables that matter most: 
- y_true: Ground truth (or y_i in the equation)
- y_ pred: "estimated target values" or the prediction we obtain from our linear regression (y_prediciton in my code)
- N is gathered from the size of the test data inputs.

## K-Fold
Found at: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
``` python
from sklearn.model_selection import KFold
# class sklearn.model_selection.KFold(n_splits=5, *, shuffle=False, random_state=None)
```
- n_split: the number of folds, in our case 10
- shuffle: shuffles data before splitting into batches
- random_state: can set seed for reproducibility (I set for debugging but will turn off for ouputs later)

## Pearson Correlation Coefficient
Compute r (Pearson Correlation Coefficient) for each feature. Tells us the correlation between the feature and target. (any feature from X (horsepower, cylinders. etc...) and the y target (mpg).
```python
from sklearn.feature_selection import r_regression
# klearn.feature_selection.r_regression(X, y, *, center=True, force_finite=True)
```
Where we care about:
- X is the data matrix - this will be our features a.k.a. data
- y is the target vector - targets


## Ridge Regression
Graduate students must implement ridge regression. This will help us find a new line of best fit that isn't overfitted to the training data. It will give us better long term predictions and lower our variance. 
### Ridge
```python
from sklearn.linear_model import Ridge
# class sklearn.linear_model.Ridge(alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto', positive=False, random_state=None)
```
- alpha: this is the lamda term we learned in class for ridge regression. When we increase this term, we are less sensitive to our training data. This term can go from 0->inf. We test multiple alphas by passing them in through GridSearchCV

### GridSearchCV()
Found: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
```python
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
# class sklearn.model_selection.GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)
```
- estimator: we pass in ridge here
- param_grid: dictionary with parameters to try as values- we will try all of these as the alpha (lambda really) to our ridge function
- cv: determines the cross validation splitting strategy. 10 to make it 10 fold as suggested
- n_jobs: number of jobs to run parallel. -1 means all processors are used
#### best_estimator_
The estimator that gave the highest score in the left out data. We can examine this estimators coefficients, intercept, the alpha that was selected by ridge regression, etc. 
- Coefficient: These are the fitted parameters of the model (coefficients for our features). These are the weights assigned to each feature-- tells us which feature contributes more to the target (mpg). The best weights are printed in the output *Best Coefficients*. 


# Example Table Output 
```
         displacement           cylinders               horsepower              weight          acceleration            model_year              origin          RMSE
Fold 1  -0.80575                -0.78096                -0.77478                -0.83250                0.40761         0.59076         0.56403         3.69041
Fold 2  -0.80059                -0.77141                -0.77836                -0.83470                0.43115         0.57708         0.58325         3.66364
Fold 3  -0.80995                -0.77963                -0.77986                -0.83550                0.42368         0.58664         0.56408         3.96284
Fold 4  -0.81833                -0.78971                -0.77573                -0.83934                0.39700         0.56718         0.57036         4.19367
Fold 5  -0.80320                -0.77684                -0.77750                -0.83189                0.42254         0.59613         0.57683         2.95831
Fold 6  -0.79541                -0.76633                -0.77120                -0.82496                0.42273         0.57050         0.56622         3.27653
Fold 7  -0.80644                -0.78202                -0.78933                -0.83636                0.45457         0.58508         0.55059         3.41353
Fold 8  -0.80514                -0.77800                -0.77966                -0.82937                0.43225         0.56231         0.56183         2.66397
Fold 9  -0.80225                -0.77527                -0.78271                -0.82969                0.42769         0.58892         0.55921         3.20871
Fold 10 -0.80489                -0.77637                -0.77536                -0.82867                0.41419         0.58074         0.55650         2.67502
```
# Example Ridge Regression Output
```python
RMSE for Ridge Regression on Fold: 3.61857
R2 for Ridge Regression on Fold: 0.79601
MAE for Ridge Regression on Fold: 2.84209
Best Alpha: 1.0
Best Coefficients: [ 2.77964566 -1.19352311 -1.06042188 -5.35437067  0.38970928  2.80157789  1.16985746]
```
We can see that the Feature that has the most impact on mpg is the model_year, at 2.8 for the weight coefficient. The least is the Weight of the vehicle at -5.35. The best alpha (or lambda from the equations we learned in class) is 1.0 for alphas [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]. 

I ran through various alphas to find the best with alpha also being = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]. My best alpha was still 1.0.

## What I Learned
This lab really helped me to understand Linear Regression. I now understand the difference between Normalization (normalizing data in a 0-1 range) and Standardization (shifting data to have a mean of 0 and an SD of 1). I learned the best instances to use both- and came to the conclusion that Standardization was more in line with what I needed for the lab. This was in case of outliers to the data, and I learned Normlization is usually used for NN/CNNs.

Ridge Regression was a bit tough to understand at first. I watched a lot of videos that explained why over-fitting occurred in Linear Regression(also learned Linear Regression and OLS literally mean the same thing). With Ridge Regression, we are trying to minimize variance by increasing a bias for a more generalized model. This really is just modifying the line of best fit so that it doesn't hyper favor only the training data and allows for better sum squared residuals on testing as well. 

Challenges I faced during implementation were understanding exactly what the sections on coefficients were asking. Here:
**"Coefficients: Compute the optimal set of coefficients for the model that minimizes
prediction error (use RMSE defined below)."**

The above did not make much sense to me unless students were asked to do ridge regression (which as a grad student, I was able to do). Were we supposed to try and make this number closer to 0? Were we supposed to try and do this with Feature Selection? An RMSE of ~3 is only off by about 3 units of the target variable, which was mpg. That to me sounds like a very low RMSE to begin with. Were we supposed to show the K-Fold outcomes and say "These coefficients gave me the lowest RMSE for a non-ridge regression implementation? I think a bit more clarity would have been nice.

Data processing was fairly easy. Remove all instances of NaN. Considering the data was plenty, I didn't think it would mean much to flub 6 entries, so I just decided to remove.

Mentioned all the libraries I used that I thought were important. I think others like r2_score are pretty self-explanatory. (Gives me the r2 score where best score is 1 and means we have best fitting model). I previously printed the r2 score of both the K-Fold and Ridge Regression outputs, but it wasn't really asked for and they were pretty close for some outputs. I printed for Ridge Regression. 0.8 is a decent fit- though we are trying to be generalized with fitting so you can't really expect a 1. 