# IBM Nordcloud DS Challange
---
Date: 12.11.2021
<br>
Author: Sergej Schweizer
<br><br>
The dataset consists of 463.291 entries from a 2021 online advertising campaign.
The training set pertains to the time period of 9.04.2021 - 13.04.2021, while the test set only
contains the day of 14.04.2021.


# 1. Import packages
---


```python
#!pip install tensorflow tensorflow_data_validation fastcluster imbalanced-learn bayesian-optimization
```


```python
import pandas as pd
import numpy as np
from typing import Tuple
from pandarallel import pandarallel

import seaborn as sns 
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_data_validation as tfdv

from sklearn import linear_model
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, classification_report, plot_confusion_matrix
from sklearn.feature_selection import RFE, SelectKBest, SelectFromModel, chi2, f_classif

import xgboost as xgb
from xgboost import XGBClassifier, plot_tree, cv, plot_importance

import itertools
import regex as re

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN

from bayes_opt import BayesianOptimization


import warnings
warnings.filterwarnings('ignore')
pandarallel.initialize()
```

    INFO: Pandarallel will run on 8 workers.
    INFO: Pandarallel will use Memory file system to transfer data between the main process and workers.


# 2. Load data
---


```python
# check the csv before load
!head -n 3 train_set.csv
```

    Session Id,DateTime,User Id,Product Type,Campaign Id,Webpage Id,Product Category,Advertisment Size,User Depth,Internet Browser Id,Gender,Age Group,City Size,Device Used,Interested In Cars,Interested In Food,Interested In News,Interested In Technology,Interested In Medicine,Interested In Politics,Interested In Fashion,Interested In Astronomy,Interested In Animals,Interested In Travel,Clicked
    229ac4c2-0ee9-4a3b-b52c-3b20c9d43039,09-04-2021 00:00,858557,C,359520,13787,4,,3.0,10.0,F,45-54,5900000.0,Mobile,1,0,1,0,1,0,0,0,1,0,No
    87c0f74a-fa7a-4b3f-bc48-ad1a5f80de2e,09-04-2021 00:00,243253,C,105960,11085,5,,2.0,8.0,F,25-34,,Mobile,1,0,1,1,1,0,0,0,1,1,No


### Notes
* separator is comma !
* existing column names
* one DateTime column


```python
# load
df_train =  pd.read_csv('train_set.csv', sep=',')
df_test =  pd.read_csv('test_set.csv', sep=',')
```


```python
# filter < in column names
df_train.columns = df_train.columns.str.replace(r'[<,>]', '')
df_test.columns = df_test.columns.str.replace(r'[<,>]', '')
```


```python
df_train.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Session Id</th>
      <th>DateTime</th>
      <th>User Id</th>
      <th>Product Type</th>
      <th>Campaign Id</th>
      <th>Webpage Id</th>
      <th>Product Category</th>
      <th>Advertisment Size</th>
      <th>User Depth</th>
      <th>Internet Browser Id</th>
      <th>...</th>
      <th>Interested In Food</th>
      <th>Interested In News</th>
      <th>Interested In Technology</th>
      <th>Interested In Medicine</th>
      <th>Interested In Politics</th>
      <th>Interested In Fashion</th>
      <th>Interested In Astronomy</th>
      <th>Interested In Animals</th>
      <th>Interested In Travel</th>
      <th>Clicked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>229ac4c2-0ee9-4a3b-b52c-3b20c9d43039</td>
      <td>09-04-2021 00:00</td>
      <td>858557</td>
      <td>C</td>
      <td>359520</td>
      <td>13787</td>
      <td>4</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>10.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>87c0f74a-fa7a-4b3f-bc48-ad1a5f80de2e</td>
      <td>09-04-2021 00:00</td>
      <td>243253</td>
      <td>C</td>
      <td>105960</td>
      <td>11085</td>
      <td>5</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b746f0ad-1aa3-492c-b7fa-5dd95643fb51</td>
      <td>09-04-2021 00:00</td>
      <td>243253</td>
      <td>C</td>
      <td>359520</td>
      <td>13787</td>
      <td>4</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 25 columns</p>
</div>




```python
# we concat both datasets for preprocessing and scalling reasons
df = pd.concat([df_train, df_test], axis=0)
```


```python
# relative amount of test data
df_test.shape[0] / df.shape[0]
```




    0.15425725947622554



### Columns descriptions:
1. Session Id – refers to session identifier
2. DateTime – refers to date and time of the entry
3. User Id – refers to the user identifier
4. Product Type – refers to the product type for which the advertisement is
5. Campaign Id – refers to the campaign identifier of the advertisement
6. Webpage Id - refers to the webpage identifier
7. Product Category – refers to the category of the product
8. Advertisement Size – refers to the area of an advertisement posted on a website, measured in pixels.
9. User Depth - refers to user’s duration of exposure to the advertisement during the respective entry
(3 being the longest, 1 being the shortest and NA being the inability to measure the time spent)
10. Internet Browser Id – refers to the identifier of the Internet browser type which is used by the user
11. Gender – refers to the gender of the user
12. Age Group – refers to the age group of the user
13. City Size – refers to the size of the city where the user is located
14. Device Used – refers to the device used by the user (could be Mobile or PC/Laptop)
15. Clicked – refers to the fact that the user clicked on the advertisement16. Interested in Cars - refers to the fact that the user is interested in cars
17. Interested in Food - refers to the fact that the user is interested in food
18. Interested in News - refers to the fact that the user is interested in news
19. Interested in Technology - refers to the fact that the user is interested in technology
20. Interested in Medicine - refers to the fact that the user is interested in Medicine
21. Interested in Politics - refers to the fact that the user is interested in Politics
22. Interested in Fashion - refers to the fact that the user is interested in fashion
23. Interested in Astronomy - refers to the fact that the user is interested in Astronomy
24. Interested in Animals - refers to the fact that the user is interested in animals
25. Interested in Travel - refers to the fact that the user is interested in travelling

# 3. Explorative Data Analysis
---
The EDA can be devided in two different parts.

Feature engineering:
* Preprocess Categorical Columns
* Preprocess Numerical Columns
* Preprocess Date Columns
* Understand the Natare of NANs
* Generate Interactions of Predictors
* Generate aggregated Predictors
* Scale Predictors

Feature selection:
* Build baseline with all Predictors
* Use SMOTE to resolve the unbalanced classes issue
* Build model with uncorelated (intra) Predictors
* Build model with correlated (with Target) Predictors
* Build model with RFE selected Predictors
* Visualize Feature Importance
---


```python
# Generate dataset statistics
train_stats = tfdv.generate_statistics_from_dataframe(df)
# Visualize
tfdv.visualize_statistics(train_stats)
```


![png](https://github.com/SergejSchweizer/IBM_Nordcloud/blob/master/output_1.png?raw=true)


### Notes
* Column: Advertisment Size has 78.97% NANs !!!!  Can we deleted it ?
* Columns with NANs wich we can try to impute: User Depth(3.94%), Internet Browser Id (3.94%), City Size (27.01%), Gender(3.94%), Age Group(3.94%), Clicked(15.43%)
* Columns: interested_in_* have zeros becase of binary character, that is ok
* Column: Session ID is unique, it carries no information for modeling.
* Column: DateTime should be used for feature extraction, day, month, hour, miniute, week of the year, day of the year, etc..




```python
# Define column lists
NUMERICAL_COLUMNS = [
    'User Id', 'Campaign Id','Webpage Id', 'Product Category', 'User Depth', 'Internet Browser Id', 'City Size',
    'Interested In Cars', 'Interested In Food', 'Interested In News', 'Interested In Technology', 'Interested In Medicine', 
    'Interested In Politics', 'Interested In Fashion', 'Interested In Astronomy', 'Interested In Animals', 'Interested In Travel'
] 
CATEGORICAL_COLUMNS = ['Product Type', 'Gender', 'Age Group', 'Device Used']
DATE_COLUMNS = ['DateTime']
TARGET_COLUMNS = ['Clicked']
```

## 3.1 Understand Target variable
---


```python
# We take a look on the target variable
plt.figure(figsize=(7, 5))
ax = sns.histplot(
    x='Clicked',
    data=df_train,
)
```


    
![png](output_18_0.png)
    


### Notes:
* Binary classification task
* unbalanced classes, should we generate additional 'yes' classes with SMOTE ?
* metrics should be used w.r.t imbalanced classes

## 3.2 Feature Engineering
### 3.2.1 Preprocess Categorical Columns
---
* Before we can do imputation we need to bring our categorical columns to onehot (pd.get_dummies function) format



```python
# create onehote columnms from categorical values
df_preprocessed = pd.DataFrame()

def generate_dummy_features(df_from: pd.DataFrame, df_to: pd.DataFrame, columns_list: list)-> pd.DataFrame:
    '''
    translate categorical variables to dummy variables    
    '''
    
    for col in columns_list:
        df_to =  pd.concat(
            [
                pd.get_dummies(df_from[col], prefix=col, drop_first=False),
                df_to
            ],axis=1
        )
        
    return df_to

df_preprocessed = generate_dummy_features(df, df_preprocessed, CATEGORICAL_COLUMNS)
```


```python
#df_preprocessed.columns
df_preprocessed.columns = df_preprocessed.columns.str.replace(r'[<,>]', '')
```

### 3.2.2 Preprocess Numerical Columns
---


```python
def copy_numerical_features(df_from: pd.DataFrame, df_to: pd.DataFrame, columns_list: list)-> pd.DataFrame:
    '''
    copy and cast to float numerical variables    
    '''
    for col in columns_list:
        df_to[col] = df_from[col].astype(float).copy()
              
    return df_to

df_preprocessed = copy_numerical_features(df, df_preprocessed, NUMERICAL_COLUMNS)
```

### 3.2.3 Preprocess Date Columns
---


```python
def generate_new_date_features(df_from: pd.DataFrame, df_to: pd.DataFrame, date_columns: list)-> pd.DataFrame:
    '''
    generate new data features
    '''
    
    for col in date_columns:              
        # to year
        #df_to[col+'_year'] = pd.to_datetime(df_from[col], errors='coerce').dt.year
        # to month
        df_to[col+'_month'] = pd.to_datetime(df_from[col], errors='coerce').dt.month
        # to day of year
        df_to[col+'_day'] = pd.to_datetime(df_from[col], errors='coerce').dt.dayofyear
        # to hour
        df_to[col+'_hour'] = pd.to_datetime(df_from[col], errors='coerce').dt.hour
        # to minute
        df_to[col+'_minutes'] = pd.to_datetime(df_from[col], errors='coerce').dt.minute
        
        df_to[col+'_dayofweek'] = pd.to_datetime(df_from[col], errors='coerce').dt.dayofweek
        
        
    return df_to

df_preprocessed = generate_new_date_features(df, df_preprocessed,  DATE_COLUMNS)
```

### 3.2.4 Understand the nature of NANs
---


```python
ax = plt.subplots(figsize=(10,7))
#sns.clustermap(
sns.heatmap(
    df_preprocessed.isna(),
    cmap="YlGnBu",
    cbar_kws={'label': 'Missing Data'}
)
```




    <AxesSubplot:>




    
![png](output_28_1.png)
    


### Notes:
* Column: advertisment Size has 78% of missing values, (will not be included in further processing)
* Column: Session id is uniqe and has no information gain (will not be included in further processing)
* Columns: User Depth, Internet Browser ID, Gender, Age Group, City Size will be imputed through iterative multivariate imputer (sklearn)



```python
# Interative Imputer,  every NAN is considered as function of values in the same row
imputer = IterativeImputer(
    estimator=linear_model.BayesianRidge(),
    n_nearest_features=None,
    imputation_order='ascending'
)

df_preprocessed[:] = imputer.fit_transform(df_preprocessed)
```

### 3.2.5 Generate Interactions of Predictors
---
* Because of limited hardware ressources we do not compute interactions of predictors


```python
def generate_new_product_features(df: pd.DataFrame, date_columns: list)-> pd.DataFrame:
    '''
    generate new product features
    
    '''

    combination_of_date_columns = list(itertools.combinations(date_columns, 2))

    # compute differences in minutes between ALL datetime columns
    for col1, col2 in combination_of_date_columns:
        column_name = f"{col1}_mull_{col2}" 
        df[column_name] = df[col1].astype(int) * df[col2].astype(int)
        
    return df

#df_formated = generate_new_product_features(df_preprocessed, df_preprocessed.columns.tolist())
```

### 3.2.6 Generate aggregated Predictors
---
* Obviously we can aggregate multiple predictors for particular User Id, this will give us more semantic value


```python
df_agg_user_id = df.groupby('User Id').agg(
    unique_session_id=('Session Id', pd.Series.nunique),
    unique_product_type=('Product Type', pd.Series.nunique),
    unique_device_used=('Device Used', pd.Series.nunique),
    unique_campain_id=('Campaign Id', pd.Series.nunique),
    unique_webpage_id=('Webpage Id', pd.Series.nunique),
    unique_product_category=('Product Category', pd.Series.nunique),
    count_product_type=('Product Type', pd.Series.count),
    count_device_used=('Device Used', pd.Series.count),
    count_campain_id=('Campaign Id', pd.Series.count),
    count_webpage_id=('Webpage Id', pd.Series.count),
    count_product_category=('Product Category', pd.Series.count),
)

def get_stats_for_user_id(user_id: int, col: str) -> int:
    return df_agg_user_id.loc[user_id, col]
```


```python
AGG_COLLS= ['unique_session_id', 'unique_product_type', 'unique_device_used', 'unique_campain_id', 'unique_webpage_id', 'unique_product_category',
           'count_product_type', 'count_device_used', 'count_campain_id', 'count_webpage_id', 'count_product_category']

for col in AGG_COLLS:
    df_preprocessed[col] = df_preprocessed['User Id'].parallel_apply(lambda x: get_stats_for_user_id(x, col))

#df_preprocessed.drop('User Id', axis=1, inplace=True)
```

### 3.2.7 Scalling
---
* Actually not necessary as we are going to use xgboost, but anyway.


```python
# We use standart scaler, zero mean an unit standard deviation
ss_scaler = StandardScaler()
df_scaled = ss_scaler.fit_transform(df_preprocessed)
df_scaled = pd.DataFrame(df_scaled, columns=df_preprocessed.columns.tolist())
#df_scaled = df_preprocessed
```


```python
plt.figure(figsize=(25, 8))
ax = df_scaled.boxplot()
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
```




    [Text(1, 0, 'Device Used_Mobile'),
     Text(2, 0, 'Device Used_PC/Laptop'),
     Text(3, 0, 'Age Group_18-24'),
     Text(4, 0, 'Age Group_25-34'),
     Text(5, 0, 'Age Group_35-44'),
     Text(6, 0, 'Age Group_45-54'),
     Text(7, 0, 'Age Group_55-64'),
     Text(8, 0, 'Age Group_18'),
     Text(9, 0, 'Age Group_65'),
     Text(10, 0, 'Gender_F'),
     Text(11, 0, 'Gender_M'),
     Text(12, 0, 'Product Type_A'),
     Text(13, 0, 'Product Type_B'),
     Text(14, 0, 'Product Type_C'),
     Text(15, 0, 'Product Type_D'),
     Text(16, 0, 'Product Type_E'),
     Text(17, 0, 'Product Type_F'),
     Text(18, 0, 'Product Type_G'),
     Text(19, 0, 'Product Type_H'),
     Text(20, 0, 'Product Type_I'),
     Text(21, 0, 'Product Type_J'),
     Text(22, 0, 'User Id'),
     Text(23, 0, 'Campaign Id'),
     Text(24, 0, 'Webpage Id'),
     Text(25, 0, 'Product Category'),
     Text(26, 0, 'User Depth'),
     Text(27, 0, 'Internet Browser Id'),
     Text(28, 0, 'City Size'),
     Text(29, 0, 'Interested In Cars'),
     Text(30, 0, 'Interested In Food'),
     Text(31, 0, 'Interested In News'),
     Text(32, 0, 'Interested In Technology'),
     Text(33, 0, 'Interested In Medicine'),
     Text(34, 0, 'Interested In Politics'),
     Text(35, 0, 'Interested In Fashion'),
     Text(36, 0, 'Interested In Astronomy'),
     Text(37, 0, 'Interested In Animals'),
     Text(38, 0, 'Interested In Travel'),
     Text(39, 0, 'DateTime_month'),
     Text(40, 0, 'DateTime_day'),
     Text(41, 0, 'DateTime_hour'),
     Text(42, 0, 'DateTime_minutes'),
     Text(43, 0, 'DateTime_dayofweek'),
     Text(44, 0, 'unique_session_id'),
     Text(45, 0, 'unique_product_type'),
     Text(46, 0, 'unique_device_used'),
     Text(47, 0, 'unique_campain_id'),
     Text(48, 0, 'unique_webpage_id'),
     Text(49, 0, 'unique_product_category'),
     Text(50, 0, 'count_product_type'),
     Text(51, 0, 'count_device_used'),
     Text(52, 0, 'count_campain_id'),
     Text(53, 0, 'count_webpage_id'),
     Text(54, 0, 'count_product_category')]




    
![png](output_38_1.png)
    


### Notes:
* Age Group <18 has huge outliers after scaling, should we cut it ? 

## 3.3 Feature Selection
---


### 3.3.1 Helper functions (utils.py)


```python
def evaluate_model_on_features(X: pd.DataFrame, Y: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    '''Train model and display evaluation metrics.'''
    
    # Train the model, predict values and get metrics
    acc, prec, rec, f1, roc, _, _, _, _, _ = train_and_get_metrics(X, Y)

    # Construct a dataframe to display metrics.
    display_df = pd.DataFrame([[acc, prec, rec, f1, roc, X.shape[1]]], columns=["Accuracy", "Precision", "Recall", "F1 Score", 'ROC',  'Feature Count'])
    
    return display_df


def train_and_get_metrics(
    X: pd.DataFrame, 
    Y: pd.DataFrame, 
    params: dict = None) -> Tuple[str,str,str,str,str,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,object]:
    '''Train a Random Forest Classifier and get evaluation metrics'''
    
    # Split train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, stratify=Y, random_state = 123)

    # Call the fit model function to train the model on the normalized features and the diagnosis values
    model = fit_model(X_train, Y_train)

    # Make predictions on test dataset and calculate metrics.
    acc, prec, rec, f1, roc = calculate_metrics(model, X_test, Y_test)

    return acc, prec, rec, f1, roc, X_train, X_test, Y_train, Y_test, model


def calculate_metrics(model: object, X_test: pd.DataFrame, Y_test: pd.DataFrame) ->  Tuple[str,str,str,str,str]:
    '''Get model evaluation metrics on the test set.'''
    
    # Get model predictions
    y_predict_r = model.predict(X_test)
    
    # Calculate evaluation metrics for assesing performance of the model.
    roc = roc_auc_score(Y_test, y_predict_r, multi_class='raise', average="weighted")
    acc = accuracy_score(Y_test, y_predict_r, )
    prec = precision_score(Y_test, y_predict_r, average="weighted")
    rec = recall_score(Y_test, y_predict_r, average="weighted")
    f1 = f1_score(Y_test, y_predict_r, average="weighted")
    
    return acc, prec, rec, f1, roc

def fit_model(X: pd.DataFrame, Y: pd.DataFrame, params: dict = None) -> object: 
    '''Use a XGBoost for this problem.'''
    
    num_class = len(Y.unique().tolist()) 
    
    if params:
        model = XGBClassifier(**params)
    else:
        model = XGBClassifier(
            #feature_names=X.columns.tolist(),
            use_label_encoder=False,
            #objective="multi:softmax",
            objective='binary:logistic',
            #eval_metric=["mlogloss"],
            eval_metric='auc',
            #num_class = num_class,
            seed=143,
        )
    
    model.fit(
        X,
        Y,
        verbose=False,
    )
    
    return model

def print_confusion_matrix(X: pd.DataFrame, Y: pd.DataFrame)-> None:
    '''Print confusion Matrix'''
    
    # split and fit model with data
    _, _, _, _, _, _, X_test, _, Y_test, model, = train_and_get_metrics(X, Y)
     
    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_confusion_matrix(
        model,
        X_test,
        Y_test,
        normalize='all',
        colorbar=False,
        ax=ax,
    )
    
def print_classification_report(X: pd.DataFrame, Y: pd.DataFrame, params: dict = None )-> None:
    '''Print confusion Matrix'''
    
    # split and fit model with data
    _, _, _, _, _, _, X_test, _, Y_test, model, = train_and_get_metrics(X, Y, params)
    
    Y_test_pred = model.predict(X_test)
        
    print(
        classification_report(
            Y_test,
            Y_test_pred,
            digits=4,
            
        )
    )

def run_rfe(X: pd.DataFrame, Y: pd.DataFrame, number_features:int)-> list:
    '''
    Compute recursive Feature Elimination
    '''
    
    # split and fit model with data
    _, _, _, _, _, X_train, X_test, Y_train, Y_test, model, = train_and_get_metrics(X, Y)
        
    # Wrap RFE around the model
    rfe = RFE(model, number_features)
    
    # Fit RFE
    rfe = rfe.fit(X_train, Y_train)
    feature_names = X_train.columns[rfe.get_support()]
    
    return feature_names

def plot_xgb_importance(X: pd.DataFrame, Y: pd.DataFrame,  number_features:int)-> None:
    '''
    Compute recursive Feature Elimination
    '''
    
    # split and fit model with data
    _, _, _, _, _, X_train, X_test, Y_train, Y_test, model, = train_and_get_metrics(X, Y)
    
    
    sorted_idx = np.argsort(model.feature_importances_)#[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_importance(
        model,
        max_num_features = number_features,
        ax=ax,
    )

# function for filtering intracorrelated predictors
def filter_intra_correlations(
    df_corr: pd.DataFrame,
    corr_thresshold: float = 0.7,
)-> list:

    corr_cols = []
    for column in df_corr.columns.tolist():
        
        if column not in corr_cols:
            cor_target = abs(df_corr[column])
            cor_features = cor_target[ (cor_target>corr_thresshold) & (cor_target<1 )  ]
            corr_cols.extend(cor_features.index.tolist())
    
    uncorelated_columns = [ x for x in df_corr.columns.tolist() if x not in  corr_cols ]

    return uncorelated_columns
```

### 3.3.2 All predictors - Baseline (XGBoost)
---


```python
# cut test data
df_train_scaled = df_scaled[:df_train.shape[0]]
df_test_scalled = df_scaled[df_train.shape[0]:]
# map target
df_train_target = df_train.loc[:,'Clicked'].map({'No':0,'Yes':1}).reset_index(drop=True)
```

### Notes:
* Important step: we split scalled_df back to train and test


```python
df_results = pd.DataFrame()
df_result = evaluate_model_on_features(
    df_train_scaled, 
    df_train_target,
)
df_result.index = ['All features baseline(xgboost)']

df_results = df_results.append(df_result)
df_results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
      <th>ROC</th>
      <th>Feature Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>All features baseline(xgboost)</th>
      <td>0.931211</td>
      <td>0.874905</td>
      <td>0.931211</td>
      <td>0.898116</td>
      <td>0.500025</td>
      <td>54</td>
    </tr>
  </tbody>
</table>
</div>




```python
print_classification_report(
    df_train_scaled, 
    df_train_target    
)
```

                  precision    recall  f1-score   support
    
               0     0.9313    0.9999    0.9644    109469
               1     0.1111    0.0001    0.0002      8079
    
        accuracy                         0.9312    117548
       macro avg     0.5212    0.5000    0.4823    117548
    weighted avg     0.8749    0.9312    0.8981    117548
    


### Notes:
* lot of False Negatives,  because classes are imbalanced !!!!!
* Before we go futher in feature selection we need to resolve the imbalance problem.

### 3.3.3 Syntetic minority oversampling (SMOTE and ADASYN)
---


```python
# SMOTE
oversample = SMOTE()
df_train_scaled_sm, df_train_target_sm = oversample.fit_resample(
    df_train_scaled,
    df_train_target)
```


```python
# ADASYN
oversample = ADASYN()
df_train_scaled_ada, df_train_target_ada = oversample.fit_resample(
    df_train_scaled,
    df_train_target)
```


```python
# after imputation we should have same amount of both classes
df_train_target_sm.value_counts()
```




    0    364895
    1    364895
    Name: Clicked, dtype: int64




```python
# with SMOTE balanced
df_result = evaluate_model_on_features(
    df_train_scaled_sm, 
    df_train_target_sm,
)
df_result.index = ['All features SMOTE balanced(xgboost)']
df_results = df_results.append(df_result)

# with ADASYN balanced
df_result = evaluate_model_on_features(
    df_train_scaled_ada, 
    df_train_target_ada,
)
df_result.index = ['All features ADASYN balanced(xgboost)']
df_results = df_results.append(df_result)
df_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
      <th>ROC</th>
      <th>Feature Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>All features baseline(xgboost)</th>
      <td>0.931211</td>
      <td>0.874905</td>
      <td>0.931211</td>
      <td>0.898116</td>
      <td>0.500025</td>
      <td>54</td>
    </tr>
    <tr>
      <th>All features SMOTE balanced(xgboost)</th>
      <td>0.962820</td>
      <td>0.965392</td>
      <td>0.962820</td>
      <td>0.962769</td>
      <td>0.962820</td>
      <td>54</td>
    </tr>
    <tr>
      <th>All features ADASYN balanced(xgboost)</th>
      <td>0.963055</td>
      <td>0.965616</td>
      <td>0.963055</td>
      <td>0.963015</td>
      <td>0.963353</td>
      <td>54</td>
    </tr>
  </tbody>
</table>
</div>




```python
print_classification_report(
    df_train_scaled_ada, 
    df_train_target_ada    
)
```

                  precision    recall  f1-score   support
    
               0     0.9307    1.0000    0.9641    109469
               1     1.0000    0.9267    0.9620    111263
    
        accuracy                         0.9631    220732
       macro avg     0.9653    0.9634    0.9630    220732
    weighted avg     0.9656    0.9631    0.9630    220732
    


### 3.3.4 Predictors without intra correlation (corr < 0.75)
---
* We filter predictors which has high intra correlation


```python
#correlation map
df_corr = df_train_scaled_ada.corr()

# Filter intracorrelated features, our dataframe does not contain target column !
UNCORELATED_COLUMNS = filter_intra_correlations(
    df_corr,
    corr_thresshold=0.75
)

# add target to see corelations with our target variable
df_train_target_scaled = pd.concat(
    [df_train_scaled_ada, df_train_target_ada],
    axis=1,
    #ignore_index=True
)

# compute correlation matrix
df_corr = df_train_target_scaled[UNCORELATED_COLUMNS + TARGET_COLUMNS].corr()

# plot correlation map
plt.subplots(figsize=(25, 25))
sns.heatmap(
    df_corr, 
    annot=True, 
    linewidths=.5, 
    fmt= '.1f',
    cmap=plt.cm.PuBu
)
#df_scaled.drop(TARGET_COLUMNS, axis=1, inplace=True)
```




    <AxesSubplot:>




    
![png](output_56_1.png)
    


### Notes:
* Very low correlation amoung predictors,


```python
df_result = evaluate_model_on_features(
    df_train_scaled_ada[UNCORELATED_COLUMNS], 
    df_train_target_ada,
)
df_result.index = ['Uncorrelated(-intra) features(xgboost)']

df_results = df_results.append(df_result)
df_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
      <th>ROC</th>
      <th>Feature Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>All features baseline(xgboost)</th>
      <td>0.931211</td>
      <td>0.874905</td>
      <td>0.931211</td>
      <td>0.898116</td>
      <td>0.500025</td>
      <td>54</td>
    </tr>
    <tr>
      <th>All features SMOTE balanced(xgboost)</th>
      <td>0.962820</td>
      <td>0.965392</td>
      <td>0.962820</td>
      <td>0.962769</td>
      <td>0.962820</td>
      <td>54</td>
    </tr>
    <tr>
      <th>All features ADASYN balanced(xgboost)</th>
      <td>0.963055</td>
      <td>0.965616</td>
      <td>0.963055</td>
      <td>0.963015</td>
      <td>0.963353</td>
      <td>54</td>
    </tr>
    <tr>
      <th>Uncorrelated(-intra) features(xgboost)</th>
      <td>0.963073</td>
      <td>0.965632</td>
      <td>0.963073</td>
      <td>0.963034</td>
      <td>0.963371</td>
      <td>49</td>
    </tr>
  </tbody>
</table>
</div>




```python
print_classification_report(
    df_train_scaled_ada[UNCORELATED_COLUMNS], 
    df_train_target_ada    
)
```

                  precision    recall  f1-score   support
    
               0     0.9307    1.0000    0.9641    109469
               1     1.0000    0.9267    0.9620    111263
    
        accuracy                         0.9631    220732
       macro avg     0.9654    0.9634    0.9630    220732
    weighted avg     0.9656    0.9631    0.9630    220732
    


### 3.3.5 Predictors with Target correlation (Corr > 0.005)
---


```python
# add target to see corelations with our target variable
df_train_target_scaled_ada = pd.concat(
    [df_train_scaled_ada, df_train_target_ada],
    axis=1,
    #ignore_index=True
)

# get correlated collumns
df_corr = df_train_target_scaled_ada.corr()

CORR_THRESHOLD = 0.005
CORRELATED_COLUMNS = df_corr[ df_corr['Clicked'].abs() > CORR_THRESHOLD]['Clicked'].index.tolist()
CORRELATED_COLUMNS.remove('Clicked')
#print(f'Correlated Columns {CORRELATED_COLUMNS}')
```


```python
df_result = evaluate_model_on_features(
    df_train_scaled_ada[CORRELATED_COLUMNS], 
    df_train_target_ada,
)
df_result.index = ['Correlated with target features(xgboost)']

df_results = df_results.append(df_result)
df_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
      <th>ROC</th>
      <th>Feature Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>All features baseline(xgboost)</th>
      <td>0.931211</td>
      <td>0.874905</td>
      <td>0.931211</td>
      <td>0.898116</td>
      <td>0.500025</td>
      <td>54</td>
    </tr>
    <tr>
      <th>All features SMOTE balanced(xgboost)</th>
      <td>0.962820</td>
      <td>0.965392</td>
      <td>0.962820</td>
      <td>0.962769</td>
      <td>0.962820</td>
      <td>54</td>
    </tr>
    <tr>
      <th>All features ADASYN balanced(xgboost)</th>
      <td>0.963055</td>
      <td>0.965616</td>
      <td>0.963055</td>
      <td>0.963015</td>
      <td>0.963353</td>
      <td>54</td>
    </tr>
    <tr>
      <th>Uncorrelated(-intra) features(xgboost)</th>
      <td>0.963073</td>
      <td>0.965632</td>
      <td>0.963073</td>
      <td>0.963034</td>
      <td>0.963371</td>
      <td>49</td>
    </tr>
    <tr>
      <th>Correlated with target features(xgboost)</th>
      <td>0.956907</td>
      <td>0.960351</td>
      <td>0.956907</td>
      <td>0.956842</td>
      <td>0.957254</td>
      <td>39</td>
    </tr>
  </tbody>
</table>
</div>




```python
print_classification_report(
    df_train_scaled_ada[CORRELATED_COLUMNS], 
    df_train_target_ada    
)
```

                  precision    recall  f1-score   support
    
               0     0.9201    1.0000    0.9584    109469
               1     1.0000    0.9145    0.9553    111263
    
        accuracy                         0.9569    220732
       macro avg     0.9600    0.9573    0.9569    220732
    weighted avg     0.9604    0.9569    0.9568    220732
    


### 3.3.6 Recursive Feature Elimination
---
* Find predictors with high influence on target (RFE)


```python
RFE_COLUMNS = run_rfe(
    df_train_scaled_ada, 
    df_train_target_ada,
    20
)
```


```python
df_result = evaluate_model_on_features(
    df_train_scaled_ada[RFE_COLUMNS], 
    df_train_target_ada,
)
df_result.index = ['RFE features(xgboost)']

df_results = df_results.append(df_result)
df_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
      <th>ROC</th>
      <th>Feature Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>All features baseline(xgboost)</th>
      <td>0.931211</td>
      <td>0.874905</td>
      <td>0.931211</td>
      <td>0.898116</td>
      <td>0.500025</td>
      <td>54</td>
    </tr>
    <tr>
      <th>All features SMOTE balanced(xgboost)</th>
      <td>0.962820</td>
      <td>0.965392</td>
      <td>0.962820</td>
      <td>0.962769</td>
      <td>0.962820</td>
      <td>54</td>
    </tr>
    <tr>
      <th>All features ADASYN balanced(xgboost)</th>
      <td>0.963055</td>
      <td>0.965616</td>
      <td>0.963055</td>
      <td>0.963015</td>
      <td>0.963353</td>
      <td>54</td>
    </tr>
    <tr>
      <th>Uncorrelated(-intra) features(xgboost)</th>
      <td>0.963073</td>
      <td>0.965632</td>
      <td>0.963073</td>
      <td>0.963034</td>
      <td>0.963371</td>
      <td>49</td>
    </tr>
    <tr>
      <th>Correlated with target features(xgboost)</th>
      <td>0.956907</td>
      <td>0.960351</td>
      <td>0.956907</td>
      <td>0.956842</td>
      <td>0.957254</td>
      <td>39</td>
    </tr>
    <tr>
      <th>RFE features(xgboost)</th>
      <td>0.962937</td>
      <td>0.965514</td>
      <td>0.962937</td>
      <td>0.962897</td>
      <td>0.963236</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>




```python
print_classification_report(
    df_train_scaled_ada[RFE_COLUMNS], 
    df_train_target_ada    
)
```

                  precision    recall  f1-score   support
    
               0     0.9305    1.0000    0.9640    109469
               1     1.0000    0.9265    0.9618    111263
    
        accuracy                         0.9629    220732
       macro avg     0.9652    0.9632    0.9629    220732
    weighted avg     0.9655    0.9629    0.9629    220732
    


## Notes:
* Dataset consisting only of RFE selected Columns, shows best performance / no. of predictors tradeoff 
* We will use this dataset for further analysis

### 3.3.7 Feature importance
---


```python
# plot feature importance based in xgboost importance scroe
plot_xgb_importance(
    df_train_scaled_ada,
    df_train_target_ada,
    30
)
```


    
![png](output_70_0.png)
    


## Notes:
* aggregated predictors seems to have high importance scores (unique*)

# 4. Modeling
---

### 4.1 Bayesian Optimization of XGBoost
---


```python
dtrain = xgb.DMatrix(df_train_scaled_sm[RFE_COLUMNS], label=df_train_target_sm)
```


```python
def bayesian_optimization_xgb(
    max_depth : int, 
    gamma: int, 
    n_estimators: int,
    learning_rate: int) -> float:
    '''
    Function for bayesian optimization
    '''
    
    # set params for xgb
    params = {
        'max_depth': int(max_depth),
        'gamma': gamma,      
        #'n_estimators': int(n_estimators),
        'learning_rate':learning_rate,
        'subsample': 0.8,
        'eta': 0.1,
        'eval_metric': 'rmse'
    }
    
    #Cross validating with the specified parameters in 5 folds and 10 iterations
    cv_result = xgb.cv(
        params,
        dtrain,
        num_boost_round=10,
        nfold=5
    )    
    
    #Return the negative RMSE
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]
```


```python
# Bayesion optimization function
bo_xgb = BayesianOptimization(
    bayesian_optimization_xgb,
    {
        'max_depth': (3, 10), 
        'gamma': (0, 1),
        'learning_rate':(0,1),
        'n_estimators':(100,120)
    }
)

# Bayesian optimization 8 steps of random exploration and for 5 iterations
bo_xgb.maximize(
    n_iter=5,
    init_points=8,
    acq='ei'
)
```

    |   iter    |  target   |   gamma   | learni... | max_depth | n_esti... |
    -------------------------------------------------------------------------
    | [0m 1       [0m | [0m-0.2355  [0m | [0m 0.553   [0m | [0m 0.3976  [0m | [0m 7.386   [0m | [0m 109.5   [0m |
    | [0m 2       [0m | [0m-0.3207  [0m | [0m 0.2571  [0m | [0m 0.4217  [0m | [0m 3.671   [0m | [0m 104.1   [0m |
    | [0m 3       [0m | [0m-0.2803  [0m | [0m 0.479   [0m | [0m 0.3865  [0m | [0m 4.676   [0m | [0m 103.6   [0m |
    | [0m 4       [0m | [0m-0.279   [0m | [0m 0.1663  [0m | [0m 0.6613  [0m | [0m 4.689   [0m | [0m 116.1   [0m |
    | [95m 5       [0m | [95m-0.2185  [0m | [95m 0.4509  [0m | [95m 0.5415  [0m | [95m 9.802   [0m | [95m 111.9   [0m |
    | [0m 6       [0m | [0m-0.4248  [0m | [0m 0.4803  [0m | [0m 0.07112 [0m | [0m 4.242   [0m | [0m 110.9   [0m |
    | [0m 7       [0m | [0m-0.2342  [0m | [0m 0.1379  [0m | [0m 0.2877  [0m | [0m 8.082   [0m | [0m 105.9   [0m |
    | [0m 8       [0m | [0m-0.2507  [0m | [0m 0.2912  [0m | [0m 0.8715  [0m | [0m 6.302   [0m | [0m 113.2   [0m |
    | [95m 9       [0m | [95m-0.2184  [0m | [95m 0.4692  [0m | [95m 0.5253  [0m | [95m 9.084   [0m | [95m 111.3   [0m |
    | [0m 10      [0m | [0m-0.2194  [0m | [0m 0.4074  [0m | [0m 0.5811  [0m | [0m 9.275   [0m | [0m 109.0   [0m |
    | [0m 11      [0m | [0m-0.2382  [0m | [0m 0.07288 [0m | [0m 1.0     [0m | [0m 8.628   [0m | [0m 113.7   [0m |
    | [0m 12      [0m | [0m-0.2277  [0m | [0m 0.0     [0m | [0m 1.0     [0m | [0m 10.0    [0m | [0m 110.5   [0m |
    | [0m 13      [0m | [0m-0.2308  [0m | [0m 0.8777  [0m | [0m 0.8182  [0m | [0m 8.423   [0m | [0m 102.4   [0m |
    =========================================================================



```python
#Extracting the best parameters
params = bo_xgb.max['params']
print(params)
```

    {'gamma': 0.4692038224630308, 'learning_rate': 0.5252540232010607, 'max_depth': 9.083703833158296, 'n_estimators': 111.32986071644471}



```python
# after bayesian optimization
df_result = evaluate_model_on_features(
    df_train_scaled_ada[RFE_COLUMNS], 
    df_train_target_ada,
    params,
)
df_result.index = ['Bayesian Optimization of RFE predictors (xgboost)']
df_results = df_results.append(df_result)
df_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
      <th>ROC</th>
      <th>Feature Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>All features baseline(xgboost)</th>
      <td>0.931211</td>
      <td>0.874905</td>
      <td>0.931211</td>
      <td>0.898116</td>
      <td>0.500025</td>
      <td>54</td>
    </tr>
    <tr>
      <th>All features SMOTE balanced(xgboost)</th>
      <td>0.962820</td>
      <td>0.965392</td>
      <td>0.962820</td>
      <td>0.962769</td>
      <td>0.962820</td>
      <td>54</td>
    </tr>
    <tr>
      <th>All features ADASYN balanced(xgboost)</th>
      <td>0.963055</td>
      <td>0.965616</td>
      <td>0.963055</td>
      <td>0.963015</td>
      <td>0.963353</td>
      <td>54</td>
    </tr>
    <tr>
      <th>Uncorrelated(-intra) features(xgboost)</th>
      <td>0.963073</td>
      <td>0.965632</td>
      <td>0.963073</td>
      <td>0.963034</td>
      <td>0.963371</td>
      <td>49</td>
    </tr>
    <tr>
      <th>Correlated with target features(xgboost)</th>
      <td>0.956907</td>
      <td>0.960351</td>
      <td>0.956907</td>
      <td>0.956842</td>
      <td>0.957254</td>
      <td>39</td>
    </tr>
    <tr>
      <th>RFE features(xgboost)</th>
      <td>0.962937</td>
      <td>0.965514</td>
      <td>0.962937</td>
      <td>0.962897</td>
      <td>0.963236</td>
      <td>20</td>
    </tr>
    <tr>
      <th>Bayesian Optimization of RFE predictors (xgboost)</th>
      <td>0.962937</td>
      <td>0.965514</td>
      <td>0.962937</td>
      <td>0.962897</td>
      <td>0.963236</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>




```python
print_classification_report(
    df_train_scaled_ada[RFE_COLUMNS], 
    df_train_target_ada,
    params
)
```

                  precision    recall  f1-score   support
    
               0     0.9305    1.0000    0.9640    109469
               1     1.0000    0.9265    0.9618    111263
    
        accuracy                         0.9629    220732
       macro avg     0.9652    0.9632    0.9629    220732
    weighted avg     0.9655    0.9629    0.9629    220732
    


### Notes:
* Quality of predictors should be improved
* Further Analisys of False Negatives is necessary

### 4.2 One Layer perceptron (Deep Learning)
---


```python
# Split train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    df_train_scaled_ada,
    df_train_target_ada,
    test_size = 0.3,
    #stratify=Y,
    random_state = 123)


num_inputs = X_train.shape[1]
num_classes = 1

model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        128,
        activation='relu',
        input_dim=num_inputs,
    ),
    tf.keras.layers.LayerNormalization(axis=-1),
    tf.keras.layers.Dense(num_classes, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy'])

history = model.fit(
    X_train.values,
    Y_train.values,
    batch_size=100,
    epochs=10,
    verbose=1,
    validation_split=0.1
)

Y_pred = model.predict(X_test)
Y_pred = np.argmax(np.round(Y_pred),axis=1)
#Y_test = np.argmax(np.round(Y_test),axis=1)

print(classification_report(
    Y_test,
    Y_pred,
    digits=4,
))
```

    Epoch 1/10
    4636/4636 [==============================] - 11s 2ms/step - loss: 0.5973 - accuracy: 0.6727 - val_loss: 0.5012 - val_accuracy: 0.7557
    Epoch 2/10
    4636/4636 [==============================] - 9s 2ms/step - loss: 0.4452 - accuracy: 0.7923 - val_loss: 0.4083 - val_accuracy: 0.8133
    Epoch 3/10
    4636/4636 [==============================] - 9s 2ms/step - loss: 0.3845 - accuracy: 0.8276 - val_loss: 0.3694 - val_accuracy: 0.8339
    Epoch 4/10
    4636/4636 [==============================] - 9s 2ms/step - loss: 0.3565 - accuracy: 0.8436 - val_loss: 0.3507 - val_accuracy: 0.8511
    Epoch 5/10
    4636/4636 [==============================] - 9s 2ms/step - loss: 0.3381 - accuracy: 0.8540 - val_loss: 0.3255 - val_accuracy: 0.8586
    Epoch 6/10
    4636/4636 [==============================] - 9s 2ms/step - loss: 0.3262 - accuracy: 0.8611 - val_loss: 0.3227 - val_accuracy: 0.8608
    Epoch 7/10
    4636/4636 [==============================] - 10s 2ms/step - loss: 0.3172 - accuracy: 0.8663 - val_loss: 0.3055 - val_accuracy: 0.8741
    Epoch 8/10
    4636/4636 [==============================] - 10s 2ms/step - loss: 0.3107 - accuracy: 0.8696 - val_loss: 0.3183 - val_accuracy: 0.8592
    Epoch 9/10
    4636/4636 [==============================] - 10s 2ms/step - loss: 0.3059 - accuracy: 0.8719 - val_loss: 0.3112 - val_accuracy: 0.8661
    Epoch 10/10
    4636/4636 [==============================] - 9s 2ms/step - loss: 0.3010 - accuracy: 0.8740 - val_loss: 0.3084 - val_accuracy: 0.8659
                  precision    recall  f1-score   support
    
               0     0.4948    1.0000    0.6620    109211
               1     0.0000    0.0000    0.0000    111521
    
        accuracy                         0.4948    220732
       macro avg     0.2474    0.5000    0.3310    220732
    weighted avg     0.2448    0.4948    0.3275    220732
    


### Notes:
* validation accurary: 86% 
* Deep Learning shows poor performance

# 5. Generate CSV output
---


```python
df_probs = pd.DataFrame()

# cut test data
df_test_scalled = df_scaled[df_train.shape[0]:]
df_probs['Session Id'] = df[df_train.shape[0]:]['Session Id']
```


```python
# we fit our model based on all features ADASYN balanced data set
model = fit_model(
    df_train_scaled_ada,
    df_train_target_ada
)
# predict probs of the test dataset
df_probs['Clicked'] =  model.predict_proba(df_test_scalled)[:,1]
```


```python
# print 10 session ids with highest Clicked probabilities
df_probs.sort_values(by='Clicked', ascending=False)[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Session Id</th>
      <th>Clicked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>54714</th>
      <td>4adcc023-5895-425b-9e58-eadd0edc3bdb</td>
      <td>0.813391</td>
    </tr>
    <tr>
      <th>51408</th>
      <td>8085e52a-dd01-45d4-ba8e-d9140a095513</td>
      <td>0.679096</td>
    </tr>
    <tr>
      <th>907</th>
      <td>a9563da7-cab8-4d76-b6cb-dbedca46ec98</td>
      <td>0.629458</td>
    </tr>
    <tr>
      <th>61405</th>
      <td>e0d9436a-e82a-415e-9662-da7e3a3a779d</td>
      <td>0.609394</td>
    </tr>
    <tr>
      <th>14208</th>
      <td>01f6fc70-caf5-4e7f-82dd-b3060a6bda61</td>
      <td>0.608115</td>
    </tr>
    <tr>
      <th>314</th>
      <td>6bc44a1e-21c7-4cad-b22c-6ed509c27dd8</td>
      <td>0.585189</td>
    </tr>
    <tr>
      <th>58916</th>
      <td>a22db18a-ae9a-4845-91d9-a943ad6b02b9</td>
      <td>0.566682</td>
    </tr>
    <tr>
      <th>60875</th>
      <td>fbcaaf6f-7306-462d-bf8f-2823af9c1204</td>
      <td>0.564757</td>
    </tr>
    <tr>
      <th>36076</th>
      <td>25f1346c-b3b1-474e-923b-e4e456e3acf3</td>
      <td>0.550714</td>
    </tr>
    <tr>
      <th>1917</th>
      <td>cbec3249-be1c-4249-a6d4-c267bf183836</td>
      <td>0.545675</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_probs.to_csv('probs.csv')
```


```python

```
