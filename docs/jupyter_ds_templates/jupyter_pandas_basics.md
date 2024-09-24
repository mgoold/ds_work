---
id: jupyter_pandas_basics
title: Pandas Basics for Jupyter
---

# jupyter_pandas_basics

## Common Imports

```
import numpy as np
import pandas as pd
from numpy.random import Generator, PCG64
import importlib 
import gc
from pandas.io.formats.info import DataFrameInfo

#widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display

#plots
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from plotnine import *
from mizani import *
from mizani.formatters import percent_format

#stats
import scipy as sp
from scipy import stats
import statsmodels as sm
from scipy import stats
from scipy.stats import beta
import statsmodels as sm
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
import shap
from sklearn.cluster import DBSCAN
import six
import sys
sys.modules['sklearn.externals.six'] = six
from skrules import SkopeRules
import umap.umap_ as umap

# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

# model performance measurement
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

# graphs
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import seaborn.objects as so


# warnings
import warnings
warnings.filterwarnings('ignore')
import collections

# time
from datetime import datetime
```

## Pandas Options to Set
```
pd.set_option('display.max_columns', None) # max columns you can display in .head()
pd.set_option('display.max_rows', None) # max rows you can display in .head()
pd.set_option('display.float_format','{:.0f}'.format) # force display of float columns/override scientific
```

## Reading and Writing Files/dfs

### Read CSV:
```
df.to_csv('/content/drive/MyDrive/marketing_data_modified.csv', index=False)
df = pd.read_csv('/content/drive/MyDrive/marketing_data_modified.csv', low_memory=False, parse_dates=True)
df.Dt_Customer = pd.to_datetime(df.Dt_Customer)
```

### Read Excel:
```
path='/Users/ouonomos/Documents/second_harvest_data/second_harvest_data_science/'
df = pd.read_excel(path+'demographic_data.xlsx')
```

### Writing TSV:
```
df_final.to_csv('finchgroup_df_final.tsv', sep='\t', header=True, index=False)
```

### Writing df from Scratch
```
conv_df = pd.DataFrame({'campaign_id':['A','B'], 'clicks':[conv_days.click_a.sum(),conv_days.click_b.sum()],
                        'conv_cnt':[conv_days.conv_a.sum(),conv_days.conv_b.sum()]})
conv_df['conv_per'] =  conv_df['conv_cnt'] / conv_df['clicks']
conv_df
```

## Cleaning Column Names/String Manipulation

### Cleaning Column Names Using Dictionary:
```
for i in df.columns:
    temp=i.replace(" ", "_")
    temp=temp.replace(":", "")
    temp=temp.replace(".", "")
    temp=temp.replace("\\", "")
    temp=temp.replace("(", "")
    temp=temp.replace(")", "")                  
    temp=temp.lower()
    rendict[i]=temp    

df.rename(columns=rendict, inplace=True)
```

### Rename Pandas columns to lower case
```df.columns = df.columns.str.lower()```

### using apply map

```df = df.applymap(lambda s: s.lower() if type(s) == str else s)```

### Transform Column Data Type

#### Example 1
```df['income'] = df['income'].str.replace(',','').str.replace('$','').astype('float')```

#### Example 2
```
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.lower()
df['crm_vendor']=df['crm_vendor'].astype('str')
df['market_segment']=df['market_segment'].astype('str')
```

## Recoding Columns
### Using Function and Lambda:

#### Example 1:

Note: mapper is generally more reliable.
```
def crm_rec(crm):
    crml=crm.lower()
    crml=crml.replace(' ', '')
    crm2=crm
    if 'salesforce' in crml:
        crm2='Salesforce'
    elif crml in ['nan','unverified','notapplicable','unknown']:
        crm2='Unknown'
    return crm2

df['crm_recode']=df.apply(lambda row: crm_rec(row['crm_vendor']),axis=1)
df['crm_recode'].value_counts().sort_index()
```

#### Example 2:
```
# Using row lambda

df['bad_stage_sig'] = df[['stage','closed','closed_won']].apply(lambda row: 1 if ((row['stage']=='Closed Lost' and row['closed_won']==1) or (row['stage']=='Closed Won' and row['closed_won']==0)) else 0, axis=1)
```

## EDA

### Fill NA
```df[' Income '].fillna(df[' Income '].median(), inplace=True)```

### Scaling Scalars
```
dfsc= df2[scalars].to_numpy()
trans = MinMaxScaler()
dfscfit = trans.fit_transform(dfsc)
scalars=[i+'_sc' for i in scalars]
dfscfin = pd.DataFrame(data=dfscfit,columns=scalars)
```
## handling feature cardinality:

### One Hot Encoding
```
cat_vars=['bus_health_rec','_6sense_rec','geography_rec','market_segment_rec','crm_recode','has_salesforce','has_hubspot','has_funding','industry_rec','emea_na','has_crm']

df2=pd.get_dummies(data=df,columns=cat_vars,dtype=float) # float is more digestible by .logit()
df2['intcpt_dummy']=1 # dummy intercept for logistic regression

# get dummies list out of columns
dummies_list=[i for i in df2.columns.values.tolist() if i not in df.columns.values.tolist()]
```

### specify model using categorical=true option
```
# this lets you handle high-cardinality features without one-hot encoding
eval_metric_list=['auc','aucpr']
model = XGBClassifier(n_estimators=10, eval_metric=eval_metric_list, max_depth=20, enable_categorical=True, verbosity=2, scale_pos_weight=scposwt, early_stopping_rounds=10, random_state=42)
```



### RFE to Eliminate Redundant Variables
```
# initial drop of columns based on RFE above
drop=X_train.columns.values[np.logical_not(rfe.support_.tolist())]
keep=[i for i in X_train.columns.values if i not in drop]
keep=['intcpt_dummy']+keep
keep
```
### Target Variable Imbalance

```df_combined['lead_payment_increase_flag'].value_counts(dropna=False)```

Result:
```
lead_payment_increase_flag
0    599571
1    137322
Name: count, dtype: int64
```

This result is then plugged into a variable with the smaller imbalanced value on the bottom:

```scposwt=599571/137322```

This variable is then plugged into the xgboost model's scale_pos_weight parameter:

```
model = XGBClassifier(n_estimators=10, eval_metric=eval_metric_list, max_depth=20, enable_categorical=True, verbosity=2, scale_pos_weight=scposwt, early_stopping_rounds=10, random_state=42)
```

## Imputation:

### Turning data value_counts into dataframe:
```
df_val_counts = df_combined[(df_combined['age_at_donation']<=90.0)]['age_at_donation'].value_counts().rename_axis('age_at_donation').reset_index(name='counts')

# Normalize data from value counts:

agesum=df_val_counts['counts'].sum()

agelist=[int(i) for i in np.array(df_val_counts['age_at_donation'])]
agewts=np.array(df_val_counts['counts'])
# df_val_counts
agedist=[i/agesum*1.0 for i in agewts]
```

### code to impute values in specified range from function.

```
def ret_rand(val):

    tempval=val
    if (val>=18 and val<=90):
        tempval=int(val)
    elif val>90:
        tempval=np.random.choice(agelist,1,agedist)[0]
    return tempval
```

#### code to apply function to recode pandas column on per-row basis:

```
df_combined['age_at_donation2'] = df_combined.apply(lambda row: ret_rand(row['age_at_donation']),axis=1)
```

## Subsetting and Merging Files Based on Data Type

### Target Variable

```
# target variable:
target = ['lead_payment_increase_flag']
```

### categorical variables for use with xgboost categorical option

```
# cat_variables for use with xgboost categorical option
cat_vars =[] # these variables ill be set to "categorical" in the data frame,
  # these will be paired with with the enable_categorical=True option
```
### scalar variable list
```
scalars=[]
```
### combine target, categorical, scalar variables into set:

```
def_target=df_combined[target]
df_cat_vars=df_combined[cat_vars].astype("category")
def_scalars=df_combined[scalars]
def_fin=def_target.merge(df_cat_vars,left_index=True, right_index=True, how="inner").merge(def_scalars,left_index=True, right_index=True, how="inner")
```

### splitting combined dataframe into primary X and y sets:

```
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### specify model using categorical=true option
```
# this lets you handle high-cardinality features without one-hot encoding
eval_metric_list=['auc','aucpr']
model = XGBClassifier(n_estimators=10, eval_metric=eval_metric_list, max_depth=20, enable_categorical=True, verbosity=2, scale_pos_weight=scposwt, early_stopping_rounds=10, random_state=42)
```

## Evaluate model on multiple metrics
```
# you can plug multiple metrics in a list into the eval_metrics parameter, which can take a list:

eval_metric_list=['auc','aucpr']
model = XGBClassifier(n_estimators=10, eval_metric=eval_metric_list, max_depth=20, enable_categorical=True, verbosity=2, scale_pos_weight=scposwt, early_stopping_rounds=10, random_state=42)
```

## extract model results

```
# Fit the model
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# get the results into an array:

results=model.evals_result()
np.max(results['validation_0']['aucpr'])

best_itr=model.best_iteration
auc_best=model.evals_result()['validation_0']['auc'][best_itr]
aucpr_best=model.evals_result()['validation_0']['aucpr'][best_itr]

```


## Plotting/Graphing/Xtab

### Lineplots
#### Line Plot From Subsetted DF
```
rates_df = conv_days[['click_day','cumu_rate_a','cumu_rate_b']]
rates_df.plot(x='click_day', y=['cumu_rate_a','cumu_rate_b'], figsize=(10,5))
plt.show()
```
#### Example 2
```
df_xtab1['max_const_tot_pay_amount_ntile10_isrecc']=df_xtab1.apply(lambda row: ntile_rec_pay(row['max_const_tot_pay_amount_ntile10'],row['is_recc_donor']),axis=1)
plt.figure(figsize=(16,12))
sns.lineplot(data=df_xtab1[(df_xtab1['max_const_tot_pay_amount_ntile10']>2)], x="cuml_months_tenure", y="cum_total_donations", hue="max_const_tot_pay_amount_ntile10_isrecc")
```

### Xtab
#### Example 1
```
tam_xtab1 = (
    df[(df['stage_group']=='Open') & (df['has_good_stage_signal']==True)].pivot_table(
        index=['emea_na','pred_won_or_lost','final_appeal_score',], columns=['stage_group','has_good_stage_signal'], values='account_name', 
        aggfunc=np.count_nonzero, margins=True, 
        margins_name='Total', fill_value=0
    )
#     .sort_index(axis=0, ascending=False)
#     .pipe(lambda d: d.div(d['Total'], axis='index'))
#     .applymap('{:.0%}'.format)
)

tam_xtab1
```
#### Example 2
```
# crosstab of median and total cumulative giving by total gift tier, 

crosstab1=pd.crosstab([df_combined.max_const_tot_pay_amount_ntile10,df_combined.is_recc_donor], [df_combined.cuml_months_tenure], values=df_combined.cuml_months_payment, aggfunc=[sum])
```
#### Unstacking Crosstab
```
crosstab1.stack(0)
```
#### Formatting Crosstab
```
crosstab1.style.format('{:,}')
```


## Modeling

### Recursive Fit Based on P Values
```
# from initial run (not shown), a secondary drop of columns based on p values:
keep2=[i for i in keep if i not in ['emp_ct_fin_sc','arr_value_sc','zi_past_2_year_employee_growth_rate_sc','bus_health_rec_9 Unknown',
                                    '_6sense_rec_Target','crm_recode_Insight.ly','crm_recode_Oracle CRM','crm_recode_SAP CRM','crm_recode_SugarCRM']]
# X_keep=X[keep2]

# everyone's examples, e.g.: 
# https://dadataguy.medium.com/logistic-regression-using-statsmodels-a63e7944de76
# have argument order for Logit(y,X) as y,X, which is confusing -- bc it's an entire other code base/api

# get readout of p-vals, etc
logit_model=sm.Logit(df_mod[y_list],df_mod[keep2])
result=logit_model.fit(maxiter=35)
print(result.summary2())
```
### Fit Model for Logistic Regression
```
# fit the model
X_train, X_test, y_train, y_test = train_test_split(df_mod[keep2], df_mod[y_list], test_size=0.3, random_state=0)

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # argument order for .fit() is .fit(X, y)
logreg = LogisticRegression()
logreg.fit(X_train, y_train.values.ravel())
```

### Accuracy

#### Logistic Accuracy
```
# Check Accuracy
y_pred = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)[:,1]
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
# Accuracy .83.  Not bad.  Good enough to move forward for excercise without grid search & cross val.  
    # In real life we would try to nudge it higher with such additional steps.
```

#### xgboost accuracy
```
predictions = model.predict(X_test)

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

## Model Performance Metrics
### Confusion Matrix
```
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
```
### Classification Scores
```
# Classification Scores
print(classification_report(y_test, y_pred))
```
### ROC
```
# ROC 
# Keeping nicely away from the 50/50 line
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
```
### 






