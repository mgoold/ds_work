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

#### code to apply function to recode pandas column on per-row basis:

```
df_combined['age_at_donation2'] = df_combined.apply(lambda row: ret_rand(row['age_at_donation']),axis=1)
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





