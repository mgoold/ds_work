---
id: jupyter_eda_feature_work
title: Jupyter EDA and Feature Work
---


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

```
scposwt=599571/137322
```


This variable is then plugged into the xgboost model's scale_pos_weight parameter:

```
model = XGBClassifier(n_estimators=10, eval_metric=eval_metric_list, max_depth=20, enable_categorical=True, verbosity=2, scale_pos_weight=scposwt, early_stopping_rounds=10, random_state=42)
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

#### for categorical variables: specify model using categorical=true option in model:

```
# this lets you handle high-cardinality features without one-hot encoding
eval_metric_list=['auc','aucpr']
model = XGBClassifier(n_estimators=10, eval_metric=eval_metric_list, max_depth=20, enable_categorical=True, verbosity=2, scale_pos_weight=scposwt, early_stopping_rounds=10, random_state=42)
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
