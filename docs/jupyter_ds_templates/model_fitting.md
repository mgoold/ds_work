---
id: jupyter_model_fitting
title: Jupyter Model Fitting
---


# Modeling Fitting

## splitting combined dataframe into primary X and y sets:

```
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Evaluate model on multiple metrics

```
# you can plug multiple metrics in a list into the eval_metrics parameter, which can take a list:

eval_metric_list=['auc','aucpr']
model = XGBClassifier(n_estimators=10, eval_metric=eval_metric_list, max_depth=20, enable_categorical=True, verbosity=2, scale_pos_weight=scposwt, early_stopping_rounds=10, random_state=42)
```

## In Logistic Regression:

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
