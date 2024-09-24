---
id: feature_selection
title: Selecting Features with sklearn, SHAP Values
---

# Feature Selection

## Initialize the XGBoost classifier with enable_categorical=True
```
eval_metric_list=['auc','aucpr']
model = XGBClassifier(n_estimators=10, eval_metric=eval_metric_list, max_depth=20, enable_categorical=True, verbosity=2, scale_pos_weight=scposwt, early_stopping_rounds=10, random_state=42)
```

## Using Sklearn

```
# getting off the shelf model importances from sklearn

feature_importance = pd.DataFrame(list(zip(feature_names, model.feature_importances_)),
                                  columns=['col_name','feature_importances_'])
feature_importance.sort_values(by=['feature_importances_'],
                               ascending=False, inplace=True)
feature_importance

# getting thresholds to test

threshold_list=[0.005,0.01,0.02,0.03,0.04,0.05]

thresh_scores=[]

for threshold in threshold_list:
    # select the features above a certain threshold
    
    tempdf=feature_importance[(feature_importance['feature_importances_']>=threshold)]
    tempkeeplist=tempdf['col_name'].tolist()
    
    X_temp = def_fin[tempkeeplist]
    y = def_fin['lead_payment_increase_flag']

    X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    best_itr=model.best_iteration
    auc_best=model.evals_result()['validation_0']['auc'][best_itr]
    aucpr_best=model.evals_result()['validation_0']['aucpr'][best_itr]
    thresh_scores.append([threshold,auc_best,aucpr_best])

thresh_scores

# evaluate best results per threshold
```

## Using SHAP importance

```
# initialize SHAP

import shap

shap.initjs()
explainer = shap.TreeExplainer(model)

# generate shap_values
shap_values=explainer.shap_values(X_test)

# create shap_importance

feature_names = X_train.columns
rf_resultX = pd.DataFrame(shap_values, columns = feature_names)
vals = np.abs(rf_resultX.values).mean(0)
shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                  columns=['col_name','feature_importance_vals'])
shap_importance.sort_values(by=['feature_importance_vals'],
                               ascending=False, inplace=True)

shap_importance

# test per feature importance

threshold_list=[0.005,0.01,0.02,0.03,0.04,0.05]

thresh_scores=[]

for threshold in threshold_list:
    # select the features above a certain threshold
    
    tempdf=shap_importance[(shap_importance['feature_importance_vals']>=threshold)]
    tempkeeplist=tempdf['col_name'].tolist()
    
    print('tempkeeplist',tempkeeplist)
    
    X_temp = def_fin[tempkeeplist]
    y = def_fin['lead_payment_increase_flag']

    X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    best_itr=model.best_iteration
    auc_best=model.evals_result()['validation_0']['auc'][best_itr]
    aucpr_best=model.evals_result()['validation_0']['aucpr'][best_itr]
    thresh_scores.append([threshold,auc_best,aucpr_best])

thresh_scores

# evaluate best result

# get feature list per best threshold
tempdf=shap_importance[(shap_importance['feature_importance_vals']>=threshold)]
```
