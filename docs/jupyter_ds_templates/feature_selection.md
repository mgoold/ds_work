---
id: feature_selection
title: Selecting Features with sklearn, SHAP Values
---

# Feature Selection

Like me, perhaps you've searched around trying to learn which feature selection package is best for your model. Unless the computing and time resources are prohibitive, I just try them all and go with the one that gives me the best results at a given threshold.

I do wonder if I could use something hyperopt to optimize the importance threshold in a range rather than using a specific list of values, but that feels overwrought.  

Anyway, these code snippets detail the process:

## Imports:

```
from xgboost import XGBClassifier # or whatever model you're using.  Assumes your model has this functionality.
import shap # for SHAP feature importance.
```

## Initialize the XGBoost classifier with enable_categorical=True

This part isn't actually necessary for feature selection per se.  But I prefer it to one-hot encoding, and the feature explosion that one-hot engenders.  So it the model option "enable_categorical=True" relates to the below process in the sense that it will give you fewer features to deal in the first place.

```
eval_metric_list=['auc','aucpr'] # this is where you assign the list of metrics you want to evaluate
model = XGBClassifier(n_estimators=10, eval_metric=eval_metric_list, max_depth=20, enable_categorical=True, verbosity=2, scale_pos_weight=scposwt, early_stopping_rounds=10, random_state=42)
model.fit(X_train,y_train)

```

## Using feature_importances from the model you've fit:

```
# getting off the shelf model importances from model
# assumes you've already fit your model.

feature_importance = pd.DataFrame(list(zip(feature_names, model.feature_importances_)),
                                  columns=['col_name','feature_importances_'])
feature_importance.sort_values(by=['feature_importances_'],
                               ascending=False, inplace=True)

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

# evaluate best results per threshold, which will be displayed in the array "thresh_scores"

```

## Using SHAP importance.

### Initialize SHAP

```
import shap

shap.initjs()
explainer = shap.TreeExplainer(model)
```

### Generate shap_values.

```
shap_values=explainer.shap_values(X_test)
```

### Create shap_importance.

```
feature_names = X_train.columns
```

### Structure SHAP values for feature selection.

```
# SHAP values aren't in a very friendly arrangement, so build a dataframe that matches
  # the column names with their respective SHAP values: 
rf_resultX = pd.DataFrame(shap_values, columns = feature_names)

# now from that dataframe, get the average shap value for each column/feature.
vals = np.abs(rf_resultX.values).mean(0)

# now build a df matching those feature names with the average vals:
shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                  columns=['col_name','feature_importance_vals'])

# now sort the vals for easier processing/display
shap_importance.sort_values(by=['feature_importance_vals'],
                               ascending=False, inplace=True)

shap_importance 
```

# Evaluate model peformance per feature importance threshold.


```
# List of thresholds to evaluate.

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

# select best result
thresh_scores
```



