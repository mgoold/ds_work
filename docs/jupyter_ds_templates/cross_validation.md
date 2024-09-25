---
id: cross_validation
title: Cross Validation with Memory Management
---

# Cross Validation and Memory Management

I recently had a project where my jupyter notebook kept conking (konking?) out because of cross-validation memory issues.  The below code resolved this issue for me.  Two items of interest:
1. I'm using the "enable_categorical=True" functionality on the XGBClassifier model in this example.  This reduces the hassle and memory burden of one-hot encoding's feature explosion.
2. I'm using indices to create a temporary slice of data with each cross validation round, and then deleting it once the model performance is evaluated and stored.  This is the main trick that keeps the memory usage from stacking up.

Code as follows:

## Imports
```
import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import gc # garbage collection
```

```
# Initialize Model
# model = XGBClassifier(n_estimators=10, max_depth=20, enable_categorical=True, verbosity=2, scale_pos_weight=scposwt)

model = XGBClassifier(**params)
# Initialize a StratifiedKfold for n=10 splits/folds
    # not shuffling bc this is time series data
        # because not shuffling, no need to use random_state, which would be redundant

skf = StratifiedKFold(n_splits=10) 

# storage for accuracy scores
accuracy_scores = []

# metrics for evaluation
eval_metric_list=['auc','aucpr']

# using sub set of columns that I validated through shap analysis

X_temp = def_fin[tempkeeplist]
y = def_fin['lead_payment_increase_flag']

X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.2, random_state=42)

for train_index, test_index in skf.split(X_train,y_train): # the split into/creation of indices happens just here

    # every df is temporary so that you're not overwriting your original source df:
    X_train_temp=X_train.iloc[train_index,:] # that is, a list of row indices, across all columns
    X_test_temp=X_train.iloc[test_index,:]
    y_train_temp=y_train.iloc[train_index]
    y_test_temp=y_train.iloc[test_index]

    # Init the model
        # (see above)

    # train and fit the model 
    model.fit(X_train_temp, y_train_temp, eval_set=[(X_test_temp, y_test_temp)])
    model.evals_result()
    
    # eval the model on AUC, AUCPR

#     best_itr=model.best_iteration
    # to my knowledge, you cannot cross validate on multiple metrics in the same pass
#     auc_best=model.evals_result()['validation_0']['auc'][best_itr]
    aucpr_best=model.evals_result()['validation_0']['aucpr'][best_itr]
    accuracy_scores.append(['auc',auc_best,'aucpr',aucpr_best])
    
    # dump the temp items from memory
    
    del X_train_temp 
    del X_test_temp
    del y_train_temp 
    del y_test_temp
    gc.collect()

# eval your final results
accuracy_scores

```
