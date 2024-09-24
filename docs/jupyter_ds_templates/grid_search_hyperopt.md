---
id: grid_search_hyperopt
title: "Grid Search Using Hyperopt"
---

# Grid Search With Hyperopt

```
## select intial feature list per feature selection process

threshold=0.03
tempdf=shap_importance[(shap_importance['feature_importance_vals']>=threshold)]

## reselect just to make sure you're working with the right set
tempkeeplist=tempdf['col_name'].tolist()

X_temp = def_fin[tempkeeplist]
y = def_fin['lead_payment_increase_flag']

X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.2, random_state=42)

## create hyperopt "space", which is just the set of ranges of values for parameters

space={ 'eta': hp.uniform("eta", .01, 0.5),
        'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 0,180),
        'reg_alpha' : hp.uniform('reg_alpha', 0,180),
        'reg_lambda' : hp.uniform('reg_lambda', 0,180),
        'subsample' : hp.uniform('subsample', 0.5,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'n_estimators': hp.quniform('n_estimators', 10, 100, 1),
        'seed': 0
    }

## create function to try the hyperparameter values
  ## per hyperopt, thse are selected via bayesian updating for values within the range specified
  ## in the parameter "space":

def hyperparameter_tuning(space):
    
    model = xgb.XGBClassifier(
        eta =space['eta'],
        max_depth = int(space['max_depth']),
        gamma =space['gamma'],
        reg_alpha =space['reg_alpha'],
        reg_lambda =space['reg_lambda'],
        subsample =space['subsample'],
        colsample_bytree =space['colsample_bytree'],
        n_estimators = int(space['n_estimators']),
        enable_categorical=True,
        random_state = np.random.default_rng(0),
        seed = 0,
        eval_metric='aucpr',
        early_stopping_rounds=20,
    )
    
    evaluation = [(X_train, y_train), (X_test, y_test)]
    
    model.fit(X_train, y_train,eval_set=evaluation,          
            verbose=False,
             )
    
    results=model.evals_result()
    avg_aucpr = np.mean(results['validation_0']['aucpr']) # trying to max out precision recall curve area

    print("Avg AUCPR: ", avg_aucpr)
    
    #as I understand it, when an eval_metric is to be maximized, you have to set a maximized * -1, as shown here:
    
    return {'loss': -avg_aucpr, 'status': STATUS_OK, 'model': model}

# use the hyperopt fmin() function to cycle through the selection process

trials = Trials()

best = fmin(fn=hyperparameter_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
            rstate=np.random.default_rng(0),
           )

print (best)

# this is how you get the parameters for the best result:

space_eval(space, best)

```





















