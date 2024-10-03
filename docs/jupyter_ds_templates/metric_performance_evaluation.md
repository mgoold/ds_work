---
id: metric_performance_evaluation
title: Pandas Model Perf Metrics for Jupyter
---

# Metric Performance Evaluation

The following code snippets are used for evaluating model performance.  They assume you've already run (fit) your model.

## Accuracy

### xgboost Example
```
# Make predictions on the test set
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

#### Logistic Accuracy
```
# Check Accuracy
y_pred = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)[:,1]
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
# Accuracy .83.  Not bad.  Good enough to move forward for excercise without grid search & cross val.  
    # In real life we would try to nudge it higher with such additional steps.
```


## ROC AUC Curve

### Example 1:

```
from yellowbrick.classifier import ROCAUC
visualizer = ROCAUC(model2)

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show() 
```

### Example 2:

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


## PRC AUC

```
# Create the visualizer, fit, score, and show it
viz = PrecisionRecallCurve(model2)
viz.fit(X_train, y_train)


viz.score(X_test, y_test)
viz.show()
```

## Confusion Matrix

```
from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(model)
cm.fit(X_train, y_train)
cm.score(X_test, y_test)
cm.show()
```

### Classification Scores
```
# Classification Scores
print(classification_report(y_test, y_pred))
```


