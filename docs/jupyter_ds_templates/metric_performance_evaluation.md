# metric_performance_eval

## Accuracy

### xgboost
```
# Make predictions on the test set
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

## Confusion Matrix

cm = ConfusionMatrix(model)
cm.fit(X_train, y_train)
cm.score(X_test, y_test)
cm.show()
```

## Sensitivity, Specificity
from yellowbrick.classifier import ConfusionMatrix
 
## ROC AUC Curve

```
from yellowbrick.classifier import ROCAUC
visualizer = ROCAUC(model2)

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show() 
```

## PRC AUC

```
# Create the visualizer, fit, score, and show it
viz = PrecisionRecallCurve(model2)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show()
```