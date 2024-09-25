---
id: shap_value_visualizations
title: Shap Value Visualization Code
---

The below are basic common shap visualizations used in the process of feature selection.

## Bi-directional X-axis Clusters 

### best summary
shap.summary_plot(shap_values,X_test)

Hi/low values indicate relation of hi-low value correlated with direction of influence.

```
explainer = shap.TreeExplainer(model)
shap_values=explainer.shap_values(X_test)
```

![Screenshot 2024-09-24 at 21 15 44](https://github.com/user-attachments/assets/31896600-8871-4c59-a0f2-5c53e619d30f)

## Average SHAP Value per Feature, by Descending Importance
### Average value shows average importance of feature.

```
shap.summary_plot(shap_values, max_display=15, show=False, plot_type='bar', feature_names=tempkeeplist)
```

![Screenshot 2024-09-24 at 21 20 02](https://github.com/user-attachments/assets/f540c866-a9e3-4af5-af10-2de1154b70ac)





