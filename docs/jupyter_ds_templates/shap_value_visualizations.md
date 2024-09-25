---
id: shap_clustering
title: Shap Clustering Code
---

# shap_value_visualizations

```
explainer = shap.TreeExplainer(model)
shap_values=explainer.shap_values(X_test)

# best summary
shap.summary_plot(shap_values,X_test)

# average shap values per feature
shap.summary_plot(shap_values, max_display=15, show=False, plot_type='bar', feature_names=tempkeeplist)
```
