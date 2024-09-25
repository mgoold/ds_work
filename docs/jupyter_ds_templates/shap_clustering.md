---
id: metric_performance_evaluation
title: Pandas Model Perf Metrics for Jupyter
---

# shap_clustering

## Rerun shap on latest model
```
import shap
import umap.umap_ as umap
import seaborn as sns


model.fit(X_test, y_test)
shap.initjs()

explainer = shap.TreeExplainer(model)
shap_values=explainer(X_test)
```

## cook down cluster values with umap two 2 clusters

```
sv_2d = umap.UMAP(
  # n_components = how many clusters to make
  # n_neighbors = how many neighbors t grab when clustering
    # neighbors takes some trial and error
  n_components=2, n_neighbors=200, min_dist=0
).fit_transform(shap_values.values[:])

# this is where you visualize the clusters
df_clust1=pd.DataFrame(sv_2d, columns=["Val_0", "Val_1"])

df_clust1=pd.DataFrame(sv_2d, columns=["Val_0", "Val_1"])
y_test_ary=y_test.to_numpy()
df_clust1['y']=y_test_ary
df_clust1['size']=1

# you have to sort them to get the colors displayed in the order you want
df_clust1=df_clust1.sort_values(by=['y'],ascending=False,)

# for class colours in scatter plots

clist = ["#d45087"] * len(df_clust1)  # pink
for i in range(len(df_clust1)):
    if y[i] == 1:
        clist[i] = "#003f5c"  # blue

sns.scatterplot(data=df_clust1, x="Val_0", y="Val_1", color=clist, sizes="size")
```






