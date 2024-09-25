---
id: shap_clustering
title: Shap Clustering Code
---

# Shap Clustering

Shap clustering is potentially a terrific tool for discerning distinctions between the features of predicted classes.  It has 2 strengths:
1. You are working toward distinguishing the features of the target classes, rather than performing unsupervised clustering which may give insight about target class distinctions only accidentally.
2. All of the values used in SHAP clustering have already been rationalized in terms of their feature influence on the SHAP value.  This avoids the hassles, present for other clustering approaches, of normalization and reconciling scalar and categorical variables.

I've found in using that:
1. It's a good idea that might not pan out, especially if the clusters you get have an ambiguous mix of target classes in the clusters.  What you're hoping for is that each class makes its own little island, but that doesn't always happen.
2. When you have a "random seeds, then grab k-nearest neighbors" approach like this, you're going to get some drift in number and shape of clusters as you re-run.  This can be a conversational rabbit hole for a stakeholder.

## Code:

### Imports

```
import shap
from sklearn.cluster import DBSCAN
import six
import sys
sys.modules['sklearn.externals.six'] = six
from skrules import SkopeRules
import umap.umap_ as umap
```

### Rerun shap on latest model

```
# fit your model
model.fit(X_test, y_test)
shap.initjs()

# get your SHAP values
explainer = shap.TreeExplainer(model)
shap_values=explainer(X_test)
```

## Cook down SHAP values with umap into 2 clusters (or number of target classes).

```
sv_2d = umap.UMAP(
  # n_components = how many clusters to make
  # n_neighbors = how many neighbors t grab when clustering
    # neighbors takes some trial and error
  n_components=2, n_neighbors=200, min_dist=0
).fit_transform(shap_values.values[:])
```

# Visualize the Clusters

```
df_clust1=pd.DataFrame(sv_2d, columns=["Val_0", "Val_1"])

df_clust1=pd.DataFrame(sv_2d, columns=["Val_0", "Val_1"])
y_test_ary=y_test.to_numpy()
df_clust1['y']=y_test_ary
df_clust1['size']=1

# you have to sort them to get the colors displayed in the order you want
df_clust1=df_clust1.sort_values(by=['y'],ascending=False,)
```

### For class colours in scatter plots

```
clist = ["#d45087"] * len(df_clust1)  # pink
for i in range(len(df_clust1)):
    if y[i] == 1:
        clist[i] = "#003f5c"  # blue

sns.scatterplot(data=df_clust1, x="Val_0", y="Val_1", color=clist, sizes="size")
```

### Example Result

![Screenshot 2024-09-24 at 22 17 22](https://github.com/user-attachments/assets/c8c92027-83e4-4c73-8818-f5b0d111f2df)

## Unsupervised Cluster Search on SHAP Values

This approach will find clusters on your SHAP values that may or may not have good segregation w/respect to
your target variable classes.

### Identify clusters using DBSCAN

You have to play around with the min_samples parameter, which acts like k-nearest neighbors, before you get
a viable set of clusters.

```
# 
# Identify clusters using DBSCAN

sv_2d_labels = DBSCAN(eps=.8, min_samples=20).fit(sv_2d).labels_

# what this gives you is a number label you can put to each cluster
# but it's only useful to do so if you have good separation of clusters by classes
    # (see below for illustration where it's not adding value)
```

### Check Count and Centroid of Clusters
    
```
# for each cluster identified, check count and central coordinates of cluster

for cluster, count in pd.Series(sv_2d_labels).value_counts().sort_index().items():
    print(cluster, count, sv_2d[sv_2d_labels == cluster].mean(0))

# result:
0 70727 [12.521754   1.6203772]
1 59253 [-3.4949746  1.0239966]
2 2919 [-1.1435567  7.969726 ]
3 1026 [  3.799107 -10.437068]
```

### Organize Data with Labels and Target Values

```
# Set Palette for Colors
cpallette=sns.color_palette("Set2",20).as_hex()

def get_color(indx):
    return cpallette[int(indx)]


df_clust1=pd.DataFrame(sv_2d, columns=["Val_0", "Val_1"])
y_test_ary=y_test.to_numpy()
df_clust1['y']=y_test_ary
df_clust1['clust_labl']=sv_2d_labels
df_clust1['labl_col']=df_clust1.apply(lambda x: get_color(x['clust_labl']), axis=1)
df_clust1['size']=1

# you have to sort it according to the color you want "on top"

df_clust1=df_clust1.sort_values(by=['y'],ascending=False,)

clist2=df_clust1['labl_col'].tolist()
```



