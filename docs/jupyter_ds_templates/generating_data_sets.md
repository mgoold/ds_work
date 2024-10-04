---
id: generating_datasets
title: Generating Datasets
---

# Generating Datasets

Sometimes you need a set you imagine but isn't ready-to-hand on kaggle or whatever.  Generating them is one of those things that shouldn't be as enjoyable as it is.  It's strangely fun to spin up an imaginary population.

I find most human characteristics are served by normal, beta, or exponential distributions.

You can see that I lead on the device of assigning the observation to a class -- i this case employment status, and then use that class designation as a dictionary lookup that can be referenced to feed separate distributions for each class -- so the mean age for retirees is higher than for those in education, for example.

Hope it helps ...

## Imports


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
```


```python
pd.set_option('display.float_format','{:.0f}'.format) # force display of float columns/override scientific
```

### Generate set of random categorical variables


```python
size=1000

emp_status = ['Retiree', 'Working', 'In_Education', 'Unemployed']
mean_age_map={'Retiree':60, 'Working':30, 'In_Education':20, 'Unemployed':20}
mean_donation_map={'Retiree':7, 'Working':5, 'In_Education':4, 'Unemployed':1}
emp_statuses = [np.random.choice(emp_status) for i in range(0,size)]
sexes = [np.random.choice(['M','F',None],p=[.45,.54,.01]) for i in range(0,size)]
```


```python
emp_statuses[:10]
```




    ['Retiree',
     'In_Education',
     'In_Education',
     'Unemployed',
     'In_Education',
     'Unemployed',
     'In_Education',
     'In_Education',
     'In_Education',
     'Retiree']



### Generate Random Set from Normal Dist


```python
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.normal.html#numpy.random.Generator.normal
# loc=mean, scale = sdev, size = number of items in dist

def get_tenure(emp_status):
    agemean=mean_age_map[emp_status]
    rng = np.random.default_rng()
    tenure = int(np.round(rng.normal(agemean, 5,)))
#     tenure=np.random.Generator.normal(loc=agemean, scale=5)
    return tenure

tenures=[get_tenure(i) for i in emp_statuses] 
```


```python
tenures[:10]
```




    [53, 20, 27, 37, 18, 31, 30, 25, 69, 16]



### Generate Random Set from Beta Dist


```python
rng = np.random.default_rng()
sample = rng.beta(a=100.0, b=20.0, size=1000)

plt.hist([sample], 50, density=True, histtype='bar')
plt.show() # this suppresses the array display
```


![output_12_0](https://github.com/user-attachments/assets/f6c336f6-dae5-4142-91bc-3ddfd7ca3b6c)

#### Interpolate Beta Dist Values into Integer Range 


```python
m = interp1d([0,1],[14,90]) # beta dist on to age range 14-90 years old
agerng=[int(np.round(m(i))) for i in sample]
# agerng
```


```python
def get_donation(emp_status):
    exp_input=mean_donation_map[emp_status]
    rng = np.random.default_rng()
    sample = rng.exponential(scale=10^exp_input, size=1000)
    donation = int(round(np.random.choice(sample)))
    return donation

donations=[get_donation(i) for i in emp_statuses] 
```


```python
fig, ax = plt.subplots()
# override scientific notatioon
ax.ticklabel_format(useOffset=False, style='plain')
plt.hist([donations], 50, density=True, histtype='bar')
plt.show() # this suppresses the array display
```

![output_16_0](https://github.com/user-attachments/assets/e4b3db47-f2b7-45dc-8526-3145644b0f04)


## Combine Lists into Dataframe:


```python
# emp_statuses 
# sexes
# ages
# tenures
# agerng
# donations
donor_data = pd.DataFrame({'emp_status': emp_statuses,'sex':sexes, 'tenure':tenures,'age':agerng,'donation':donations})
```
