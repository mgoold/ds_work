---
id: donor_data
title: >
  Donor Data: A Kaggle Classic
---


# Donor Data: A Kaggle Classic

Here's some work I did for an interview process.  The question wasn't really "can you do multivariate data"? But more along the lines of "what would you break out for a quick id of potential donor populations" and "what other data might you bring in"?


```python
# imports

import pandas as pd
import openpyxl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
from sklearn import preprocessing as prep
from faker import Faker
```

# Source Material

https://medium.com/swlh/interpreting-linear-regression-through-statsmodels-summary-4796d359035a
https://cooldata.files.wordpress.com/2018/01/cool-data-a-handbook-for-predictive-modeling.pdf

Original Set At:
https://www.kaggle.com/datasets/michaelpawlus/fundraising-data/code


```python
def process_strdollars(val):
    val=float(val.replace('$','').replace(',',''))
    return val
```


```python
def conditional_log(val):
    if val!=0:
        val=np.log(val)
    return val
```


```python
def impute_gender(row):
    val=0
    
    if row['Prefix is Mr.']==1:
        val=1
    elif row['Prefix is Ms.']==1:
        val=2
    elif row['Prefix is Mrs.']==1:
        val=2
    
    return val
```


```python
def rec_mar_status(val):
    retval=val
    
    if pd.isna(val)==True:
        retval='NA'
    elif val not in ('M','U','S'):
        retval='O'
    return retval
```

# Retrieving Data

I doctored it in postgres so that I could do some window function work on the log of lhc.
As noted in the https://cooldata.files.wordpress.com/2018/01/cool-data-a-handbook-for-predictive-modeling.pdf ...
taming the lhc outlers by taking the log is a good move.
But we can also then do a decile on that log, can call it "lhc level", which will be a lot easier for a stakeholder to digest.

I also grouped the honorifics (e.g. "is Mr.") columns into an "imputed gender" category on the hunch that this would gather more explanatory power and was the analytic point of those categories anyway.  After which I dropped the honorific columns.


```python
# retrieving sql output...

df = pd.read_csv('/Users/ouonomos/Documents/cool_unv_data2.tsv', sep='\t')

```

# A Super Quick Intuition

## The Graduate Year Concept

If you were to look at the book noted above, you'd find that "year graduated" is in inverse correlation to the log of lhc, which makes sense ... you will have less money to donate if you haven't graduated, or haven't been earning as long as a graduate.  I turned "year graduated" into "years since grad" variable by subtracting it from 2022.

Now you see a positive relationship pictured below.  But it's suspicious ... a great bulk of the data points are very old.  We're going to have to be better informed about the meaning of this variable, and qualify it per some "still alive" or "years left to donate" principle.


```python
plt.scatter(df['years_since_grad'],df['log_lhc'], alpha=0.5)
plt.show()
```


    
![png](donor_data_files/donor_data_10_0.png)
    


Let's do some work with seaborn to cap the years after graduation to <=30 and eliminate 0 level donors.  Let's also Make the donor levels stand out:


```python
# n.b. :
# https://stackoverflow.com/questions/31594549/how-to-change-the-figure-size-of-a-seaborn-axes-or-figure-level-plot
# https://seaborn.pydata.org/generated/seaborn.move_legend.html
# https://stackoverflow.com/questions/26139423/plot-different-color-for-different-categorical-levels

df2=df.loc[(df['years_since_grad']<=30) & (df['lhc_clean']>0)]

sns.set(rc={'figure.figsize':(9,5)})
ax=sns.scatterplot(x='years_since_grad', y='log_lhc', data=df2, hue='lhc_level', ec=None, palette="viridis")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
```


    
![png](donor_data_files/donor_data_12_0.png)
    


This is quite suggestive.  Around 16 years we begin to get donors in the higher deciles.  Also, we can see that in every range of years after graduation there is potential to move donors from level to another.  Remember that they are coming into the set every year; it isn't a only static population aging from the left hand of the grid and getting wealthier.

## Continuing to Work Graduate Year Concept...

Now let's break it down to illustrate a common quick way to get at potential opportunities.

The idea is to breakout levels of giving (here, quartiles of log LHC) by other explanatory categories, and look for areas of overlap.  For example, there is overlap in years since grad between log_lhc_quart where the spouse_id is present, with respect to min-max range of years_since_grad. 

What we're aiming for is something like "These donors are in the same earning window (years since grad) and have the same marital status, but are at a lower giving tier.  Maybe we should approach them."  This is a basic routine you would extend for additionally fine grain, and for different combinations of variables, according to your intuition.

Note that the main finding is that we can increase giving by causing the lower tiers to get married.  Just kidding.


```python
# n.b. df2 is the frame filtered on years_since_grad where donations are >0

table = pd.pivot_table(df2, values=['years_since_grad'], index=['log_lhc_qrt', 'spouseid_present'],
                       aggfunc={'years_since_grad': ["min", "max"]})
table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">years_since_grad</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>max</th>
      <th>min</th>
    </tr>
    <tr>
      <th>log_lhc_qrt</th>
      <th>spouseid_present</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <th>0</th>
      <td>23</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <th>0</th>
      <td>23</td>
      <td>20</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3</th>
      <th>0</th>
      <td>30</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>10</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">4</th>
      <th>0</th>
      <td>30</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



# On to Regression Review

The choice of spouseid_present in the table above was informed by a review of correlation and regression on log_lhc below.  Again, this is straight out of the accompanying book for the data set and is pretty much what I'd do for a start.  


# A Spot of Feature Engineering ...

Doing some onehot on marital for the greatest frequencies of marital status after bucketing the small freq cats.




```python
# recode for greatest frequency, keeping NaN
df['marital_status_rec']=df['marital_status'].apply(rec_mar_status)

df = pd.get_dummies(df, columns=['marital_status_rec'], dtype=int)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_number</th>
      <th>lifetimehc</th>
      <th>email_present</th>
      <th>busphone_present</th>
      <th>grad_year</th>
      <th>marital_status</th>
      <th>spouseid_present</th>
      <th>jobtitle_present</th>
      <th>varsityath_present</th>
      <th>studgovt_present</th>
      <th>...</th>
      <th>years_since_grad</th>
      <th>imputed_gender</th>
      <th>log_lhc</th>
      <th>lhc_level</th>
      <th>log_lhc_qrt</th>
      <th>marital_status_rec_M</th>
      <th>marital_status_rec_NA</th>
      <th>marital_status_rec_O</th>
      <th>marital_status_rec_S</th>
      <th>marital_status_rec_U</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1053807</td>
      <td>$0.25</td>
      <td>1</td>
      <td>0</td>
      <td>2001</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>21</td>
      <td>1</td>
      <td>-0.60206</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1053757</td>
      <td>$0.25</td>
      <td>0</td>
      <td>0</td>
      <td>2001</td>
      <td>U</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>21</td>
      <td>1</td>
      <td>-0.60206</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1078367</td>
      <td>$0.00</td>
      <td>1</td>
      <td>0</td>
      <td>2003</td>
      <td>M</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>19</td>
      <td>1</td>
      <td>0.00000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1023190</td>
      <td>$0.00</td>
      <td>1</td>
      <td>0</td>
      <td>1979</td>
      <td>M</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>43</td>
      <td>0</td>
      <td>0.00000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1053899</td>
      <td>$0.00</td>
      <td>0</td>
      <td>0</td>
      <td>2001</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>21</td>
      <td>2</td>
      <td>0.00000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
df.columns
```




    Index(['id_number', 'lifetimehc', 'email_present', 'busphone_present',
           'grad_year', 'marital_status', 'spouseid_present', 'jobtitle_present',
           'varsityath_present', 'studgovt_present', 'otherstudacts_present',
           'greek_present', 'prefix_is_mr', 'prefix_is_ms', 'prefix_is_dr',
           'prefix_is_mrs', 'lhc_clean', 'years_since_grad', 'imputed_gender',
           'log_lhc', 'lhc_level', 'log_lhc_qrt', 'marital_status_rec_M',
           'marital_status_rec_NA', 'marital_status_rec_O', 'marital_status_rec_S',
           'marital_status_rec_U'],
          dtype='object')



# Reviewing Correlation ...


```python


corr_eval_columns=['email_present', 'busphone_present',
       'spouseid_present', 'jobtitle_present',
       'varsityath_present', 'studgovt_present', 'otherstudacts_present',
       'greek_present','prefix_is_dr','years_since_grad','imputed_gender',
       'log_lhc','marital_status_rec_M',
       'marital_status_rec_NA', 'marital_status_rec_O', 'marital_status_rec_S',
       'marital_status_rec_U']
sns.heatmap(df[corr_eval_columns].corr());
```


    
![png](donor_data_files/donor_data_21_0.png)
    



```python
df[corr_eval_columns].corr()[['log_lhc']].sort_values(by='log_lhc', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>log_lhc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>log_lhc</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>years_since_grad</th>
      <td>0.441914</td>
    </tr>
    <tr>
      <th>marital_status_rec_M</th>
      <td>0.369081</td>
    </tr>
    <tr>
      <th>spouseid_present</th>
      <td>0.337671</td>
    </tr>
    <tr>
      <th>busphone_present</th>
      <td>0.262517</td>
    </tr>
    <tr>
      <th>email_present</th>
      <td>0.240848</td>
    </tr>
    <tr>
      <th>jobtitle_present</th>
      <td>0.181078</td>
    </tr>
    <tr>
      <th>marital_status_rec_O</th>
      <td>0.141047</td>
    </tr>
    <tr>
      <th>prefix_is_dr</th>
      <td>0.122937</td>
    </tr>
    <tr>
      <th>greek_present</th>
      <td>0.118937</td>
    </tr>
    <tr>
      <th>studgovt_present</th>
      <td>0.103421</td>
    </tr>
    <tr>
      <th>varsityath_present</th>
      <td>0.060106</td>
    </tr>
    <tr>
      <th>otherstudacts_present</th>
      <td>0.032727</td>
    </tr>
    <tr>
      <th>marital_status_rec_NA</th>
      <td>-0.040020</td>
    </tr>
    <tr>
      <th>marital_status_rec_S</th>
      <td>-0.089273</td>
    </tr>
    <tr>
      <th>imputed_gender</th>
      <td>-0.096891</td>
    </tr>
    <tr>
      <th>marital_status_rec_U</th>
      <td>-0.371626</td>
    </tr>
  </tbody>
</table>
</div>



# Basic Multivar Regression

Exactly like the book.  This is never a bad exploration. Interestingly, my one-hot on marital_status_rec_M gives me more correlation, but a little less r2, presumably because spouseid_present is more general.


```python
x_explain=df[['years_since_grad','spouseid_present','busphone_present','email_present']]
x_explain = sm.add_constant(x_explain)
ks = sm.OLS(df['log_lhc'],x_explain)
ks_res =ks.fit()
print(ks_res.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                log_lhc   R-squared:                       0.330
    Model:                            OLS   Adj. R-squared:                  0.329
    Method:                 Least Squares   F-statistic:                     614.4
    Date:                Tue, 21 Nov 2023   Prob (F-statistic):               0.00
    Time:                        09:44:08   Log-Likelihood:                -7594.4
    No. Observations:                5000   AIC:                         1.520e+04
    Df Residuals:                    4995   BIC:                         1.523e+04
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    const               -0.4651      0.041    -11.443      0.000      -0.545      -0.385
    years_since_grad     0.0310      0.001     32.542      0.000       0.029       0.033
    spouseid_present     0.8431      0.043     19.406      0.000       0.758       0.928
    busphone_present     0.3428      0.034     10.072      0.000       0.276       0.410
    email_present        0.5405      0.033     16.442      0.000       0.476       0.605
    ==============================================================================
    Omnibus:                       90.868   Durbin-Watson:                   0.547
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               63.012
    Skew:                           0.161   Prob(JB):                     2.08e-14
    Kurtosis:                       2.555   Cond. No.                         114.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


# Other Directions

Having made a pass at the above set, what other data might I bring in?  There are two potential complementary directions that I'm sure are obvious to everyone:

1. Variables that focus the users demographically.  So for example user zipcodes, neighborhoods, etc.  The idea is to refine them so that you can group scalars like lifetime hc and disposable income and see how much of an outlier a user is relative to their group.  A high outlier + low donor level = better prospect.  
2. Scalars like putative disposable income.  This is gnarly to do yourself; you're likely to need a vendor. Again, these complement your demographic grouping categorical variables.
