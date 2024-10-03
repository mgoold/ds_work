---
id: jupyter_visualization_xtabs
title: Jupyter Xtabs and Visualization
---


### Xtab

#### Example 1

```
tam_xtab1 = (
    df[(df['stage_group']=='Open') & (df['has_good_stage_signal']==True)].pivot_table(
        index=['emea_na','pred_won_or_lost','final_appeal_score',], columns=['stage_group','has_good_stage_signal'], values='account_name', 
        aggfunc=np.count_nonzero, margins=True, 
        margins_name='Total', fill_value=0
    )
#     .sort_index(axis=0, ascending=False)
#     .pipe(lambda d: d.div(d['Total'], axis='index'))
#     .applymap('{:.0%}'.format)
)

tam_xtab1
```
#### Example 2
```
# crosstab of median and total cumulative giving by total gift tier, 

crosstab1=pd.crosstab([df_combined.max_const_tot_pay_amount_ntile10,df_combined.is_recc_donor], [df_combined.cuml_months_tenure], values=df_combined.cuml_months_payment, aggfunc=[sum])
```

#### Unstacking Crosstab
```
crosstab1.stack(0)
```

#### Formatting Crosstab
```
crosstab1.style.format('{:,}')
```
