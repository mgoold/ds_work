---
id: retention_queries
title: Working with Retention Data
---

# Data Assets for This Section Are [Here](https://github.com/mgoold/ds_gallimaufry/tree/main/data_science_assets/retention_queries):
* a jupyter notebook for generating test data
* some example SQL.

# May I Have Your Retention Please: Working with Retention Data

Analyzing retention data generally resolves into two trivial tasks: comparing user counts from one period to another, or breaking out users by tenure.

## Breaking Out by Tenure 

### Finding New Users

Here's an example of breaking out users by tenure.  In this case we're breaking out new users who are in their first month.  We could go on to do this for unique users by other tenure groups.  Depending on your reporting user case, you can do this by breaking out unique metrics as shown here, or with a case statement that covers all cases and unique user_id.  In real life, it is critical to actually make metrics or a case statement for all tenure cases, so that you can QA your cumulative total unique users across all break out metrics vs the total unique metric.  Do this QA even if you don't intend to show all the metrics. 

drop table if exists user_retention;

CREATE TABLE IF NOT EXISTS user_retention 
(
	activity_date timestamp without time zone,
	userid int,
	sales int
);


COPY user_retention 
FROM '/Users/ouonomos/git_repo/redshift_dev/test_data/user_retention_test.csv' DELIMITERS E'\t' 
CSV HEADER
;

-- /* may have your retention please */

-- find users joining in a given month for the first time

select
DATE_TRUNC('month', t1.activity_date) active_month
,count(distinct t1.userid) uusers
,count(distinct case when first_active_month=DATE_TRUNC('month', t1.activity_date) then t1.userid end) new_users
from
(
	SELECT 
	t1.userid
	,t1.activity_date
	,min(DATE_TRUNC('month', t1.activity_date)) over (partition by t1.userid) first_active_month -- note that you have to force an eval
		-- at the user level in order to get their global first activity date
	from user_retention t1
) t1
group by 1 
order by 1 desc
;

### Finding New Users With Day Lag

Here's a variation that evaluates a lag statement to find users with a gap of 3 days.

select
t1.activity_date
,count(distinct t1.userid) uusers
,count(distinct case when DATE_PART('Day', t1.activity_date::TIMESTAMP - t1.lag_activity_date::TIMESTAMP)=4 then t1.userid end) lost_3_days
from 
(
	select
	t1.userid
	,date_trunc('day',t1.activity_date) activity_date
	,lag(date_trunc('day',t1.activity_date),1) over (partition by t1.userid order by t1.activity_date) lag_activity_date
	from user_retention t1
) t1
group by 1
order by 1
;

### Finding Users Who Are In Two Successive Months

A classic.  Notice in this case that we're using a left join.  As the inner joins are identical, you could also do this with a "With CTE" and just write the inner query once.

select
distinct
t1.active_month
,count(distinct t1.userid) unique_users
,count(distinct t2.userid) retained_users 
from
(
	SELECT 
	distinct
	date_trunc('month',t1.activity_date) active_month
	,t1.userid
	from user_retention t1
) t1
left join -- you could also do this without the nesting, like:
	--"left join user_retention t2 on t1.userid=t2.userid and DATE_PART('Month', t1.active_month::TIMESTAMP) - DATE_PART('Month',t2.active_month::TIMESTAMP)=1"
	--but I feel the nesting makes it more readable
(
	SELECT 
	distinct
	date_trunc('month',t1.activity_date) active_month
	,t1.userid
	from user_retention t1
) t2 on t1.userid=t2.userid
and DATE_PART('Month', t1.active_month::TIMESTAMP) - DATE_PART('Month',t2.active_month::TIMESTAMP)=1 -- in other words, they were in the previous month's data set as well
group by 1
order by 1
;
