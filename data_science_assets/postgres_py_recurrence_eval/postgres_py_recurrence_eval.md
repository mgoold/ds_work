
# Evaluating Date Period Recurrance with Postgres + Python

The following outlines a process of evaluating the cadence recurring intervals in donation dates -- but it could be used to evaluate recurrence for any date series.  You'd care about this if you were trying to assess the LTV of a recurring regular vs. intermittent donor, for example, or trying to model whether an intermittent donor would become a regular, reliable one.

The user will notice the donation context implied in the cadence evaluated, which I've set for month intervals of 1,2,3,4,6 and 12.  

I prefer to separate code according to function -- generally:
* postgres if it requires joins, unions, window functions
* python for looping and complex structure data work
* pandas + imports for proper modeling
* observable/d3 for viz.

There's more comment in-line on coding choices in what follows.

## Step 1  Raw Data

We're evaluating the cadence of dates on a per-user_id basis.  So you'll need data like this:

| unique\_user\_id |date |
| --- | --- |
| 1 |2023\-07\-22 |
| 1 |2023\-08\-22 |
| 1 |2023\-09\-22 |
| 1 |2023\-10\-22 |
| 1 |2023\-11\-22 |
| 1 |2023\-12\-22 |
| 1 |2024\-01\-22 |
| 1 |2024\-02\-22 |

## Step 2: Calculating Intervals

In my case, I care about the month-inteval between one date an the next, so I'm going to trunc month, do some lead and lag window work, and finally calculate a month interval between the dates.  If you have a sql flavor like redshift or oracle then you do a simple date_diff('month',from,to) calc.  But if all you have is desktop postgres then date_diff calcs are a pain in the neck.  There are lots of postgresql date_diff workarounds, including UDFs; my inline solve is below:


```
drop table if exists donation_dates;

create table donation_dates
as
select
t1.user_id
,t1.donation_date
,row_number() over (partition by t1.user_id order by t1.donation_date) row_number -- convenience function for counting and re-joining after python work is complete

-- for QA; we can discard before we read out to file
,t1.rev_rec_month
,t1.lag_rev_rec_month
,t1.rev_rec_year
,t1.lag_rev_rec_year


,case when t1.donation_year=t1.lag_donation_year then date_part('month',age(t1.donation_month,t1.lag_donation_month))
	when t1.donation_year>t1.lag_donation_year then 
	12-date_part('month',age(t1.lag_donation_month,t1.lag_donation_year)) + -- months in earliest year
	(date_part('year',age(t1.donation_year,t1.lag_donation_year))-1)*12 + -- number of whole years before current year, * 12 months per year 
	date_part('month',age(t1.donation_month,t1.donation_year)) -- number of months in current year
	end months_intvl

from
(
	select
	t1.*
	,lag(donation_year,1) over (partition by user_id order by donation_date) lag_donation_year
	,lag(donation_month,1) over (partition by user_id order by donation_date) lag_donation_month
	from
	(
		select
		t1.user_id
		,t1.donation_date
		,date_trunc('year',t1.donation_date)::date donation_year
		,date_trunc('month',t1.donation_date)::date donation_month
		from mydatetable t1
	) t1
) t1
order by t1.user_id, t1.donation_date
;
```

## Step 3: array_agg and .tsv Readout:

There's lots of ways of evaluating a date sequence; with enough window function work you might do it in postgres alone.  I wanted to think in loops so python made more sense for me.  For such loop programming, I avoided pandas because it seemed to impose an additional layer on python processing for little appreciable benefit.  I had the sense for googling that looping row-wise over an array was more computation-expensive than processing a vector.  That decision is why I'm crunching rows into vectors using array_agg below.

I also looked at outputting to json -- which is easy with row_to_json &c.  But I found that reading the resultant json into python was problematic and ditched it in favor of what I already knew how to do.

Here's the array_agg and read to .tsv:

```
copy
(
	select
	t1.user_id -- we'll thus get one row of lists per user
	,array_agg(t1.months_intvl) months_intvl_arry
	,array_agg(t1.row_number) row_number_arry
	,array_agg(t1.donation_date::date) donor_date_arry -- didn't really need this, as it turns out.
	from donordf_pre t1
	group by 1
	order by 1  
) to 'mypath/mydatefile.tsv'  CSV  DELIMITER E'\t'; -- don't need a header
;
```

## Step 3: Evaluate Date Sequence

Some notes on this step:
* agg_array outputs a string "set" like "{NULL,1,2,3}", so it's necessary to strip it and turn it into a list prior to evaluation.
* other than that, we're just looking backward in the list of date intervals and, depending on our rules, noting in a new vector whether the intervals constitute a repeating pattern, and if so, what that pattern is.  I do this by inserting a "0" (no recurring), or the month interval value (e.g. a value of "3" indicating both that their is a recurring interval and what the recurring period is).  
* once we've created our new vector of recurring period signals, we can write the lot back to another .tsv in a matrix format, and join it to the main set in sql on the user_id and row_number index values.

Anyway, here goes:


```
# Imports
import csv
import json
import pandas as pd 

tempary=[]

# read in the data

path='mypath'
myinfile = 'myinfile.tsv'
myoutfile = 'myfoutile.tsv'

with open(path+myinfile) as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
        tempary.append(row)

## Functions for assessing recurring donation cadence

def assign_reccurance(temp_ary,i):
	templen=len(temp_ary)-1
	tempval=0 # default value

	# don't bother evaluating if there isn't a sequence of at least 3 numbers
		# could have handled this in function that calls this one ...
	if len(temp_ary)>=2:

		# first check: is the month interval in 1,2,3,4 and has this happened for 3 successive periods?
		if i in [1,2,3,4] and i==temp_ary[templen-1] and i==temp_ary[templen-2]:
			
			tempval=i

		# 2nd check: is the month interval =6 and has this happened for 2 successive periods?

		elif i==6 and i==temp_ary[templen-1]: # only requiring sequence of 2
			tempval=i

		# else check if current val is 12-ish
			# I found that annual donors wavered a little around EOY cadence, so I included some ambiguity

		elif i in [11,12,13] and i==temp_ary[templen-1]: # only requiring sequence of 2
			tempval=12 # sometimes interpreting the actual 11 or 13 month interval as 12

		# see notes below on potential enhancements

	return tempval
	

def process_recurrence(rec_array):
	recur_val_array=[]
	rec_val=0
	ary_len=len(rec_array)
	temp_rec_ary=[]

	for i in range(0,ary_len):

		# first recurrence flag value is necessarily 0
		if i==0:
			temp_rec_ary=[0]
		else:
			# otherwise get list up to current place in sequence
			temp_rec_ary=rec_array[:i+1]

		rec_val=assign_reccurance(temp_rec_ary,rec_array[i])

		# add recurring|not recurring flag to list of recurrence signals
		recur_val_array.append(rec_val)

	return recur_val_array


def strip_clean_setstr(tempstr):
	chars_to_remove = ['{', '}']
	tempstr=tempstr.replace('NULL','0')
	tempstr=tempstr.replace('{','')
	tempstr=tempstr.replace('}','')
	templist=tempstr.split(',')
	templist=[int(i) for i in templist]

	return templist

for i in tempary:
	templist=strip_clean_setstr(i[1]) # have to turn "set string {1,3,3}" into actual integer list

	# append results of recurrence evaluation to current row
	i=i.append(process_recurrence(templist))


# list of column names for output file

col_list=['user_id','months_intvl','row_number','recurrence_indx']

# write results out to matrix for rejoining in sql:

with open(path+outfile, 'w', newline='') as tsvfile:
	writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
	# write header row
	writer.writerow(col_list)

	# each row, one per user id, will have several lists of equal length
	for i in tempary:
		i[1]=strip_clean_setstr(i[1])
		i[2]=strip_clean_setstr(i[2])

		# for each item in list, write a new row:
		for k in range(0,len(i[1])):
			try:
				# repeat user_id i[0] down rows
				templist=[i[0],i[1][k],i[2][k],i[4][k]] 
				writer.writerow(templist)
			except:
				print('error at i[0]',i[0],'k',k) 


```

## Step 4: Admire Your Results

Going through all this rigamarole will give you results like these fake ones:

| user\_id |months\_intvl |row\_number |recurrence\_indx |
| --- | --- | --- | --- |
| 1 |0 |1 |0 |
| 1 |1 |2 |0 |
| 1 |1 |3 |0 |
| 1 |1 |4 |1 |
| 1 |1 |5 |1 |
| 1 |1 |6 |1 |
| 1 |1 |7 |1 |
| 1 |1 |8 |1 |

The row_number column gives the original date sequence we processed for this user_id.  Compare the "recurrence_index" to the "months_interval" column, which is the series of month intervals between dates we're evaluating.  The forth value in recurrence_index is the first non-zero value of "1", because that is the first time (per our rules) that this user_id had a series of 3 or more successive identical month intervals -- in this case "1", indicating that they are a monthly donor.

When we join this data back to the main set, we can easily window on the user_id to get the earliest date/tenure at which the user became a regular donor.  In the aggregate, we can begin predicting, for example, change to|from recurring donor status as a function of tenure and the type of recurring interval.

Incidentally we have made the tenure calculation trivial -- this is simply a postgres cumlative sum function of the "months_interval" value.

HTH.










