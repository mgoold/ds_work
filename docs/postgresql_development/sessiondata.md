---
id: sessiondata
title: Working with Session Data
---

# Working with Session Data

## Related Asssets
At [this link](https://github.com/mgoold/ds_gallimaufry/tree/main/data_science_assets/postgresql_session), you can find SQL, some test data, and a notebook with examples of how to fake session data in python.

# Tutorial/Notes
When we talk about working with session data, we mean systematically partitioning a set of sequenced events according to some characteristics that make the partition globally unique over all the data in the set.  Typically this partition is created with respect to unique user ids and sequences of time stamped events.

"But wait," I hear you yawn, "no one needs to know this.  Website data always comes with a session id now."  Yep.  But in real life, at least my real life, you end up combining user event sequences across platforms. For example, I had to evaluate internal CS behavior across 2 kinds of software that they used in a total workflow.  So you have to roll your own session is across multiple sources of time sequenced event, per userid, in order to answer questions like "how do users go back and forth among tools? Where is the time loss when they do so?".    

For a first real world example, let's look at a basic set of integers:
```
drop table if exists Logs;

create table Logs
(
	log_id int
)
;

insert into Logs
values
(1),
(2),
(3),
(7),
(8),
(10)
;

select * from Logs order by 1;
```
Result:


| log_id |
|---|
| 1 |
| 2 |
| 3 |
| 7 |
| 8 |
| 10 |

Our goal here will to create a unique identifier -- a sessionid, for each continuous set of values.  So you can see that 1-3, 7-8, and 10 are the 3 groups of numbers we'll try and partition.  Thinking about it more, you can see that we want to generate a signal whenever we have an interval between 2 successive numbers that is greater than one.  So the first step is to diff the numbers so that you can evaluate that difference with further code:

```
	select
	t1.log_id
	,t1.log_id-t1.lag_log_id log_diff
	from
	(
		select
		t1.log_id
		,lag(t1.log_id,1) over (order by t1.log_id) lag_log_id
		from Logs t1
	) t1
	order by t1.log_id
```
Result:

| log\_id |log\_diff |
| --- | --- |
| 1 |*NULL* |
| 2 |1 |
| 3 |1 |
| 7 |4 |
| 8 |1 |
| 10 |2 |

OK.  Now we've got values of 1 or >1 depending on whether the intervals are continuous.  We can reduce this to 1s and 0s with a case statement like `case when log_diff!=1 then 1 else 0 end`.  Then, we'll do a running sum on that case statement, so that each time we pass a 1 the value will be incremented:

```
select
t1.*
,sum(case when log_diff!=1 then 1 else 0 end) over (order by t1.log_id) row_num
from
(
	select
	t1.log_id
	,t1.log_id-t1.lag_log_id log_diff
	from
	(
		select
		t1.log_id
		,lag(t1.log_id,1) over (order by t1.log_id) lag_log_id
		from Logs t1
	) t1
	order by t1.log_id
) t1
)
```

| log\_id |log\_diff |session\_id |
| --- | --- | --- |
| 1 |*NULL* |0 |
| 2 |1 |0 |
| 3 |1 |0 |
| 7 |4 |1 |
| 8 |1 |1 |
| 10 |2 |2 |

That'll do it!  You can see that running sum generates a unique partition (a session id) as desired.  
The extension of this concept to a real session id simply involves partitioning on a userid (and whatever other group variables are of interest).  You can see that a diff on event timestamps could be changed to evaluate a rule of 30 minutes, as with the classic user web session.

Here's a robust example, which leverages the example session .csv referenced in the assets session above.

```
drop table if exists test_sessiondata;

CREATE TABLE IF NOT EXISTS test_sessiondata 
(
	userid integer,
	sessionid integer,
	pagetime timestamp without time zone, 
	user_agent_string text
);


COPY test_sessiondata 
FROM '/<mypath>/test_sessiondata.csv' DELIMITERS E'\t' 
CSV HEADER
;

-- /*the sessionizer */
-- / this is a method for creating a unique session id across raw session data/

select
t1.* 
-- postgres doesn't have an ignore nulls option, so you have to use coalesce
-- other thing is to use unbounded preceding, so that it will sum across all rows in the interval, keeping the previous same value across
-- the partition

, sum(coalesce(session_flag,0)) OVER (ORDER BY t1.userid, t1.pagetime RANGE UNBOUNDED PRECEDING) AS sessionid
from 
(
	select
	t1.*
	,case when t1.min_lag>30 then 1 -- here is where you look back to flag based on time interval between events
		when t1.userid<>lag(t1.userid,1) over (order by t1.userid, t1.pagetime) then 1
		end session_flag 
	from
	(
		select
		t1.*
		,DATE_PART('Minute', t1.pagetime::TIMESTAMP - time_lag::TIMESTAMP) min_lag -- subtracting current time from previous row time in minutes
		from
		(
			select
			t1.userid
			,t1.pagetime
			,lag(t1.pagetime,1) over (partition by t1.userid order by t1.pagetime) as time_lag
			from test_sessiondata t1
			order by 1,3
			limit 1000
		) t1
	) t1
) t1
;
```

