# Query Cluster

This project clusters sanitized queries into similarity groups and gives aggregate statistics and a sample query.

## Input `queries.csv` format:
```csv
query,count,runtime,users
"index=web sourcetype=apache_error warn",1434,7,"user1,user3"
"stats count by user",100,5,"user2,user4"
"search index=main sourcetype=access_combined status=404",500,3,"user1,user5"
```

## Output fields
- cluster_id
- cluster centroid query (as representative query )
- total run count
- total run time
- list of users

# Installation
```
$ python --version
Python 3.12.3
$ python3 -m venv .venv
$ . .venv/bin/activate
(.venv) $ pip install -r requirements.txt 
```

# Sample usage
```bash
(.venv) $ python cluster.py --aggregate --input samples.csv
Loading weights: 100%|█| 199/199 [00:00<00:00, 2131.93it/s, Materializing param
cluster,query,runtime,runcount,users
-1,"stats sum(bytes) as b by host,user",670,8,user13 user21 user3 user4
0,index=web sourcetype=apache_error error,2134,15,user1 user20 user25 user3 user6
1,index=main sourcetype=syslog error,730,9,user11 user16 user19 user2 user22 user3 user5
2,search index=main sourcetype=access_combined status=400,1621,45,AAA BBB user1 user17 user2 user4 user5 user8
3,search index=web sourcetype=apache_access status=403,1270,11,user1 user15 user24 user5
4,search index=web sourcetype=apache_access status=400,1570,14,user12 user14 user2 user23 user4 user5
5,"stats count by user, host",400,9,user2 user4 user7
6,stats count(user) by host,370,6,user18 user4 user5 user9
(.venv) $
```

# TODO
- Tuning: depends on query count, number of desired clusters
  
