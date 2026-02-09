# SPL Cluster

This project clusters SPL queries by similarity. You should sanitize your queries first.

Cluster sanitized_query into similarity groups.

## Output fields
- query_cluster_id (cluster identifier)
- One cluster_centroid_query or representative_query [TODO]

# Installation
```
$ python --version
Python 3.12.3
$ python3 -m venv .venv
$ . .venv/bin/activate
$ pip install -r requirements.txt 
```

# Example Run
(.venv) $ python fields.py   --aggregate --input samples.csv
Loading weights: 100%|█| 199/199 [00:00<00:00, 1447.87it/s, Materializing param=pooler.dense.
cluster,query,runtime,runcount,users
-1,"stats sum(bytes) as b by host, user",670,8,"""user3 ""user4 user13"" user21"""
0,index=web sourcetype=apache_error error,700,8,"""""""user1 ""user1 ""user3 user20"" user25"" user6"""
1,index=main sourcetype=syslog error,730,9,"""user2 ""user3 ""user5 user11"" user16"" user19"" user22"""
2,search index=main sourcetype=access_combined status=400,1621,45,"""AAA ""user1 ""user2 ""user4 BBB"" user17"" user5"" user8"""
3,stats count by user,400,9,"""user2 user4"" user7"""
4,stats count(user) by host,370,6,"""user4 ""user5 user18"" user9"""
5,search index=web sourcetype=apache_access status=400,1570,14,"""user2 ""user4 ""user5 user12"" user14"" user23"""
6,search index=web sourcetype=apache_access status=404,1270,11,"""user1 ""user5 user15"" user24"""
(.venv) $ 
```

# TODO
- Tuning: depends on query count, number of desired clusters
  
## `queries.csv` format:
```csv
query,count,runtime,users
"index=web sourcetype=apache_error warn",1434,7,"user1,user3"
"stats count by user",100,5,"user2,user4"
"search index=main sourcetype=access_combined status=404",500,3,"user1,user5"
```

## Sample usage
```bash
$ python cluster.py --aggregate --input samples.csv
(.venv) $ python fields.py --aggregate --input samples.csv
Loading weights: 100%|█| 199/199 [00:00<00:00, 1621.57it/s, Materializing param=pooler.dense.
cluster,query,runtime,runcount,users
-1,"stats sum(bytes) as b by host,user",670,8,user13 user21 user3 user4
0,index=web sourcetype=apache_error error,2134,15,"""user1 user1 user20 user25 user3 user6"
1,index=main sourcetype=syslog error,730,9,user11 user16 user19 user2 user22 user3 user5
2,search index=main sourcetype=access_combined status=400,1621,45,AAA BBB user1 user17 user2 user4 user5 user8
3,search index=web sourcetype=apache_access status=403,1270,11,user1 user15 user24 user5
4,search index=web sourcetype=apache_access status=400,1570,14,user12 user14 user2 user23 user4 user5
5,"stats count by user, host",400,9,user2 user4 user7
6,stats count(user) by host,370,6,user18 user4 user5 user9
```
