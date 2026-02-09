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

```
$ python fields.py 
Cluster -1:
- stats sum(bytes) as b by host, user
- stats sum(bytes) by host
Cluster 0:
- index=web sourcetype=apache_error WARN
- index=web sourcetype=apache_error INFO
- index=web sourcetype=apache_error ERROR
- index=web sourcetype=apache_error DEBUG
Cluster 1:
- index=main sourcetype=syslog WARN
- index=main sourcetype=syslog ERROR
- index=main sourcetype=syslog DEBUG
- index=main sourcetype=syslog INFO
Cluster 2:
- search index=main sourcetype=access_combined status=404
- search index=main sourcetype=access_combined status=400
- search index=main sourcetype=access_combined status=500
- search index=main sourcetype=access_combined status=401
Cluster 3:
- search index=web sourcetype=apache_access status=403
- search index=web sourcetype=apache_access status=404
Cluster 4:
- search index=web sourcetype=apache_access status=400
- search index=web sourcetype=apache_access status=401
- search index=web sourcetype=apache_access status=500
Cluster 5:
- stats count by user
- stats count by user, host
Cluster 6:
- stats count(user) by host
- stats count(host) by user
(.venv) $ 

```

# TODO
- Tuning: depends on query count, number of desired clusters
- Output:
-- ID (input row number) of all queries in cluster
-- one representative query for each cluster
-- sum of run times for each cluster
-- sum of run counts for each cluster
-- set of users for each cluster
  
# How to run the code:

## `queries.csv` format:
```csv
query,count,num_users,users
"index=web sourcetype=apache_error warn",1434,7,"user1,user3"
"stats count by user",100,5,"user2,user4"
"search index=main sourcetype=access_combined status=404",500,3,"user1,user5"
```

## run
```bash
python cluster_spl.py --input queries.csv
