# SPL Cluster

This project clusters SPL queries by similarity. You should sanitize your queries first.
You might want to sort your queries by popularity or cost first.

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
- Input: Read queries from a CSV file instead of sample data inline
- Output: Output all the clusters, but instead of all queries in each
cluster, output a sample query for each cluster,
and output the query ID (input row numbers) instead of the full query 
for the rest
