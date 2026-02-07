# SPL Cluster

This project clusters SPL queries by similarity. You should sanitize your queries first.

# Installation
```
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

Tuning.
