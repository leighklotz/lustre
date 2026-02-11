# Query Cluster

This project clusters sanitized queries into similarity groups and gives aggregate statistics and one or more sample queries.

# Installation and Quick Start
```bash
$ python --version
Python 3.12.3
$ python3 -m venv .venv
$ . .venv/bin/activate
(.venv) $ pip install -r requirements.txt 
$ . .venv/bin/activate
(.venv) $ python cluster.py --input example-input-queries.csv --output-summary output/summary.csv --output-samples output/cluster-samples.csv
```

## Input `queries.csv` format:
```csv
query,count,runtime,users
"index=web sourcetype=apache_error warn",1434,7,"user1,user3"
"stats count by user",100,5,"user2,user4"
"search index=main sourcetype=access_combined status=404",500,3,"user1,user5"
```

## Summary File Output fields
- `cluster`
- `cluster_size`
- `centroid_query`
- `cluster_total_tunetime`
- `cluster_total_runcount`
- `cluster_all_users`

(`cluster` is -1 for outlier cluster)


## Samples File Output Fields
- `cluster`
- `cluster_size`
- `sample_type`
- `distance_from_centroid`
- `query`

(`sample_type` is one of `median` | `edge` | `centroid`)

## Cluster Summary Output
These are the clusters. There is one sample query per cluster.

```csv
cluster,cluster_size,centroid_query,cluster_total_runtime,cluster_total_runcount,cluster_all_users
-1,2,"stats sum(bytes) as b by host,user",670,8,user13 user21 user3 user4
0,4,index=web sourcetype=apache_error error,2134,15,user1 user20 user25 user3 user6
1,4,index=main sourcetype=syslog error,730,9,user11 user16 user19 user2 user22 user3 user5
2,4,search index=main sourcetype=access_combined status=400,1621,45,AAA BBB user1 user17 user2 user4 user5 user8
3,2,search index=web sourcetype=apache_access status=403,1270,11,user1 user15 user24 user5
4,3,search index=web sourcetype=apache_access status=400,1570,14,user12 user14 user2 user23 user4 user5
5,2,"stats count by user, host",400,9,user2 user4 user7
6,2,stats count(user) by host,370,6,user18 user4 user5 user9
(.venv) klotz@tensor:~/wip/lustre$ cat summary.csv 
cluster,cluster_size,centroid_query,cluster_total_runtime,cluster_total_runcount,cluster_all_users
-1,2,"stats sum(bytes) as b by host,user",670,8,user13 user21 user3 user4
0,4,index=web sourcetype=apache_error error,2134,15,user1 user20 user25 user3 user6
1,4,index=main sourcetype=syslog error,730,9,user11 user16 user19 user2 user22 user3 user5
2,4,search index=main sourcetype=access_combined status=400,1621,45,AAA BBB user1 user17 user2 user4 user5 user8
3,2,search index=web sourcetype=apache_access status=403,1270,11,user1 user15 user24 user5
4,3,search index=web sourcetype=apache_access status=400,1570,14,user12 user14 user2 user23 user4 user5
5,2,"stats count by user, host",400,9,user2 user4 user7
6,2,stats count(user) by host,370,6,user18 user4 user5 user9
```

## Cluster Samples Output
Below are the selected queries from each cluster. There are three sample queries per cluster.

When you specify `--output-samples <filepath>` it outputs some sample
queries from each cluster to a separate file, so you can inspect the
queries extracted to each cluster more closely.


```csv
cluster,cluster_size,sample_type,distance_from_centroid,query
-1,2,centroid,1.4513,"stats sum(bytes) as b by host,user"
-1,2,edge,1.4513,stats sum(bytes) by host
0,4,centroid,0.4100,index=web sourcetype=apache_error error
0,4,edge,0.6737,index=web sourcetype=apache_error warn
0,4,median,0.5722,index=web sourcetype=apache_error debug
1,4,centroid,0.4200,index=main sourcetype=syslog error
1,4,edge,0.6807,index=main sourcetype=syslog warn
1,4,median,0.5992,index=main sourcetype=syslog info
2,4,centroid,0.0947,search index=main sourcetype=access_combined status=400
2,4,edge,0.3763,search index=main sourcetype=access_combined status=5001
2,4,median,0.3629,search index=main sourcetype=access_combined status=404
3,2,centroid,0.1001,search index=web sourcetype=apache_access status=403
3,2,edge,0.1001,search index=web sourcetype=apache_access status=404
4,3,centroid,0.0503,search index=web sourcetype=apache_access status=400
4,3,edge,0.0845,search index=web sourcetype=apache_access status=500
4,3,median,0.0761,search index=web sourcetype=apache_access status=401
5,2,centroid,1.4877,"stats count by user, host"
5,2,edge,1.4877,stats count by user
6,2,centroid,0.3422,stats count(user) by host
6,2,edge,0.3422,stats count(host) by user
(.venv) $ 
```

# Cluster Tuning Parameters
- `--min-cluster-size` is the minimum number of queries per cluster depends on the number of desired clusters depends and query corpus count
- `--num-samples` is not actually tuning but output confirmation: Number of sample queries to show per cluster (default: 3)

# Provenance
- https://docs.google.com/document/d/1P-r0vkVVEiCkKaIO6s2TkrO2h5Q6T5K0cxz0uMm5LM8/edit?tab=t.0
- https://chatgpt.com/share/698b6e90-06f8-800c-824c-9cabc3b926a2
- https://github.com/copilot/c/782cdd8e-fd1c-4ad2-92de-53d0dc1905d7
