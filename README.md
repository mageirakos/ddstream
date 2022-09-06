# ddstream

# System General:
- Ubuntu v. 20.04.4
- 8 core CPU
- 16GB RAM
- Scala version 2.12.10
- Java version 1.8.0_312
- Python version 3.8.10
- server IP: 150.140.193.89


# Run Spark Cluster
Spark cluster has 1 master and 3 worker nodes.

1. Start doccker compose cluster:   
`$ docker compose up`
    * Stop: `$ docker compose down`
2. Connect on master node:   
`$ docker exec -it spark-master /bin/bash`  
2. Submit application  with 2 cores for each executor (6 in total) and 2GB of RAM (you need to be in `/data` of spark-master):  
`$ spark-submit --master spark://spark-master:7077 --total-executor-cores 6 --executor-memory 2048m --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.2 ddstream/run.py`
4. On a different terminal with the venv activated have the kafka producer emiting the stream to the correct topic:  
`$ python3 scripts/kafka_producer.py -s test -r 100 --topic test`

- **Note:** 
    - For code changes you need to copy the updated code in the shared_data folder which should exist in top level of this project. This is the current volume used in the containers `./shared_data:/data`

# References
[1] Xu, L., Ye, X., Kang, K., Guo, T., Dou, W., Wang, W., & Wei, J. (2020).
DistStream: An Order-Aware Distributed Framework for Online-Offline Stream
Clustering Algorithms. 2020 IEEE 40th International Conference on Distributed
Computing Systems (ICDCS), 842–852.
https://doi.org/10.1109/ICDCS47774.2020.00075

[2]Cao, F., Estert, M., Qian, W., & Zhou, A. (2006). Density-Based Clustering over
an Evolving Data Stream with Noise. Proceedings of the 2006 SIAM International
Conference on Data Mining, 328–339. https://doi.org/10.1137/1.9781611972764.29

[3]Aggarwal, C. C., Yu, P. S., Han, J., & Wang, J. (2003). A Framework for
Clustering Evolving Data Streams. In Proceedings 2003 VLDB Conference (pp. 81–
92). Elsevier. https://doi.org/10.1016/B978-012722442-8/50016-1

[4]O’Callaghan, L., Mishra, N., Meyerson, A., Guha, S., & Motwani, R. (2002).
Streaming-data algorithms for high-quality clustering. Proceedings 18th International
Conference on Data Engineering, 685–694.
https://doi.org/10.1109/ICDE.2002.994785s
