# thesis
MSc Thesis, University of Patras

# System General:
- Ubuntu v. 20.04.4 LTS:
- 8 core CPU
- 16GB RAM
- Scala version 2.12.10
- Java version 1.8.0_312
- Python version 3.8.10
- server IP: 150.140.193.89

# Installation

1. Install Java (v. 1.8.0_312):   
    `$ sudo apt install openjdk-8-jdk`  
2. Install Python (v. 3.8.10)
3. (optional) Install tmux
4. Install Docker (v. 20.10.14):   
https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04
5. Install Docker Compose (v. 2.4.1):   
https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-compose-on-ubuntu-20-04
6. Install Scala Build Tool (Linux deb):  
 https://www.scala-sbt.org/download.html
7. Download Spark (v. 3.0.2 pre-built for Hadoop 3.2+):  
 `$ curl -SLO https://archive.apache.org/dist/spark/spark-3.0.2/spark-3.0.2-bin-hadoop3.2.tgz`
8. Extract Spark in ~/opt:   
`$ tar -xvf ~/Downloads/spark-3.0.3-bin-hadoop3.2.tgz`
9. Rename Spark:   
`$ mv spark-3.0.3-bin-hadoop3.2/ spark`
10. Add to path (.bashrc) :   
`$ export PATH=$PATH:/home/mageirakos/opt/spark/bin`
11. Download Kafka (Binary for Spark 2.12):  
`$ curl -SLO https://dlcdn.apache.org/kafka/3.1.0/kafka_2.12-3.1.0.tgz`
12. Extract Kafka in ~/opt:  
`$ tar -xvf ~/Downloads/kafka_2.12-3.1.0.tgz `
13. Install wheel :   
`$ pip install wheel`
14. Install Requirements :   
`$ pip install -r requirements.txt`


# Run Spark Cluster
Spark cluster has 1 master and 3 worker nodes.

1. Start doccker compose cluster:   
`$ docker compose up`
    * Stop: `$ docker compose down`
2. Run shell on master node:   
`$ docker ps --format '{{.ID}} {{.Ports}} {{.Names}}'`  
`$ docker exec -it <spark-master-id> /bin/bash`
3. Launch pyspark with 2 cores for each executor (6 in total) and 1GB of RAM:     
`$ pyspark --master spark://localhost:9077 --total-executor-cores 6 --executor-memory 1024m`


# Run Kafka Cluster
* Kafka Quickstart: https://kafka.apache.org/quickstart
* Using Kafka instance connection address: `localhost:9092`


1. Make sure you are in a tmux session because server might disconnect
2. From inside the kafka folder:
    ``` bash
    # Start the ZooKeeper service
    # Note: Soon, ZooKeeper will no longer be required by Apache Kafka.
    $ bin/zookeeper-server-start.sh config/zookeeper.properties
    ```
3. and in another terminal: 
    ``` bash
    # Start the Kafka broker service
    $ bin/kafka-server-start.sh config/server.properties
    ```

Both servers are java applications. If you are trying to kill them but lost the terminal try:  
`$ pgrep java -u mageirakos`  
`$ kill <pid>`


## General commands
* Create topic:   
`$ bin/kafka-topics.sh --create --topic {topic name} --bootstrap-server localhost:9092`
* List topics:   
`$ bin/kafka-topics.sh --list --bootstrap-server localhost:9092`
* Create Kafka Consumer:  
`$ bin/kafka-console-consumer.sh --topic {topic name} --from-beginning --bootstrap-server localhost:9092`
* List Consumers:
`$ bin/kafka-consumer-groups.sh --list --bootstrap-server localhost:9092`
* Deleting Consumer Group:   
`$ bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 --delete --group {consumer group name}`


# Scripts

## Dummy Spark application subscribed to a Kafka topic:
Under `/scripts/test_kafka` there is a kafkaListener.py and kafkaProducer.py
1. Run Kafka server and have a topic "weather" created.
2. Start Spark application subscribed to the weather kafka topic:  
`$ ~/opt/spark/bin/spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.2 scripts/test_kafka/kafkaListener.py`
3. Start Kafka Producer:  
`$ python3 scripts/test_kafka/kafkaProducer.py`

In python you need to specify `--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.2` during deployment. 


# Troubleshoot

* Docker commands not working: `$ su - mageirakos`


# Useful Links

* Spark Structured Streaming Example (Kafka, Spark, Cassandra):   
https://www.youtube.com/watch?v=CGT8v8_9i2g
