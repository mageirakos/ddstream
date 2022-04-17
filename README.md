# thesis
MSc Thesis, University of Patras

# System:
- Ubuntu v. 20.04.4 LTS:
- 8 core CPU
- 16GB RAM
- Scala version 2.12.10
- Java version 1.8.0_312
- Python version 3.8.10
- 

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
`$ tar -xvf spark-3.0.3-bin-hadoop3.2.tgz`
9. Rename Spark:   
`$ mv spark-3.0.3-bin-hadoop3.2/ spark`
10. Add to path (.bashrc) :   
`$ export PATH=$PATH:/home/mageirakos/opt/spark/bin`

11. Install wheel :   
`$ pip install wheel`
11. Install Requirements :   
`$ pip install -r requirements.txt`


# Run Spark Cluster
Spark cluster has 1 master and 3 worker nodes.

1. Start doccker compose cluster:   
`$ docker compose up`
2. Run shell on master node:   
`$ docker ps --format '{{.ID}} {{.Names}}'`  
`$ docker exec -it <spark-master-id> /bin/bash`
3. Launch spark-shell with 2 cores for each executor (6 in total) and 1GB of RAM:     
`$ spark-shell --master spark://spark-master:7077 --total-executor-cores 6 --executor-memory 1024m`

# Troubleshoot

Docker commands not working: `$ su - mageirakos`
