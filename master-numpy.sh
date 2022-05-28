#!/bin/bash

export PYSPARK_PYTHON=python3
export PATH=$PATH:/spark/bin

chmod 755 master.sh
/master.sh
