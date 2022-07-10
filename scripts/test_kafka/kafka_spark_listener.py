from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

kafka_topic_name = "test"
kafka_bootstrap_servers = "localhost:9092"

if __name__ == "__main__":
    ssc = SparkSession.builder.appName("test").master("local[*]").getOrCreate()

    ssc.sparkContext.setLogLevel("ERROR")

    # Streaming Query
    weather_df = (
        ssc.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
        .option("subscribe", kafka_topic_name)
        .option("startingOffsets", "latest")
        .load()
    )

    print("Printing Schema from test topic: ")
    weather_df.printSchema()

    weather_df1 = weather_df.selectExpr(
        "CAST(key AS STRING)", "CAST(value AS STRING)", "timestamp"
    )

    weather_write_stream = (
        weather_df1.writeStream.trigger(processingTime="5 seconds")
        .outputMode("update")
        .option("truncate", "false")
        .format("console")
        .start()
    )

    weather_write_stream.awaitTermination()
    print("Stream Data Processing Application Completed.")
