from settings import *

# general
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# general
import argparse


def parse_args():
    # Parse cli arguments
    parser = argparse.ArgumentParser(
        description="""This is the script responsible for running DDStream"""
    )
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")

    optional.add_argument(
        "--topic",
        default="test",
        help="Kafka topic (default 'test')",
    )
    optional.add_argument(
        "-t",
        "--timeout",
        default=10,
        type=int,
        help="Timeout of streaming application (default 10 seconds)",
    )

    args = parser.parse_args()
    topic = args.topic
    timeout = args.timeout
    return topic, timeout


if __name__ == "__main__":

    TOPIC, TIMEOUT = parse_args()

    ssc = SparkSession.builder.appName("ddstream").master("local[*]").getOrCreate()
    ssc = SparkSession.builder.appName("ddstream").getOrCreate()
    # TODO: not sure log level working
    ssc.sparkContext.setLogLevel("WARN")

    # Streaming Query
    # TODO: Maybe using StreamingListener is important (https://spark.apache.org/docs/2.2.0/api/python/pyspark.streaming.html)

    # TODO: Why is this not working?
    # kddSchema = (
    #     StructType()
    #     .add("duration", "float")
    #     .add("src_bytes", "float")
    #     .add("dst_bytes", "float")
    #     .add("wrong_fragment", "float")
    #     .add("urgent", "float")
    #     .add("hot", "float")
    #     .add("num_failed_logins", "float")
    #     .add("num_compromised", "float")
    #     .add("root_shell", "float")
    #     .add("su_attempted", "float")
    #     .add("num_root", "float")
    #     .add("num_file_creations", "float")
    #     .add("num_shells", "float")
    #     .add("num_access_files", "float")
    #     .add("count", "float")
    #     .add("srv_count", "float")
    #     .add("serror_rate", "float")
    #     .add("srv_serror_rate", "float")
    #     .add("rerror_rate", "float")
    #     .add("srv_rerror_rate", "float")
    #     .add("same_srv_rate", "float")
    #     .add("diff_srv_rate", "float")
    #     .add("srv_diff_host_rate", "float")
    #     .add("dst_host_count", "float")
    #     .add("dst_host_srv_count", "float")
    #     .add("dst_host_same_srv_rate", "float")
    #     .add("dst_host_diff_srv_rate", "float")
    #     .add("dst_host_same_src_port_rate", "float")
    #     .add("dst_host_srv_diff_host_rate", "float")
    #     .add("dst_host_serror_rate", "float")
    #     .add("dst_host_srv_serror_rate", "float")
    #     .add("dst_host_rerror_rate", "float")
    #     .add("dst_host_srv_rerror_rate", "float")
    #     .add("cluster", "string")
    # )

    input_df = (
        ssc.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    # print("Printing Schema input_df: ")
    # input_df.printSchema()

    input_df1 = input_df.selectExpr(
        "CAST(key AS STRING)", "CAST(value AS STRING)", "timestamp"
    )

    # Used: https://stackoverflow.com/questions/61324549/how-is-the-string-column-in-dataframe-split-into-multiple-columns-when-spark-str
    ######################## Test Schema ########################
    # testSchema = (
    #     StructType()
    #     .add("col1", StringType())
    #     .add("col2", StringType())
    #     .add("col3", StringType())
    #     .add("col4", StringType())
    #     .add("col5", StringType())
    #     .add("col6", StringType())
    # )
    # raw_df_2 = input_df1.select(from_json(col("value"), testSchema).alias("tablename"))
    # raw_df_3 = raw_df_2.select("tablename.*")

    output_df = (
        input_df1.withColumn("tmp", split(col("value"), ","))
        .withColumn("col1", col("tmp")[0].cast(IntegerType()))
        .withColumn("col2", col("tmp")[1].cast(IntegerType()))
        .withColumn("col3", col("tmp")[2].cast(IntegerType()))
        .withColumn("col4", col("tmp")[3].cast(IntegerType()))
        .withColumn("col5", col("tmp")[4].cast(IntegerType()))
        .withColumn(
            "col6", col("tmp")[5].cast(IntegerType())
        )  # .select("col1","col2","col3","col4","col5","timestamp") # training data
        .select("col1", "col2", "col3", "col4", "col5", "col6", "timestamp")
    )
    ##############################################################

    # TODO: remove cluster column from training stream (keep for validation)
    # output_df = (
    #     input_df1.withColumn("tmp", split(col("value"), ","))
    #     .withColumn("duration", col("tmp")[0].cast(FloatType()))
    #     .withColumn("src_bytes", col("tmp")[1].cast(FloatType()))
    #     .withColumn("dst_bytes", col("tmp")[2].cast(FloatType()))
    #     .withColumn("wrong_fragment", col("tmp")[3].cast(FloatType()))
    #     .withColumn("urgent", col("tmp")[4].cast(FloatType()))
    #     .withColumn("hot", col("tmp")[5].cast(FloatType()))
    #     .withColumn("num_failed_logins", col("tmp")[6].cast(FloatType()))
    #     .withColumn("num_compromised", col("tmp")[7].cast(FloatType()))
    #     .withColumn("root_shell", col("tmp")[8].cast(FloatType()))
    #     .withColumn("su_attempted", col("tmp")[9].cast(FloatType()))
    #     .withColumn("num_root", col("tmp")[10].cast(FloatType()))
    #     .withColumn("num_file_creations", col("tmp")[11].cast(FloatType()))
    #     .withColumn("num_shells", col("tmp")[12].cast(FloatType()))
    #     .withColumn("num_access_files", col("tmp")[13].cast(FloatType()))
    #     .withColumn("count", col("tmp")[14].cast(FloatType()))
    #     .withColumn("srv_count", col("tmp")[15].cast(FloatType()))
    #     .withColumn("serror_rate", col("tmp")[16].cast(FloatType()))
    #     .withColumn("srv_serror_rate", col("tmp")[17].cast(FloatType()))
    #     .withColumn("rerror_rate", col("tmp")[18].cast(FloatType()))
    #     .withColumn("srv_rerror_rate", col("tmp")[19].cast(FloatType()))
    #     .withColumn("same_srv_rate", col("tmp")[20].cast(FloatType()))
    #     .withColumn("diff_srv_rate", col("tmp")[21].cast(FloatType()))
    # .withColumn("srv_diff_host_rate", col("tmp")[22].cast(FloatType()))
    # .withColumn("dst_host_count", col("tmp")[23].cast(FloatType()))
    # .withColumn("dst_host_srv_count", col("tmp")[24].cast(FloatType()))
    # .withColumn("dst_host_same_srv_rate", col("tmp")[25].cast(FloatType()))
    # .withColumn("dst_host_diff_srv_rate", col("tmp")[26].cast(FloatType()))
    # .withColumn("dst_host_same_src_port_rate", col("tmp")[27].cast(FloatType()))
    # .withColumn("dst_host_srv_diff_host_rate", col("tmp")[28].cast(FloatType()))
    # .withColumn("dst_host_serror_rate", col("tmp")[29].cast(FloatType()))
    # .withColumn("dst_host_srv_serror_rate", col("tmp")[30].cast(FloatType()))
    # .withColumn("dst_host_rerror_rate", col("tmp")[31].cast(FloatType()))
    # .withColumn("dst_host_srv_rerror_rate", col("tmp")[32].cast(FloatType()))
    # .withColumn("cluster", col("tmp")[33].cast(StringType()))
    # .select(
    #     "duration",
    #     "src_bytes",
    #     "dst_bytes",
    #     "wrong_fragment",
    #     "urgent",
    #     "hot",
    #     "num_failed_logins",
    #     "num_compromised",
    #     "root_shell",
    #     "su_attempted",
    #     "num_root",
    #     "num_file_creations",
    #     "num_shells",
    #     "num_access_files",
    #     "count",
    #         "srv_count",
    #         "serror_rate",
    #         "srv_serror_rate",
    #         "rerror_rate",
    #         "srv_rerror_rate",
    #         "same_srv_rate",
    #         "diff_srv_rate",
    #         "srv_diff_host_rate",
    #         "dst_host_count",
    #         "dst_host_srv_count",
    #         "dst_host_same_srv_rate",
    #         "dst_host_diff_srv_rate",
    #         "dst_host_same_src_port_rate",
    #         "dst_host_srv_diff_host_rate",
    #         "dst_host_serror_rate",
    #         "dst_host_srv_serror_rate",
    #         "dst_host_rerror_rate",
    #         "dst_host_srv_rerror_rate",
    #         "cluster",
    #     )
    # )

    # Print data type for each column/field
    # for field in output_df.schema.fields:
    #     print(field.name + " , " + str(field.dataType))

    write_stream = (
        output_df.writeStream.trigger(processingTime="5 seconds")
        .outputMode("update")
        .option("truncate", "false")
        .format("console")
        .start()
    )

    write_stream.awaitTermination(TIMEOUT)
    write_stream.stop()  # TODO: not sure what this does, or if ssc.stop() is the correct
    print("Stream Data Processing Application Completed.")
