from settings import *
from model import DDStreamModel

# general
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *


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


def split_data(streaming_df, database="nsl-kff"):
    #TODO: Find out how to densevector
    # from pyspark.mllib.linalg import VectorUDT, Vectors
    # from pyspark.ml.feature import VectorAssembler

    # TODO: Test correctness
    # TODO: Currently features is ArrayType -> chekc if we need numpy array
    if database == "nsl-kdd":
        num_cols = 33
    elif database == "test":
        num_cols = 5

    # def toDense(vs):
    #     return udf(lambda vs: Vectors.dense(vs), VectorUDT())

    # arraytovector = udf(lambda vs: Vectors.dense(vs), VectorUDT())
    
    # TODO: Find out how to turn it into densevector:
    # possible solutions:
    # - https://stackoverflow.com/questions/68959072/pyspark-how-to-convert-a-string-created-from-a-dense-vector-back-to-a-dense-v
    # - https://stackoverflow.com/questions/49832877/adding-a-vectors-column-to-a-pyspark-dataframe
    # - VectorAssembler: https://stackoverflow.com/questions/32982425/encode-and-assemble-multiple-features-in-pyspark
    # - https://stackoverflow.com/questions/57522319/store-densevector-in-dataframe-column-in-pyspark
    # - https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.linalg.Vectors.html
    # - https://stackoverflow.com/questions/39025707/how-to-convert-arraytype-to-densevector-in-pyspark-dataframe

    data = (
        streaming_df.withColumn("tmp", split(col("value"), ",")) #.map(lambda vals: [float(x) for x in vals])
        .withColumn("label", col("tmp").getItem(num_cols))
        .withColumn("features", slice(col("tmp"), 1, num_cols))
        # .withColumn("other", arraytovector(from_json('value',"array<string>")))
        # .withColumn("other", toDense(streaming_df.tmp))
    )

    # data = VectorAssembler(inputCols=["features"], outputCol="other").transform(data).select("label", "features", "other")

    return data


def get_training_data_exploded(streaming_df, database="nsl-kdd"):
    """Return streaming dataframe of training data based on selected database"""
    if database == "nsl-kdd":
        training_data = (
            streaming_df.withColumn("tmp", split(col("value"), ","))
            .withColumn("duration", col("tmp")[0].cast(FloatType()))
            .withColumn("src_bytes", col("tmp")[1].cast(FloatType()))
            .withColumn("dst_bytes", col("tmp")[2].cast(FloatType()))
            .withColumn("wrong_fragment", col("tmp")[3].cast(FloatType()))
            .withColumn("urgent", col("tmp")[4].cast(FloatType()))
            .withColumn("hot", col("tmp")[5].cast(FloatType()))
            .withColumn("num_failed_logins", col("tmp")[6].cast(FloatType()))
            .withColumn("num_compromised", col("tmp")[7].cast(FloatType()))
            .withColumn("root_shell", col("tmp")[8].cast(FloatType()))
            .withColumn("su_attempted", col("tmp")[9].cast(FloatType()))
            .withColumn("num_root", col("tmp")[10].cast(FloatType()))
            .withColumn("num_file_creations", col("tmp")[11].cast(FloatType()))
            .withColumn("num_shells", col("tmp")[12].cast(FloatType()))
            .withColumn("num_access_files", col("tmp")[13].cast(FloatType()))
            .withColumn("count", col("tmp")[14].cast(FloatType()))
            .withColumn("srv_count", col("tmp")[15].cast(FloatType()))
            .withColumn("serror_rate", col("tmp")[16].cast(FloatType()))
            .withColumn("srv_serror_rate", col("tmp")[17].cast(FloatType()))
            .withColumn("rerror_rate", col("tmp")[18].cast(FloatType()))
            .withColumn("srv_rerror_rate", col("tmp")[19].cast(FloatType()))
            .withColumn("same_srv_rate", col("tmp")[20].cast(FloatType()))
            .withColumn("diff_srv_rate", col("tmp")[21].cast(FloatType()))
            .withColumn("srv_diff_host_rate", col("tmp")[22].cast(FloatType()))
            .withColumn("dst_host_count", col("tmp")[23].cast(FloatType()))
            .withColumn("dst_host_srv_count", col("tmp")[24].cast(FloatType()))
            .withColumn("dst_host_same_srv_rate", col("tmp")[25].cast(FloatType()))
            .withColumn("dst_host_diff_srv_rate", col("tmp")[26].cast(FloatType()))
            .withColumn("dst_host_same_src_port_rate", col("tmp")[27].cast(FloatType()))
            .withColumn("dst_host_srv_diff_host_rate", col("tmp")[28].cast(FloatType()))
            .withColumn("dst_host_serror_rate", col("tmp")[29].cast(FloatType()))
            .withColumn("dst_host_srv_serror_rate", col("tmp")[30].cast(FloatType()))
            .withColumn("dst_host_rerror_rate", col("tmp")[31].cast(FloatType()))
            .withColumn("dst_host_srv_rerror_rate", col("tmp")[32].cast(FloatType()))
            .withColumn("cluster", col("tmp")[33].cast(StringType()))
            .select(
                "duration",
                "src_bytes",
                "dst_bytes",
                "wrong_fragment",
                "urgent",
                "hot",
                "num_failed_logins",
                "num_compromised",
                "root_shell",
                "su_attempted",
                "num_root",
                "num_file_creations",
                "num_shells",
                "num_access_files",
                "count",
                "srv_count",
                "serror_rate",
                "srv_serror_rate",
                "rerror_rate",
                "srv_rerror_rate",
                "same_srv_rate",
                "diff_srv_rate",
                "srv_diff_host_rate",
                "dst_host_count",
                "dst_host_srv_count",
                "dst_host_same_srv_rate",
                "dst_host_diff_srv_rate",
                "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate",
                "dst_host_serror_rate",
                "dst_host_srv_serror_rate",
                "dst_host_rerror_rate",
                "dst_host_srv_rerror_rate",
                # "cluster", # This is the label, don't select on training data
            )
        )
    elif database == "test":
        training_data = (
            streaming_df.withColumn("tmp", split(col("value"), ","))
            .withColumn("col1", col("tmp")[0].cast(IntegerType()))
            .withColumn("col2", col("tmp")[1].cast(IntegerType()))
            .withColumn("col3", col("tmp")[2].cast(IntegerType()))
            .withColumn("col4", col("tmp")[3].cast(IntegerType()))
            .withColumn("col5", col("tmp")[4].cast(IntegerType()))
            .withColumn("col6", col("tmp")[5].cast(IntegerType()))
            .select("col1", "col2", "col3", "col4", "col5")  # , "col6", "timestamp")
        )
    return training_data


def print_field_dtypes(streaming_df):
    for field in streaming_df.schema.fields:
        print(field.name + " , " + str(field.dataType))
    return


def main():
    pass


if __name__ == "__main__":

    TOPIC, TIMEOUT = parse_args()

    # ssc = SparkSession.builder.appName("ddstream").master("local[*]").getOrCreate()
    ssc = SparkSession.builder.appName("ddstream").getOrCreate()
    # TODO: not sure this works correctly
    ssc.sparkContext.setLogLevel("WARN")
    # All local files need to be added like so:
    ssc.sparkContext.addPyFile('ddstream/model.py')

    # Streaming Query
    # TODO: Maybe using StreamingListener is important (https://spark.apache.org/docs/2.2.0/api/python/pyspark.streaming.html)

    input_df = (
        ssc.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    input_df1 = input_df.selectExpr(
        "CAST(key AS STRING)", "CAST(value AS STRING)", "timestamp"
    )

    # TODO: Might need to change the input to be (key, Vector(doubles)): https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.linalg.DenseVector.html
    # Dense vectors are simply represented as NumPy array objects, so there is no need to covert them for use in MLlib
    #

    # database options : [ 'test', 'nsl-kdd' ]
    data = split_data(input_df, database="test")
    training_data = data.select("features") # ArrayType<string>

    # test a broadcast variable
    # broadcasted_var = ssc.sparkContext.broadcast(('a','b','c'))
    # model = DDStreamModel(broadcasted_var, ssc.sparkContext)

    training_data = get_training_data_exploded(input_df, database="test")

    write_stream = (
        training_data.writeStream.trigger(processingTime="5 seconds")
        .outputMode("update")
        .option("truncate", "false")
        .format("console")
        # .foreach(model.run) #TODO: I probably want foreachBatch() to follow batch approach
        # .foreachBatch(model.run_batch)
        .start()
    )

    write_stream.awaitTermination(TIMEOUT)
    print("Stream Data Processing Application Completed.")
    write_stream.stop()
