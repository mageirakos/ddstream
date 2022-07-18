from settings import *
from model import DDStreamModel

# general
import argparse
from pyspark.sql import SparkSession

# from pyspark.sql import udf
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.linalg import DenseVector, Vectors, VectorUDT


def parse_args():
    # Parse cli arguments
    parser = argparse.ArgumentParser(
        description="""This is the script responsible for running DDStream"""
    )
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")

    optional.add_argument(
        "-d",
        "--dataset",
        default="test",
        help="Dataset input (default 'test')",
    )
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
    input_data = args.dataset
    topic = args.topic
    timeout = args.timeout
    return input_data, topic, timeout


def split_data(streaming_df, database="nsl-kdd"):
    # print("STEP 0: In split_data")
    """Change the input stream to be (label, Vector<features>)"""
    if database == "nsl-kdd":
        num_feats = 33
    elif database == "test":
        num_feats = 5
    elif database in ["init_toy", "toy"]:
        num_feats = 2

    def get_features(arr):
        """
        Get features from input array and cast them as float.

        features : first n-1 columns of the array
        label    : last column in the aray
        """
        # print("STEP 1: In get_features")
        res = []
        for i in range(len(arr) - 1):
            res.append(float(arr[i]))
        # print(f"\n\nfeatures: {res}\n\n")
        return res

    # it was DenseVector from Akis but I changed it to VectorUDT() because of https://stackoverflow.com/questions/49623620/what-type-should-the-dense-vector-be-when-using-udf-function-in-pyspark
    dense_features = udf(lambda arr: Vectors.dense(get_features(arr)), VectorUDT())

    split_df = streaming_df.select(split(streaming_df["value"], ",").alias("array"))

    # print(f"SPLIT_DF:{split_df}")
    result = split_df.withColumn(
        "label", split_df["array"].getItem(num_feats)
    ).withColumn("features", dense_features(split_df["array"]))
    # print("STEP 2: Leaving split_data")
    return result


def print_field_dtypes(streaming_df):
    for field in streaming_df.schema.fields:
        print(field.name + " , " + str(field.dataType))
    return


if __name__ == "__main__":

    INPUT_DATA, TOPIC, TIMEOUT = parse_args()

    ssc = SparkSession.builder.appName("ddstream").getOrCreate()
    # TODO: not sure this works correctly
    ssc.sparkContext.setLogLevel("WARN")
    # TODO: All local files need to be added like so:
    ssc.sparkContext.addPyFile("ddstream/model.py")

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
    print("\n\n")

    # database options : [ 'test', 'nsl-kdd', 'toy', 'init_toy']
    data = split_data(input_df, database=INPUT_DATA)
    training_data = data.select("label", "features")

    # Default:
    # test a broadcast variable
    broadcasted_var = ssc.sparkContext.broadcast(("a", "b", "c"))
    print(f"START broadcast: {broadcasted_var} {broadcasted_var.value}")

    model = DDStreamModel(broadcasted_var=broadcasted_var)

    # TODO: Decide what to do with the initDBSCAN
    initialDataPath = f"./data/init_toy_dataset.csv"  # must be path in container file tree (shared volume)
    # TODO: set initialEpsilon to 0.02 (after we standardise)
    initialEpsilon = 0.5
    model.initDBSCAN(ssc, initialEpsilon, initialDataPath)

    training_data_stream = (
        training_data.writeStream.trigger(processingTime="5 seconds")
        .outputMode("update")
        .option("truncate", "false")
        .format("console")
        # .foreachBatch(model.run)
        .start()
    )

    training_data_stream.awaitTermination(TIMEOUT)  # end of stream
    print(f"END broadcast: {broadcasted_var} \t {broadcasted_var.value}")
    print("Stream Data Processing Application Completed.")
    training_data_stream.stop()
