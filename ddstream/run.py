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
    print("STEP 0: In split_data")
    ''' Change the input stream to be (label, Vector<features>)'''
    if database == "nsl-kdd":
        num_feats = 33
    elif database == "test":
        num_feats = 5
    elif database in ["init_toy", "toy"]:
        num_feats = 2

    def get_features(arr):
        '''
        Get features from input array and cast them as float.
        
        features : first n-1 columns of the array
        label    : last column in the aray
        '''
        print("STEP 1: In get_features")
        res = []
        for i in range(len(arr) - 1):
            res.append(float(arr[i]))
        print(f"\n\nfeatures: {res}\n\n")
        return res

    # it was DenseVector from Akis but I changed it to VectorUDT() because of https://stackoverflow.com/questions/49623620/what-type-should-the-dense-vector-be-when-using-udf-function-in-pyspark
    
    dense_features = udf(lambda arr: Vectors.dense(get_features(arr)), VectorUDT())
    # dense_features = udf(lambda arr: Vectors.dense(arr[:-1]), VectorUDT())

    split_df = streaming_df.select(split(streaming_df["value"], ",").alias("array"))

    print(f"SPLIT_DF:{split_df}")
    result = split_df.withColumn("label", split_df["array"].getItem(num_feats)).withColumn(
        "features", dense_features(split_df["array"])
    )
    print("STEP 2: Leaving split_data")
    return result


# def get_training_data_exploded(streaming_df, database="nsl-kdd"):
#     """Return streaming dataframe of training data based on selected database"""
#     if database == "nsl-kdd":
#         training_data = (
#             streaming_df.withColumn("tmp", split(col("value"), ","))
#             .withColumn("duration", col("tmp")[0].cast(FloatType()))
#             .withColumn("src_bytes", col("tmp")[1].cast(FloatType()))
#             .withColumn("dst_bytes", col("tmp")[2].cast(FloatType()))
#             .withColumn("wrong_fragment", col("tmp")[3].cast(FloatType()))
#             .withColumn("urgent", col("tmp")[4].cast(FloatType()))
#             .withColumn("hot", col("tmp")[5].cast(FloatType()))
#             .withColumn("num_failed_logins", col("tmp")[6].cast(FloatType()))
#             .withColumn("num_compromised", col("tmp")[7].cast(FloatType()))
#             .withColumn("root_shell", col("tmp")[8].cast(FloatType()))
#             .withColumn("su_attempted", col("tmp")[9].cast(FloatType()))
#             .withColumn("num_root", col("tmp")[10].cast(FloatType()))
#             .withColumn("num_file_creations", col("tmp")[11].cast(FloatType()))
#             .withColumn("num_shells", col("tmp")[12].cast(FloatType()))
#             .withColumn("num_access_files", col("tmp")[13].cast(FloatType()))
#             .withColumn("count", col("tmp")[14].cast(FloatType()))
#             .withColumn("srv_count", col("tmp")[15].cast(FloatType()))
#             .withColumn("serror_rate", col("tmp")[16].cast(FloatType()))
#             .withColumn("srv_serror_rate", col("tmp")[17].cast(FloatType()))
#             .withColumn("rerror_rate", col("tmp")[18].cast(FloatType()))
#             .withColumn("srv_rerror_rate", col("tmp")[19].cast(FloatType()))
#             .withColumn("same_srv_rate", col("tmp")[20].cast(FloatType()))
#             .withColumn("diff_srv_rate", col("tmp")[21].cast(FloatType()))
#             .withColumn("srv_diff_host_rate", col("tmp")[22].cast(FloatType()))
#             .withColumn("dst_host_count", col("tmp")[23].cast(FloatType()))
#             .withColumn("dst_host_srv_count", col("tmp")[24].cast(FloatType()))
#             .withColumn("dst_host_same_srv_rate", col("tmp")[25].cast(FloatType()))
#             .withColumn("dst_host_diff_srv_rate", col("tmp")[26].cast(FloatType()))
#             .withColumn("dst_host_same_src_port_rate", col("tmp")[27].cast(FloatType()))
#             .withColumn("dst_host_srv_diff_host_rate", col("tmp")[28].cast(FloatType()))
#             .withColumn("dst_host_serror_rate", col("tmp")[29].cast(FloatType()))
#             .withColumn("dst_host_srv_serror_rate", col("tmp")[30].cast(FloatType()))
#             .withColumn("dst_host_rerror_rate", col("tmp")[31].cast(FloatType()))
#             .withColumn("dst_host_srv_rerror_rate", col("tmp")[32].cast(FloatType()))
#             .withColumn("cluster", col("tmp")[33].cast(StringType()))
#             .select(
#                 "duration",
#                 "src_bytes",
#                 "dst_bytes",
#                 "wrong_fragment",
#                 "urgent",
#                 "hot",
#                 "num_failed_logins",
#                 "num_compromised",
#                 "root_shell",
#                 "su_attempted",
#                 "num_root",
#                 "num_file_creations",
#                 "num_shells",
#                 "num_access_files",
#                 "count",
#                 "srv_count",
#                 "serror_rate",
#                 "srv_serror_rate",
#                 "rerror_rate",
#                 "srv_rerror_rate",
#                 "same_srv_rate",
#                 "diff_srv_rate",
#                 "srv_diff_host_rate",
#                 "dst_host_count",
#                 "dst_host_srv_count",
#                 "dst_host_same_srv_rate",
#                 "dst_host_diff_srv_rate",
#                 "dst_host_same_src_port_rate",
#                 "dst_host_srv_diff_host_rate",
#                 "dst_host_serror_rate",
#                 "dst_host_srv_serror_rate",
#                 "dst_host_rerror_rate",
#                 "dst_host_srv_rerror_rate",
#                 # "cluster", # This is the label, don't select on training data
#             )
#         )
#     elif database == "test":
#         training_data = (
#             streaming_df.withColumn("tmp", split(col("value"), ","))
#             .withColumn("col1", col("tmp")[0].cast(IntegerType()))
#             .withColumn("col2", col("tmp")[1].cast(IntegerType()))
#             .withColumn("col3", col("tmp")[2].cast(IntegerType()))
#             .withColumn("col4", col("tmp")[3].cast(IntegerType()))
#             .withColumn("col5", col("tmp")[4].cast(IntegerType()))
#             .withColumn("col6", col("tmp")[5].cast(IntegerType()))
#             .select("col1", "col2", "col3", "col4", "col5")  # , "col6", "timestamp")
#         )
#     # TODO: Fix if needed
#     elif databse == "toy":
#         pass
#     return training_data


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
    print("DTYPES: ", training_data.dtypes)
    print("\n\n")

    # Default:
    # test a broadcast variable
    # broadcasted_var = ssc.sparkContext.broadcast(('a','b','c'))
    # print(f"START broadcast: {broadcasted_var} {broadcasted_var.value}")

    # model = DDStreamModel(broadcasted_var=broadcasted_var)

    write_stream = (
        training_data.writeStream.trigger(processingTime="5 seconds")
        .outputMode("update")
        .option("truncate", "false")
        .format("console")
        # .foreachBatch(model.run)
        .start()
    )

    # training_data = get_training_data_exploded(input_df, database="test")

    # write_stream = (
    #     training_data.writeStream.trigger(processingTime="5 seconds")
    #     .outputMode("update")
    #     .option("truncate", "false")
    #     .format("console")
    #     .foreachBatch(model.run)
    #     .start()
    # )

    write_stream.awaitTermination(TIMEOUT)  # end of stream
    # print(f"END broadcast: {broadcasted_var} \t {broadcasted_var.value}")
    print("Stream Data Processing Application Completed.")
    write_stream.stop()
