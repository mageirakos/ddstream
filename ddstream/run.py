from settings import *
from model import DDStreamModel

# general
import argparse
from pyspark.sql import SparkSession

# import numpy as np

# from pyspark.sql import udf
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.linalg import DenseVector, Vectors, VectorUDT


# TODO: Use all of the arguments in the code outside of parse_args
def parse_args():
    # Parse cli arguments
    parser = argparse.ArgumentParser(
        description="""This is the script responsible for running DDStream"""
    )
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")

    # TODO: For all arguments understand what they are and where they are used
    optional.add_argument(
        "--batchTime",
        default="5",
        type=int,
        help="Batch Time (default '5')",
    )
    optional.add_argument(
        "--appName",
        default="ddstream",
        help="Application name (default 'ddstream')",
    )
    optional.add_argument(
        "--numDimensions",
        default="80",
        type=int,
        help="Number of dimensions (default '80')",
    )
    optional.add_argument(
        "--speedRate",
        default="1000",
        type=int,
        help="Speed Rate input (default '1000')",
    )
    # The number of initialData should be the same as the flow rate
    optional.add_argument(
        "--initialDataAmount",
        default="2500",
        type=int,
        help="Initial data amount (default '2500')",
    )
    optional.add_argument(
        "--lmbda",
        default="0.25",
        type=float,
        help="Lambda (default '0.25')",
    )
    optional.add_argument(
        "--epsilon",
        default="16",
        type=int,
        help="epsilon (default '16')",
    )
    optional.add_argument(
        "--initialEpsilon",
        default="0.02",
        type=float,
        help="Initial epsilon (default '0.02')",
    )
    optional.add_argument(
        "--mu",
        default="10.0",
        type=float,
        help="mu (default '10.0')",
    )
    optional.add_argument(
        "--beta",
        default="0.2",
        type=float,
        help="beta (default '0.2')",
    )
    optional.add_argument(
        "--tfactor",
        default="1.0",
        type=float,
        help="tfactor (default '1.0')",
    )
    # TODO: Fix
    optional.add_argument(
        "--initialDataPath",
        default="",
        help="Initial data path (default '')",
    )
    optional.add_argument(
        "--offlineEpsilon",
        default="16.0",
        type=float,
        help="offline epsilon (default '16.0')",
    )
    # TODO: What is this where is it used
    optional.add_argument(
        "--trainingDataAmount",
        default="100",
        help="Training data amount (default '100')",
    )
    # TODO: What?
    optional.add_argument(
        "--osTr",
        default="",
        help="osTr (default '')",
    )
    optional.add_argument(
        "--k",
        default="5",
        type=int,
        help="k (default '5')",
    )
    # TODO: What?
    optional.add_argument(
        "--cmDataPath",
        default="/",
        help="cmDataPath (default '/')",
    )
    optional.add_argument(
        "--offlineMu",
        default="10.0",
        type=float,
        help="Offline mu (default '10.0')",
    )
    # TODO: what?
    optional.add_argument(
        "--check",
        default="1",
        type=int,
        help="check  (default '1')",
    )
    # inputPath
    optional.add_argument(
        "-d",
        "--dataset",
        default="test",
        help="Dataset input (default 'test')",
    )
    # trainingTopic
    optional.add_argument(
        "--topic",
        default="test",
        help="Kafka topic (default 'test')",
    )
    # timeout
    optional.add_argument(
        "-t",
        "--timeout",
        default=10,
        type=int,
        help="Timeout of streaming application (default 10 seconds)",
    )
    args = parser.parse_args()

    batchTime = args.batchTime
    appName = args.appName
    numDimensions = args.numDimensions
    speedRate = args.speedRate
    initialDataAmount = args.initialDataAmount
    lmbda = args.lmbda
    epsilon = args.epsilon
    initialEpsilon = args.initialEpsilon
    mu = args.mu
    beta = args.beta
    tfactor = args.tfactor
    initialDataPath = args.initialDataPath
    offlineEpsilon = args.offlineEpsilon
    trainingDataAmount = args.trainingDataAmount
    osTr = args.osTr
    k = args.k
    cmDataPath = args.cmDataPath
    offlineMu = args.offlineMu
    check = args.check

    inputPath = args.dataset
    trainingTopic = args.topic
    timeout = args.timeout
    return (
        batchTime,
        appName,
        numDimensions,
        speedRate,
        initialDataAmount,
        lmbda,
        epsilon,
        initialEpsilon,
        mu,
        beta,
        tfactor,
        initialDataPath,
        offlineEpsilon,
        trainingDataAmount,
        osTr,
        k,
        cmDataPath,
        offlineMu,
        check,
        inputPath,
        trainingTopic,
        timeout,
    )


# TODO: Might need to change again into numpy array instead of typical "Vector"
# or do it later when calling DBSCAN from sklearn? http://blog.madhukaraphatak.com/spark-vector-to-numpy/
def split_data(streaming_df, database="nsl-kdd"):
    # print("STEP 0: In split_data")
    """Change the input stream to be (label, Vector<features>)"""
    #TODO: This can be take from numDimensions
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

    # https://stackoverflow.com/questions/49623620/what-type-should-the-dense-vector-be-when-using-udf-function-in-pyspark
    dense_features = udf(lambda arr: Vectors.dense(get_features(arr)), VectorUDT())
    # np array # dense_features = udf(lambda arr: np.array(get_features(arr)))
    split_df = streaming_df.select(
        streaming_df["key"].cast("Long"),
        split(streaming_df["value"], ",").alias("full_input_array"),
    )
    # print(f"SPLIT_DF:{split_df}")
    result = split_df.withColumn(
        "label", split_df["full_input_array"].getItem(num_feats)
    ).withColumn("features", dense_features(split_df["full_input_array"]))
    # print("STEP 2: Leaving split_data")
    return result


def print_field_dtypes(streaming_df):
    for field in streaming_df.schema.fields:
        print(field.name + " , " + str(field.dataType))
    return


if __name__ == "__main__":

    (
        BATCH_TIME,
        APP_NAME,
        NUM_DIMENSIONS,
        SPEED_RATE,
        INITIAL_DATA_AMOUNT,
        LAMBDA,
        EPSILON,
        INITIAL_EPSILON,
        MU,
        BETA,
        TFACTOR,
        INITIAL_DATA_PATH,
        OFFLINE_EPSILON,
        TRAINING_DATA_AMAOUNT,
        OS_TR,
        K,
        CM_DATA_PATH,
        OFFLINE_MU,
        CHECK,
        INPUT_DATA,
        TOPIC,
        TIMEOUT,
    ) = parse_args()

    ssc = SparkSession.builder.appName("ddstream").getOrCreate()
    ssc.sparkContext.setLogLevel("WARN")
    # All local files need to be added like so:
    ssc.sparkContext.addPyFile("ddstream/model.py")
    ssc.sparkContext.addPyFile("ddstream/microcluster.py")

    # Streaming Query
    # TODO: Maybe using StreamingListener is a must (I need it to print some results )
    # https://spark.apache.org/docs/2.2.0/api/python/pyspark.streaming.html
    # https://www.youtube.com/watch?v=iqIdmCvSwwU

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

    print("\n")

    # database options : [ 'test', 'nsl-kdd', 'toy', 'init_toy']
    data = split_data(input_df1, database=INPUT_DATA)
    # # For testing:
    # random_stream = (
    #     data.writeStream.trigger(processingTime="5 seconds")
    #     .outputMode("update")
    #     .option("truncate", "false")
    #     .format("console")
    #     # .foreachBatch(model.run)
    #     .start()
    # )
    # random_stream.awaitTermination(TIMEOUT)  # end of stream
    # random_stream.stop()

    # We don't need the label on the training data
    testing_data = data.select("key", "label", "features")
    training_data = data.select("key", "features")

    # Default:
    # test a broadcast variable
    # broadcasted_var = ssc.sparkContext.broadcast(("a", "b", "c"))
    # print(f"START broadcast: {broadcasted_var} {broadcasted_var.value}")
    # model = DDStreamModel(broadcasted_var=broadcasted_var)

    model = DDStreamModel(NUM_DIMENSIONS, BATCH_TIME)

    # Step 1. Initialize Micro Clusters
    initialDataPath = "./data/init_toy_dataset.csv"  # must be path in container file tree (shared volume)
    # TODO: set initialEpsilon to 0.02 (after we standardise)
    initialEpsilon = 0.5
    model.initDBSCAN(ssc, initialEpsilon, initialDataPath)

    # Step 2. Start Training Stream
    print("\nSTART TRAINING\n")
    training_data_stream = (
        #TODO: I think processingTime === self.batchTime -> change it + add it to model
        training_data.writeStream
        .trigger(processingTime="5 seconds")
        .outputMode("update")
        .option("truncate", "false")
        .format("console")
        .foreachBatch(model.run)
        .start()
    )

    # TODO: Step 3. Print Results (info kept on streaming linstener)
    # https://www.youtube.com/watch?v=iqIdmCvSwwU

    training_data_stream.awaitTermination(TIMEOUT)  # end of stream
    training_data_stream.stop()

    # print(f"END broadcast: {broadcasted_var} \t {broadcasted_var.value}")
    print(f"END broadcastPMic: {model.broadcastPMic} \t {model.broadcastPMic.value}")

    print("Stream Data Processing Application Completed.")
