from utils import *
from model import DDStreamModel
from offline import DDStreamOfflineModel

# general
import argparse, os, pickle
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
        help="Batch Time (default 5)",
    )
    required.add_argument(
        "--appName",
        default="ddstream",
        help="Application name (default 'ddstream')",
    )
    required.add_argument(
        "--numDimensions",
        type=int,
        help="Number of dimensions",
    )
    #TODO: Fix not correct ( unless I specify it each time )
    required.add_argument(
        "--speedRate",
        default="1000",
        type=int,
        help="Speed Rate input (default 1000)",
    )
    required.add_argument(
        "--numLabels",
        type=int,
        help="Number of distinct Labels",
    )
    # The number of initialData should be the same as the flow rate
    #TODO: Fix not correct
    required.add_argument(
        "--initialDataAmount",
        default="150",
        type=int,
        help="Initial data amount",
    )
    optional.add_argument(
        "--lmbda",
        default="0.25",
        type=float,
        help="Lambda (default 0.25)",
    )
    optional.add_argument(
        "--epsilon",
        default="0.4",
        type=float,
        help="epsilon (default 0.4)",
    )
    optional.add_argument(
        "--initialEpsilon",
        default="0.4",
        type=float,
        help="Initial epsilon (default 0.4)",
    )
    # A core object is defined as an object, in whose ≤ neighborhood the overall weight of data points is at least an integer μ.
    # mu == minPoints
    optional.add_argument(
        "--mu",
        default="10",
        type=float,
        help="mu (default 10)",
    )
    optional.add_argument(
        "--beta",
        default="0.2",
        type=float,
        help="beta (default 0.2)",
    )
    # optional.add_argument(
    #     "--tfactor",
    #     default="1.0",
    #     type=float,
    #     help="tfactor (default 1.0)",
    # )
    required.add_argument(
        "--initialDataPath",
        help="Initial data path",
    )
    optional.add_argument(
        "--offlineEpsilon",
        default="0.4",
        type=float,
        help="offline epsilon (default 0.4)",
    )
    # TODO: What is this where is it used
    required.add_argument(
        "--trainingDataAmount",
        default="10000",
        type=int,
        help="Training data amount (default 10_000)",
    )
    # # TODO: What?
    # optional.add_argument(
    #     "--osTr",
    #     default="",
    #     help="osTr (default '')",
    # )
    # # TODO: What?
    # optional.add_argument(
    #     "--k",
    #     default="5",
    #     type=int,
    #     help="k (default 5)",
    # )
    # # TODO: What?
    # optional.add_argument(
    #     "--cmDataPath",
    #     default="/",
    #     help="cmDataPath (default '/')",
    # )
    optional.add_argument(
        "--offlineMu",
        default="10",
        type=float,
        help="Offline mu (default 10)",
    )
    # # TODO: what?
    # optional.add_argument(
    #     "--check",
    #     default="1",
    #     type=int,
    #     help="check  (default 1)",
    # )
    # dataset
    optional.add_argument(
        "-d",
        "--dataset",
        default="toy",
        help="Dataset input (default 'toy')",
    )
    optional.add_argument(
        "--streamPath",
        default="./data/toy_dataset.csv",
        help="Dataset input (default './data/toy_dataset.csv')",
    )
    # trainingTopic
    optional.add_argument(
        "--topic",
        default="test",
        help="Kafka topic (default 'test')",
    )
    # timeout
    #TODO: Set TIMEOUT based on rate of speed and batch time...
    optional.add_argument(
        "-t",
        "--timeout",
        default=10,
        type=int,
        help="Timeout of streaming application (default 10 seconds)",
    )
    required.add_argument(
        "--save",
        default=False,
        type=bool,
        help="add --save if you wish to save experiments (default False)",
    )
    args = parser.parse_args()

    batchTime = args.batchTime
    appName = args.appName
    numDimensions = args.numDimensions
    speedRate = args.speedRate
    numLabels = args.numLabels
    initialDataAmount = args.initialDataAmount
    lmbda = args.lmbda
    epsilon = args.epsilon
    initialEpsilon = args.initialEpsilon
    mu = args.mu
    beta = args.beta
    # tfactor = args.tfactor
    initialDataPath = args.initialDataPath
    offlineEpsilon = args.offlineEpsilon
    trainingDataAmount = args.trainingDataAmount
    # osTr = args.osTr
    # k = args.k
    # cmDataPath = args.cmDataPath
    offlineMu = args.offlineMu
    # check = args.check
    dataset = args.dataset
    streamPath = args.streamPath
    trainingTopic = args.topic
    timeout = args.timeout
    save = args.save
    return (
        batchTime,
        appName,
        numDimensions,
        speedRate,
        numLabels,
        initialDataAmount,
        lmbda,
        epsilon,
        initialEpsilon,
        mu,
        beta,
        # tfactor,
        initialDataPath,
        offlineEpsilon,
        trainingDataAmount,
        # osTr,
        # k,
        # cmDataPath,
        offlineMu,
        # check,
        dataset,
        streamPath,
        trainingTopic,
        timeout,
        save,
    )


# TODO: Might need to change again into numpy array instead of typical "Vector"
# or do it later when calling DBSCAN from sklearn? http://blog.madhukaraphatak.com/spark-vector-to-numpy/
def split_data(streaming_df, database="nsl-kdd"):
    # print("STEP 0: In split_data")
    """Change the input stream to be (label, Vector<features>)"""
    # TODO: This can be take from numDimensions
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
        split(streaming_df["value"], "/").alias("ts-vals"),
    )

    # split_df = split_df.select(
    # split(split_df["ts-vals"].getItem(1), ",").alias("full_input_array"),
    # )
    # print(f"SPLIT_DF:{split_df}")
    split_df = split_df.withColumn("time", split_df["ts-vals"].getItem(0)).withColumn(
        "label_feats", split_df["ts-vals"].getItem(1)
    )

    split_df = split_df.select(
        split_df["key"].cast("Long"),
        split_df["time"].cast("Long"),
        split(split_df["label_feats"], ",").alias("full_input_array"),
    )
    result = split_df.withColumn(
        "label", split_df["full_input_array"].getItem(num_feats).cast("Int")
    ).withColumn("features", dense_features(split_df["full_input_array"]))
    print("STEP 2: Leaving split_data")
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
        STREAM_SPEED,
        NUM_LABELS,
        INITIAL_DATA_AMOUNT,
        LAMBDA,
        EPSILON,
        INITIAL_EPSILON,
        MU,
        BETA,
        # TFACTOR,
        INITIAL_DATA_PATH,
        OFFLINE_EPSILON,
        STREAM_DATA_AMOUNT,
        # OS_TR,
        # K,
        # CM_DATA_PATH,
        OFFLINE_MU,
        # CHECK,
        DATASET_NAME,
        STREAM_DATA_PATH,
        TOPIC,
        TIMEOUT,
        SAVE,
    ) = parse_args()

    ssc = SparkSession.builder.appName(f"{APP_NAME}").getOrCreate()
    ssc.sparkContext.setLogLevel("WARN")
    # All local files need to be added like so:
    ssc.sparkContext.addPyFile("ddstream/model.py")
    ssc.sparkContext.addPyFile("ddstream/microcluster.py")
    ssc.sparkContext.addPyFile("ddstream/offline.py")
    ssc.sparkContext.addPyFile("ddstream/utils.py")

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
    data = split_data(input_df1, database=DATASET_NAME)
    # # For testing:
    # training_data = data.select("time", "features")
    # random_stream = (
    #     training_data.writeStream.trigger(processingTime="5 seconds")
    #     .outputMode("append")
    #     .option("truncate", "false")
    #     .format("console")
    #     # .foreachBatch(model.run)
    #     .start()
    # )
    # random_stream.awaitTermination(TIMEOUT)  # end of stream
    # random_stream.stop()
    # exit()

    # 10 sec to have time to start the data stream
    #TODO: have the args one overwrite this one instead of the opposite
    TIMEOUT = 10 + (STREAM_DATA_AMOUNT / (BATCH_TIME * STREAM_SPEED) ) * BATCH_TIME
    #TODO: Change dataset name earlier at start
    DATASET_NAME = STREAM_DATA_PATH.split('/')[-1].split('.')[0]
    
    assert TIMEOUT == 10 + ( STREAM_DATA_AMOUNT / (BATCH_TIME * STREAM_SPEED) )* BATCH_TIME
    assert INITIAL_EPSILON == 0.4
    assert LAMBDA == 0.25 
    assert BETA == 0.2
    assert MU == 10
    assert NUM_LABELS == 3
    
    model = DDStreamModel(
        numDimensions=NUM_DIMENSIONS,
        batchTime=BATCH_TIME,
        epsilon=EPSILON,
        beta=BETA,
        mu=MU,
        lmbda=LAMBDA,
        num_labels = NUM_LABELS,
        Tp=4)

    assert model.Tp == 4

    # Step 1. Initialize Micro Clusters
    model.initDBSCAN(ssc, INITIAL_EPSILON, INITIAL_DATA_PATH)

    # Step 2. Start Training Stream
    training_data = data.select("time", "features", "label")
    print("\nSTART ONLINE PHASE with:")
    print(f"TOTAL_DATA : {STREAM_DATA_AMOUNT} data")
    print(f"BATCH_TIME : {BATCH_TIME} sec")
    print(f"STREAM SPEED : {STREAM_SPEED} data per sec")
    print(f"TIMEOUT : {TIMEOUT} sec (you have 10sec to start stream)")
    print(f"SAVE? : {SAVE}")
    print("\n\n")

    training_data_stream = (
        training_data.writeStream.trigger(processingTime=f"{BATCH_TIME} seconds")
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

    # print(f"END broadcastPMic: {model.broadcastPMic} \t {model.broadcastPMic.value}")

    print("Stream Data Processing Completed.")
    print("\n\n\nSTART OFFLINE PHASE\n")

    # TODO: Finish Offline phase

    # with open(f"data/toy_dataset.csv") as f:
    #     pass
    coreMicroClusters = model.getMicroClusters()
    for i, mc in enumerate(coreMicroClusters):
        print(f"cluster {i}")
        print(f"Micro cluster weight = {mc.weight}")
        print(f"Micro cluster cf1x = {mc.cf1x}")
        print(f"Micro cluster cf2x = {mc.cf2x}")
        print(f"Micro cluster center = {mc.getCentroid()}")
        print(f"Micro cluster center = {mc.getCentroid()}")
        print(f"Micro cluster purity = {mc.calcPurity()}")
    avg_purity = model.calcAvgPurity(coreMicroClusters)
    print(f"AVERAGE PURITY (micro clusters) = {avg_purity}")

    assert OFFLINE_EPSILON ==  0.4
    assert OFFLINE_MU == 10
    offline_model = DDStreamOfflineModel(epsilon=OFFLINE_EPSILON, mu=OFFLINE_MU)
    print()
    coreMacroClusters = offline_model.getFinalClusters(coreMicroClusters)
    print(f"Outside:")
    for j, mc in enumerate(coreMacroClusters):
        print(f"cluster {j}")
        print(f"Macro cluster weight = {mc.weight}")
        print(f"Macro cluster cf1x = {mc.cf1x}")
        print(f"Macro cluster cf2x = {mc.cf2x}")
        print(f"Macro cluster center = {mc.getCentroid()}")
        print(f"Macro cluster purity = {mc.calcPurity()}")
    #TODO: Fix bellow since (must not be same as microclusters)
    avg_purity = offline_model.calcAvgPurity(coreMacroClusters)
    print(f"AVERAGE PURITY (macro clusters) = {avg_purity}")


    # TODO NOW: Save MACRO_CLUSTERS, DETAILS, MACRO_METRICS
    append_to_DETAILS(
        appName=APP_NAME,
        dataset=DATASET_NAME, # can probably just get it from STREAM_DATA_PATH
        dataset_location=STREAM_DATA_PATH, #probably not needed at all unless I start kafka from here
        num_of_labels=NUM_LABELS,
        data_amount=STREAM_DATA_AMOUNT,
        init_data_location=INITIAL_DATA_PATH,
        init_data_amount=INITIAL_DATA_AMOUNT, #TODO: Fix not correct (should have assertion with Kafka)
        num_of_features=NUM_DIMENSIONS,
        stream_speed_per_sec=STREAM_SPEED,
        batch_time=BATCH_TIME,
        init_epsilon=INITIAL_EPSILON,
        init_mu=MU,
        lmbda=LAMBDA,
        mu=MU,
        beta=BETA,
        epsilon=EPSILON,
        offline_mu=OFFLINE_MU,
        offline_epsilon=OFFLINE_EPSILON,
    )

    #TODO: Fix assertion
    # assert total_batches == INITIAL_DATA_AMOUNT / (STREAM_SPEED * BATCH_TIME)

    for i, macrocl in enumerate(coreMacroClusters):
        append_to_MACRO_CLUSTERS(
            microcluster_id=id(macrocl),
            total_batches=None,#TODO: Fix this & assertion above
            centroid=macrocl.getCentroid().tolist(),
            cf1x=macrocl.cf1x.tolist(),
            cf2x=macrocl.cf2x.tolist(),
            weight=macrocl.weight,
            pts=macrocl.pts,
            lbl_counts=macrocl.lbl_counts,
            correctPts=macrocl.correctPts,
            label=macrocl.getLabel(), #TODO: Fix this is not correct
            purity=macrocl.calcPurity() #TODO: Fix this is not correct
            )
    
    avg_purity = offline_model.calcAvgPurity(coreMacroClusters)
    append_to_MACRO_METRICS(name="PURITY(avg)", value=avg_purity)

    from pprint import pprint            
    # TODO NOW: Write everything 
    print(f"Writing to file:")
    print(f"DETAILS")
    pprint(DETAILS)
    print(f"MICRO_CLUSTERS")
    pprint(MICRO_CLUSTERS)
    print(f"MICRO_METRICS")
    pprint(MICRO_METRICS)
    print(f"MACRO_CLUSTERS")
    pprint(MACRO_CLUSTERS)
    print(f"MACRO_METRICS")
    pprint(MACRO_METRICS)

    if SAVE:
        EXPERIMENT_DATA_PATH = f"/data/experiments/{DATASET_NAME}_{STREAM_SPEED}/"
        
        try:
            os.mkdir("/data/experiments/")
        except FileExistsError:
            pass
        try:
            os.mkdir(EXPERIMENT_DATA_PATH)
        except FileExistsError:
            pass

        print(f"Saving experiments in '{EXPERIMENT_DATA_PATH}'")
        with open(EXPERIMENT_DATA_PATH+f'DETAILS.pkl', 'wb') as f:
            pickle.dump(DETAILS, f)
        with open(EXPERIMENT_DATA_PATH+f'MICRO_CLUSTERS.pkl', 'wb') as f:
            pickle.dump(MICRO_CLUSTERS, f)
        with open(EXPERIMENT_DATA_PATH+f'MICRO_METRICS.pkl', 'wb') as f:
            pickle.dump(MICRO_METRICS, f)
        with open(EXPERIMENT_DATA_PATH+f'MACRO_CLUSTERS.pkl', 'wb') as f:
            pickle.dump(MACRO_CLUSTERS, f)
        with open(EXPERIMENT_DATA_PATH+f'MACRO_METRICS.pkl', 'wb') as f:
            pickle.dump(MACRO_METRICS, f)
    else:
        print('not saving experiments')
    # # to load 
    # with open('saved_dictionary.pkl', 'rb') as f:
    # loaded_dict = 
