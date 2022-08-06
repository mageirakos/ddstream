DATA_PATH = "./data/"
KAFKA_BOOTSTRAP_SERVERS = "kafka:29092"  # when run from container
# KAFKA_BOOTSTRAP_SERVERS = "localhost:9092" # when run from outside docker network
# KAFKA_BOOTSTRAP_SERVERS = "172.24.0.7:9092" # -//-
global DETAILS, MACRO_METRICS, MICRO_METRICS, MICRO_CLUSTERS, MACRO_CLUSTERS

DETAILS = {
    "appName": [],
    "dataset": [],
    "dataset location": [],
    "num of labels": [],
    "data amount": [],
    "init data location": [],
    "init data amount": [],
    # data hyperparameters
    "num of features": [],
    "stream speed per sec": [],
    "batch time": [],
    # model hyperparameters
    "init epsilon": [],
    "init mu": [],
    "lambda": [],
    "mu": [],
    "beta": [],
    "epsilon": [],
    "offline mu": [],
    "offline epsilon": [],
}

MACRO_METRICS = {
    "name": [],
    "value": [],
}

MICRO_METRICS = {
    "batch id": [],
    "name": [],
    "value": [],
}

MICRO_CLUSTERS = {
    "batch id": [],
    "microcluster id": [],
    "centroid": [],
    "cf1x": [],
    "cf2x": [],
    "weight": [],
    "t0": [],  # microcluster creation time
    "lastEdit": [],
    "pts": [],
    "lbl counts": [],  # make it a string or something
    "correctPts": [],
    "label": [],
    "purity": [],
}

MACRO_CLUSTERS = {
    "microcluster id": [],
    "total batches": [],
    "centroid": [],
    "cf1x": [],
    "cf2x": [],
    "weight": [],
    "pts": [],
    "lbl counts": [],  # make it a string or something
    "correctPts": [],
    "label": [],
    "purity": [],
}


def append_to_DETAILS(
    appName,
    dataset,
    dataset_location,
    num_of_labels,
    data_amount,
    init_data_location,
    init_data_amount,
    num_of_features,
    stream_speed_per_sec,
    batch_time,
    init_epsilon,
    init_mu,
    lmbda,
    mu,
    beta,
    epsilon,
    offline_mu,
    offline_epsilon,
):
    global DETAILS
    DETAILS["appName"].append(appName)
    DETAILS["dataset"].append(dataset)
    DETAILS["dataset location"].append(dataset_location)
    DETAILS["num of labels"].append(num_of_labels)
    DETAILS["data amount"].append(data_amount)
    DETAILS["init data location"].append(init_data_location)
    DETAILS["init data amount"].append(init_data_amount)
    DETAILS["num of features"].append(num_of_features)
    DETAILS["stream speed per sec"].append(stream_speed_per_sec)
    DETAILS["batch time"].append(batch_time)
    DETAILS["init epsilon"].append(init_epsilon)
    DETAILS["init mu"].append(init_mu)
    DETAILS["lambda"].append(lmbda)
    DETAILS["mu"].append(mu)
    DETAILS["beta"].append(beta)
    DETAILS["epsilon"].append(epsilon)
    DETAILS["offline mu"].append(offline_mu)
    DETAILS["offline epsilon"].append(offline_epsilon)
    return True


def append_to_MACRO_METRICS(name, value):
    global MACRO_METRICS
    MACRO_METRICS["name"].append(name)
    MACRO_METRICS["value"].append(value)
    return True


def append_to_MICRO_METRICS(batch_id, name, value):
    global MICRO_METRICS
    MICRO_METRICS["batch id"].append(batch_id)
    MICRO_METRICS["name"].append(name)
    MICRO_METRICS["value"].append(value)
    return True


def append_to_MICRO_CLUSTERS(
    batch_id,
    microcluster_id,
    centroid,
    cf1x,
    cf2x,
    weight,
    t0,
    lastEdit,
    pts,
    lbl_counts,
    correctPts,
    label,
    purity,
):
    MICRO_CLUSTERS["batch id"].append(batch_id)
    MICRO_CLUSTERS["microcluster id"].append(microcluster_id)
    MICRO_CLUSTERS["centroid"].append(centroid)
    MICRO_CLUSTERS["cf1x"].append(cf1x)
    MICRO_CLUSTERS["cf2x"].append(cf2x)
    MICRO_CLUSTERS["weight"].append(weight)
    MICRO_CLUSTERS["t0"].append(t0)
    MICRO_CLUSTERS["lastEdit"].append(lastEdit)
    MICRO_CLUSTERS["pts"].append(pts)
    MICRO_CLUSTERS["lbl counts"].append(lbl_counts)
    MICRO_CLUSTERS["correctPts"].append(correctPts)
    MICRO_CLUSTERS["label"].append(label)
    MICRO_CLUSTERS["purity"].append(purity)
    return True


def append_to_MACRO_CLUSTERS(
    microcluster_id,
    total_batches,
    centroid,
    cf1x,
    cf2x,
    weight,
    pts,
    lbl_counts,
    correctPts,
    label,
    purity,
):
    MACRO_CLUSTERS["microcluster id"].append(microcluster_id)
    MACRO_CLUSTERS["total batches"].append(total_batches)
    MACRO_CLUSTERS["centroid"].append(centroid)
    MACRO_CLUSTERS["cf1x"].append(cf1x)
    MACRO_CLUSTERS["cf2x"].append(cf2x)
    MACRO_CLUSTERS["weight"].append(weight)
    MACRO_CLUSTERS["pts"].append(pts)
    MACRO_CLUSTERS["lbl counts"].append(lbl_counts)
    MACRO_CLUSTERS["correctPts"].append(correctPts)
    MACRO_CLUSTERS["label"].append(label)
    MACRO_CLUSTERS["purity"].append(purity)
    return True
