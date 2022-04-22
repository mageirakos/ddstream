# Make sure you have install the package - pip install git+https://github.com/MrParosk/pyDenStream.git
from denstream import DenStream
from pprint import pprint
from sklearn import metrics
import numpy as np

clusters = {
    "back": 0,
    "buffer_overflow": 1,
    "ftp_write": 2,
    "guess_passwd": 3,
    "imap": 4,
    "ipsweep": 5,
    "land": 6,
    "loadmodule": 7,
    "multihop": 8,
    "neptune": 9,
    "nmap": 10,
    "normal": 11,
    "perl": 12,
    "phf": 13,
    "pod": 14,
    "portsweep": 15,
    "rootkit": 16,
    "satan": 17,
    "smurf": 18,
    "spy": 19,
    "teardrop": 20,
    "warezclient": 21,
    "warezmaster": 22,
}


def file_reader(file_name):
    """Generator for reading the dataset. Call with next(gen) to get the next row."""
    for i, data in enumerate(open(file_name, "r")):
        if data == "":
            continue
        data = data.rstrip("\n").split(",")
        X = np.array(data[:-1]).astype(float)
        Y = data[-1]
        yield {
            "time": int(i + 1),
            "feature_array": X.reshape((1, X.shape[0])),
            "label": int(clusters[Y]),
        }


eps = 0.2
lambd = 0.25
beta = 0.75
mu = 2
min_samples = 5

label_metrics_list = [metrics.homogeneity_score, metrics.completeness_score]

DATA_PATH = "./../thesis/data/"
gen = file_reader(DATA_PATH + "nsl-kdd-clean-scaled.txt")

print("Running pyDenStream...")
ds = DenStream(eps, beta, mu, lambd, min_samples, label_metrics_list)
ds.fit_generator(gen, request_period=1000)

print(f"o_micro_clusters: {len(ds.o_micro_clusters)}")
print(f"p_micro_clusters: {len(ds.p_micro_clusters)}")
print(f"completed_o_clusters: {len(ds.completed_o_clusters)}")
print(f"completed_p_clusters: {len(ds.completed_p_clusters)}")

pprint(ds.metrics_results)
