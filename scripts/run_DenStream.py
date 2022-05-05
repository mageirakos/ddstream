# Docs: https://riverml.xyz/dev/api/cluster/DenStream/

from river import stream, cluster

params = {
    "converters": {
        "duration": float,
        "src_bytes": float,
        "dst_bytes": float,
        "land": float,
        "wrong_fragment": float,
        "urgent": float,
        "hot": float,
        "num_failed_logins": float,
        "logged_in": float,
        "num_compromised": float,
        "root_shell": float,
        "su_attempted": float,
        "num_root": float,
        "num_file_creations": float,
        "num_shells": float,
        "num_access_files": float,
        "is_guest_login": float,
        "count": float,
        "srv_count": float,
        "serror_rate": float,
        "srv_serror_rate": float,
        "rerror_rate": float,
        "srv_rerror_rate": float,
        "same_srv_rate": float,
        "diff_srv_rate": float,
        "srv_diff_host_rate": float,
        "dst_host_count": float,
        "dst_host_srv_count": float,
        "dst_host_same_srv_rate": float,
        "dst_host_diff_srv_rate": float,
        "dst_host_same_src_port_rate": float,
        "dst_host_srv_diff_host_rate": float,
        "dst_host_serror_rate": float,
        "dst_host_srv_serror_rate": float,
        "dst_host_rerror_rate": float,
        "dst_host_srv_rerror_rate": float,
        "cluster": str,
    }
}


denstream = cluster.DenStream(
    decaying_factor=0.25,
    beta=1.2,
    mu=2,
    epsilon=5,
    n_samples_init=1000,
    stream_speed=1000,
)
# nsl-kdd-clean-scaled-header
# temp-small
DATA_PATH = "./../thesis/data/"
for x, y in stream.iter_csv(
    DATA_PATH + "nsl-kdd-clean-scaled-header.txt", target="cluster", **params
):
    # print(x,y)
    denstream = denstream.learn_one(x)

print(
    denstream.predict_one(
        {
            "count": -0.7289497184250541,
            "diff_srv_rate": -0.3492750103829506,
            "dst_bytes": -0.024032758192795798,
            "dst_host_count": -1.7933638573943145,
            "dst_host_diff_srv_rate": -0.44093174471784924,
            "dst_host_rerror_rate": -0.38513219925327796,
            "dst_host_same_src_port_rate": 0.17040509165660098,
            "dst_host_same_srv_rate": 1.0696418299526633,
            "dst_host_serror_rate": -0.6417913327308392,
            "dst_host_srv_count": 1.2647171037248568,
            "dst_host_srv_diff_host_rate": -0.1071149862196999,
            "dst_host_srv_rerror_rate": -0.3742733197216746,
            "dst_host_srv_serror_rate": -0.627352295451972,
            "duration": -0.11354840373626408,
            "hot": -0.09193152797180429,
            "is_guest_login": -0.09598770513349468,
            "land": -0.008910299337739708,
            "logged_in": 1.2381725927142462,
            "num_access_files": -0.04391591310971378,
            "num_compromised": -0.021872171578008325,
            "num_failed_logins": -0.026219769982544758,
            "num_file_creations": -0.02780746500248676,
            "num_root": -0.02172373325676766,
            "num_shells": -0.018904226064123595,
            "rerror_rate": -0.3721782160049241,
            "root_shell": -0.039375745398740974,
            "same_srv_rate": 0.7720933378224009,
            "serror_rate": -0.6401293050536373,
            "src_bytes": -0.009998993778195362,
            "srv_count": -0.36842005323398214,
            "srv_diff_host_rate": -0.37387866035601564,
            "srv_rerror_rate": -0.3730909421766657,
            "srv_serror_rate": -0.6339655896963234,
            "su_attempted": -0.02766492418726678,
            "urgent": -0.0063004080276369435,
            "wrong_fragment": -0.09122136504337669,
        }
    )
)

print(f"n_clusters: {denstream.n_clusters}")
# print(f"clusters: {denstream.clusters}")
# print(f"centers: {denstream.centers}")
print(f"p_micro_clusters: {denstream.p_micro_clusters}")
print(f"o_micro_clusters: {denstream.o_micro_clusters}")
