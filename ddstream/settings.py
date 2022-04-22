DATA_PATH = "./data/"
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"


TEST_DF_SCHEMA = [
    "split(value,',')[0] as col1",
    "split(value,',')[1] as col2",
    "split(value,',')[2] as col3",
    "split(value,',')[3] as col4",
    "split(value,',')[4] as col5",
    "split(value,',')[4] as col6",
]
