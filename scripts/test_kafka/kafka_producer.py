from kafka import KafkaProducer
import sched, time, argparse, sys


def file_reader(file_name, convertToBytes=False):
    """Generator for reading the dataset. Call with next(gen) to get the next row."""
    for row in open(file_name, "r"):
        if convertToBytes:
            yield bytes(row.rstrip("\n"), encoding="utf-8")
        else:
            yield row.rstrip("\n")


def generate_stream():
    """Send stream of data to Kafka topic based on rate."""
    global reader, total_processed
    part = 0
    key = 0
    total_current_interval = 0
    while total_current_interval < rate:
        try:
            stream = next(reader)
        except StopIteration:
            reader = file_reader(dataset)
            stream = next(reader)

        producer.send(topic, bytes(stream, encoding="utf8"))
        print(f"Sending data to Kafka '{topic}', #{total_processed}")

        total_current_interval += 1
        total_processed += 1


def schedule_produce_jobs():
    """Schedules produce jobs to Kafka every time time_interval given a specific rate."""
    scheduler = sched.scheduler(time.time, time.sleep)
    total_jobs = total_data // rate

    for i in range(total_jobs):
        next_t = (i * time_interval) / 1000
        scheduler.enter(next_t, 1, generate_stream, argument=())
    scheduler.run()


def parse_args():
    DATA_PATH = "./data/"  # assuming script is run from top_level
    source_path = {
        "nsl-kdd": DATA_PATH + "NSL-KDD/KDDTrain+_20Percent.txt",
        "kdd-99": DATA_PATH + "",
    }

    # Parse cli arguments
    parser = argparse.ArgumentParser(
        description="""This is the script responsible for producing the Kafka Stream"""
    )
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")

    required.add_argument(
        "-s",
        "--source",
        default="nsl-kdd",
        help="Source dataset for the stream (default nsl-kdd)",
        required=True,
    )
    optional.add_argument(
        "-r",
        "--rate",
        default=1000,
        type=int,
        help="Number of rows per interval (default 1000)",
    )
    optional.add_argument(
        "-i",
        "--interval",
        default=1000,
        type=int,
        help="Interval at which the stream is updated (default 1000 ms)",
    )
    optional.add_argument(
        "--total_data",
        default=10_000,
        type=int,
        help="Target number of data processed (default 10_000)\
        \nRecommended that total_data is exactly divisible by rate.",
    )
    optional.add_argument(
        "--partitions",
        default=1,
        type=int,
        help="Number of Kafka partitions (default 1)",
    )
    optional.add_argument(
        "--topic",
        default="test",
        help="Kafka topic (default 'test')",
    )

    args = parser.parse_args()
    dataset = source_path[args.source]
    rate = args.rate
    time_interval = args.interval
    total_data = args.total_data
    num_partitions = args.partitions
    topic = args.topic
    return dataset, rate, time_interval, total_data, num_partitions, topic


if __name__ == "__main__":
    dataset, rate, time_interval, total_data, num_partitions, topic = parse_args()

    reader = file_reader(dataset)
    try:
        producer = KafkaProducer(bootstrap_servers="localhost:9092")
    except:
        print(f"Failed: Make sure Kafka server is running and {topic} topic exists")
        sys.exit()

    start, total_processed = time.time(), 0
    schedule_produce_jobs()
    print(
        f"- Finished sending {total_processed} data; Spend {round(time.time() - start, 3)} seconds"
    )
