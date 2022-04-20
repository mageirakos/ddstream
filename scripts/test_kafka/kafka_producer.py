from kafka import KafkaProducer
import sched, time


def file_reader(file_name, convertToBytes=False):
    """
    Generator for reading the dataset.
    Call with next(gen) to get the next row.
    """
    for row in open(file_name, "r"):
        if convertToBytes:
            yield bytes(row.rstrip("\n"), encoding="utf-8")
        else:
            yield row.rstrip("\n")


def generate_stream(rate):
    """
    :param rate: messages per second
    """
    # TODO: remove globals
    global processedCount, timeInterval, total, reader, dataset
    part = 0
    key = 0
    numPartitions = 1
    targetDone = 0
    while targetDone < rate:  # and processedCount < total:
        try:
            stream = next(reader)
        except StopIteration:
            reader = file_reader(dataset)
            stream = next(reader)

        # TODO: Figure out where to use the key -> this I think points to the correct partition
        key = part % numPartitions
        producer.send(topic, bytes(stream, encoding="utf8"))
        print(stream)

        targetDone += 1
        processedCount += 1


def schedule_produce_jobs():
    """
    Schedules produce jobs to Kafka every time timeInterval given a specific rate.

    :param rate = rows per second
    :param timeInterval = wait time before next job (in ms)
    """
    # TODO: remove globals
    global rate, total_rows

    scheduler = sched.scheduler(time.time, time.sleep)
    total_jobs = total_rows // rate  # rec. to be exactly divisable by rate
    for i in range(total_jobs):
        next_t = (i * timeInterval) / 1000
        scheduler.enter(next_t, 1, generate_stream, argument=(rate,))
    scheduler.run()


if __name__ == "__main__":
    # TODO: Add these as options when running the script
    rate = 1
    total_rows = 50_000
    numPartitions = 1
    targetDone = 0
    processedCount = 0
    timeInterval = 1000  # in ms
    dataCount = 25192  # total number of data

    nsl_kdd_dataset = "./data/NSL-KDD/KDDTrain+_20Percent.txt"

    reader = file_reader(nsl_kdd_dataset)
    topic = "weather"
    producer = KafkaProducer(bootstrap_servers="localhost:9092")

    schedule_produce_jobs()
    # TODO: Kafka Consumer gia na einai eisodos sto denstream stin morfi pou theloume
    # TODO: Spark Consumer gia na doume pws erxode ta data
    # TODO: Option na dialegeis dataset
    # TODO: Na valw kai alla datasets
