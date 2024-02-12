# manage.py

import os
import time

from kafka import KafkaProducer


def wait_for_kafka_broker():
    max_retries = 10
    retries = 0
    while retries < max_retries:
        try:
            producer = KafkaProducer(
                bootstrap_servers=[os.getenv('BROKER_URL_IN_CONTAINER', 'localhost:9092')]
            )
            producer.close()
            print("Kafka broker is available.")
            return
        except Exception as e:
            print(f"Failed to connect to Kafka broker: {e} | try count : {retries}")
            retries += 1
            time.sleep(5)  # 5초 대기 후 다시 시도
    print("Unable to connect to Kafka broker after multiple retries. Exiting...")
    exit(1)
