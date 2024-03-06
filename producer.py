# manage.py

import os
import time

from kafka import KafkaProducer

from utils.kafka import get_broker_url


def wait_for_kafka_broker(comment='wait_for_kafka_broker'):
    print(f"{comment}")
    max_retries = 10
    retries = 0
    while retries < max_retries:
        try:
            broker_url = get_broker_url()
            producer = KafkaProducer(
                bootstrap_servers=[broker_url]
            )
            producer.close()
            print("\tL Kafka broker is available.")
            return
        except Exception as e:
            print(f"Failed to connect to Kafka broker: {e} | try count : {retries}")
            retries += 1
            time.sleep(5)  # 5초 대기 후 다시 시도
    print("Unable to connect to Kafka broker after multiple retries. Exiting...")
    exit(1)
