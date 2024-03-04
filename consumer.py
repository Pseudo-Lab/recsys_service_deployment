import json
import os

from multiprocessing import Process
from kafka import KafkaConsumer

# Kafka Consumer 인스턴스 생성
from db_clients.dynamodb import DynamoDBClient
from producer import wait_for_kafka_broker

wait_for_kafka_broker('wait broker in consumer.py')
print(f"[Consumer Broker Connection] os.getenv('BROKER_URL_IN_CONTAINER', 'localhost:9092') : {os.getenv('BROKER_URL_IN_CONTAINER', 'localhost:9092')}")

# Kafka Consumer 인스턴스 생성
consumer = KafkaConsumer(
    'log_movie_click',
    bootstrap_servers=[os.getenv('BROKER_URL_IN_CONTAINER', 'localhost:9092')],
    auto_offset_reset='earliest',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

table_clicklog = DynamoDBClient(table_name='clicklog')


# Kafka Consumer 메시지 처리 루프
def process_messages():
    for message in consumer:
        log_data = message.value

        print(f"message.value : {message.value}")
        # DynamoDB에 데이터 저장
        try:
            response = table_clicklog.put_item(click_log=log_data)
            print('Successfully saving data')
            # 성공적으로 저장되었을 때의 로깅 또는 처리
        except Exception as e:
            # 에러 처리 로직
            print(f"Error saving data to DynamoDB: {e}")


# Kafka Consumer를 백그라운드에서 실행
if __name__ == '__main__':
    consumer_process = Process(target=process_messages)
    consumer_process.start()
