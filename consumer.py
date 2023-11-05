from kafka import KafkaConsumer
import json

# Kafka Consumer 생성
consumer = KafkaConsumer('movie_title_ver2',
                         bootstrap_servers=['localhost:9092'],
                         value_deserializer=lambda x: json.loads(x.decode('utf-8')))

# 메시지를 수신하면 출력
for message in consumer:
    print(message.value)