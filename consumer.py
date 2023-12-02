from kafka import KafkaConsumer
from clients import MysqlClient, DynamoDB
import json

# Kafka Consumer 인스턴스 생성
consumer = KafkaConsumer(
    'log_movie_click',
    bootstrap_servers='kafka:9093',
    auto_offset_reset='earliest',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

table_clicklog = DynamoDB(table_name='clicklog')

# Kafka Consumer 메시지 처리 루프
for message in consumer:
    log_data = message.value

    print(log_data)
    # DynamoDB에 데이터 저장
    try:
        response = table_clicklog.put_item(click_log=log_data)
        print('Successfully saving data')
        # 성공적으로 저장되었을 때의 로깅 또는 처리
    except Exception as e:
        # 에러 처리 로직
        print(f"Error saving data to DynamoDB: {e}")