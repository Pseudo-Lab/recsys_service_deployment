import os

from dotenv import load_dotenv

load_dotenv('.env.dev')

def get_broker_url():
    if os.getenv('IN_CONTAINER') == 'YES':
        broker_url = os.getenv('BROKER_URL_IN_CONTAINER')
    else:
        broker_url = 'localhost:9092'
    print(f"\tL [IN_CONTAINER? {os.getenv('IN_CONTAINER', 'NO')}] broker url : {broker_url}")
    return broker_url