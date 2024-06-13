import time

from db_clients.dynamodb import DynamoDBClient
from movie.utils import get_username_sid

table_llm = DynamoDBClient(table_name='llm')


def log_question(request, question):
    username, session_id = get_username_sid(request, _from='llmrec_hyeonwoo')
    log = {
        'userId': username,
        'model': 'hyeonwoo',
        'timestamp': int(time.time()),
        'question': question
    }
    table_llm.put_item(click_log=log)
    return
