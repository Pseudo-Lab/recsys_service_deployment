import time

from db_clients.dynamodb import DynamoDBClient
from movie.utils import get_username_sid

table_llm = DynamoDBClient(table_name='llm')


def log_llm(request, model_name, question='', answer='', _from=''):
    username, session_id = get_username_sid(request, _from=_from)
    log = {
        'userId': username,
        'sessionId': session_id,
        'model': model_name,
        'timestamp': int(time.time()),
        'question' : question,
        'answer': answer
    }
    table_llm.put_item(click_log=log)
    return
