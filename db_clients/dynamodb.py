import os
from decimal import Decimal

import boto3
import pandas as pd
from boto3.dynamodb.conditions import Key, Attr
from dotenv import load_dotenv

load_dotenv('.env.dev')


class DynamoDBClient:
    def __init__(self, table_name: str):
        self.resource = boto3.resource(
            'dynamodb',
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            region_name=os.environ['AWS_REGION_NAME'],
        )

        self.client = boto3.client(
            'dynamodb',
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            region_name=os.environ['AWS_REGION_NAME'],
        )
        self.table = self.resource.Table(table_name)  # clicklog 테이블 등으로 연결

    def put_item(self, click_log):
        resp = self.table.put_item(Item=click_log)

    def get_a_user_logs(self, user_name: str):
        query = {"KeyConditionExpression": Key("userId").eq(user_name)}
        resp = self.table.query(**query)
        return pd.DataFrame(resp['Items'])

    def get_a_session_logs(self, session_id: str):
        query = {"KeyConditionExpression": Key("userId").eq('Anonymous'),
                 "FilterExpression": Attr("sessionId").eq(session_id)
                 }
        resp = self.table.query(**query)
        return pd.DataFrame(resp['Items'])

    def get_description(self, table_name='clicklog'):
        response = self.client.describe_table(TableName=table_name)
        return response


    # def delete_item(self, user_id: str, timestamp: Decimal, session_id: str):
    #     key = {
    #         'userId': user_id,
    #         'timestamp': timestamp
    #     }
    #     condition_expression = Attr('sessionId').eq(session_id)
    #     try:
    #         response = self.table.delete_item(
    #             Key=key,
    #             ConditionExpression=condition_expression
    #         )
    #         return response


