import os

import boto3
import pandas as pd
import pymysql
from boto3.dynamodb.conditions import Key, Attr
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv('.env.dev')


class MysqlClient:
    def __init__(self):
        self.endpoint = "pseudorec.cvhv2t0obyv3.ap-northeast-2.rds.amazonaws.com"
        self.port = 3306
        self.user = "admin"
        self.region = "ap-northeast-2c"
        self.dbname = "movielens25m"
        self.passwd = os.getenv('RDS_MYSQL_PW')
        os.environ['LIBMYSQL_ENABLE_CLEARTEXT_PLUGIN'] = '1'
        # self.connection = pymysql.connect(host=endpoint, user=user, passwd=passwd, port=port, database=dbname)
        self.db_url = f"mysql+mysqlconnector://admin:{os.getenv('RDS_MYSQL_PW')}@{self.endpoint}:3306/{self.dbname}"
        self.engine = create_engine(self.db_url)

    def get_connection(self):
        connection = pymysql.connect(host=self.endpoint, user=self.user, passwd=self.passwd, port=self.port,
                                     database=self.dbname)
        return connection

    def get_count(self, table_name):
        with self.get_connection().cursor() as cursor:
            cursor.execute(f"select count(*) from {table_name}")
            return cursor.fetchall()[0][0]

    def get_movies(self):
        with self.get_connection() as connection:
            df = pd.read_sql(sql='select * from movies', con=connection)
            return df

    def get_daum_movies(self):
        with self.get_connection() as connection:
            df = pd.read_sql(sql='select * from daum_movies', con=connection)
            return df

    def get_daum_ratings(self):
        with self.get_connection() as connection:
            df = pd.read_sql(sql='select * from daum_ratings', con=connection)
            return df

    def get_url(self, title):
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(f"""
            select url from movies where title = '{title}'
            """)
            url = cursor.fetchall()[0][0]
            return url

    def get_table_names(self):
        print("Tables : ")
        with self.get_connection().cursor() as cursor:
            sql = "SHOW TABLES"
            cursor.execute(sql)
            result = cursor.fetchall()
            for row in result:
                print(row[0])

    def get_data_type(self, table_name):
        with self.get_connection().cursor() as cursor:
            cursor.execute(f"SHOW COLUMNS FROM {table_name}")
            columns = cursor.fetchall()
        for column in columns:
            column_name = column[0]
            data_type = column[1]
            print(f"Column: {column_name}, Data Type: {data_type}")

    def get_nickname_ratings(self, nickname):
        with self.get_connection().cursor() as cursor:
            cursor.execute(f"""SELECT *
            FROM daum_ratings
            WHERE nickName = '{nickname}'
            """)
            result = cursor.fetchall()
        return pd.DataFrame(result)



