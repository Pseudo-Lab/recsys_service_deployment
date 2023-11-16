import os

import pandas as pd
import pymysql


class MysqlClient:
    def __init__(self):
        self.endpoint = "pseudorec.cvhv2t0obyv3.ap-northeast-2.rds.amazonaws.com"
        self.port = 3306
        self.user = "admin"
        self.region = "ap-northeast-2c"
        self.dbname = "movielens25m"
        self.passwd = os.environ['RDS_MYSQL_PW']
        os.environ['LIBMYSQL_ENABLE_CLEARTEXT_PLUGIN'] = '1'
        # self.connection = pymysql.connect(host=endpoint, user=user, passwd=passwd, port=port, database=dbname)

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