import os

import pymysql


class RDSMysqlClient:
    def __init__(self):
        self.ENDPOINT = "pseudorec.cvhv2t0obyv3.ap-northeast-2.rds.amazonaws.com"
        self.PORT = 3306
        self.USER = "admin"
        self.REGION = "ap-northeast-2c"
        self.DBNAME = "movielens25m"
        os.environ['LIBMYSQL_ENABLE_CLEARTEXT_PLUGIN'] = '1'
        self.passwd = os.environ['RDS_MYSQL_PW']

    def get_connection(self):
        connection = pymysql.connect(host=self.ENDPOINT, user=self.USER, passwd='', port=self.PORT, database=self.DBNAME)
        return connection

    def get_count(self, table_name):
        with self.get_connection().cursor() as cursor:
            cursor.execute(f"select count(*) from {table_name}")
            return cursor.fetchall()[0][0]