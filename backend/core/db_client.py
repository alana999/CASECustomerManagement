import os
import pymysql
import pandas as pd

class DBClient:
    """数据库客户端（单例模式）"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DBClient, cls).__new__(cls)
            cls._instance._init_conn()
        return cls._instance
        
    def _init_conn(self):
        # 实际项目中应从 .env 读取
        self.host = os.getenv("DB_HOST", "localhost")
        self.user = os.getenv("DB_USER", "root")
        self.password = os.getenv("DB_PASS", "123mysql")
        self.database = os.getenv("DB_NAME", "bank")
        
    def get_connection(self):
        return pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        
    def execute_query(self, sql: str) -> list:
        """执行查询并返回字典列表"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                return cursor.fetchall()
        finally:
            conn.close()
            
    def query_to_dataframe(self, sql: str) -> pd.DataFrame:
        """执行查询并返回 DataFrame（供模型使用）"""
        conn = self.get_connection()
        try:
            return pd.read_sql(sql, conn)
        finally:
            conn.close()

# 暴露单例实例
db_client = DBClient()
