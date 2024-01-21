```python
import sys
sys.path.append('../')

from clients import MysqlClient
from dotenv import load_dotenv
import os

load_dotenv()
os.getenv('AWS_ACCESS_KEY_ID')
os.getenv('AWS_SECRET_ACCESS_KEY')
mysql = MysqlClient()
```