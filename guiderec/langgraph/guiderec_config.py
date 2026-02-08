import os
from pathlib import Path

from dotenv import load_dotenv

# Find .env.dev in project root
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / '.env.dev')

class Config:
    neo4j_url = os.environ.get("GUIDEREC_NEO4J_URI", os.environ.get("NEO4J_URI"))
    neo4j_user = os.environ.get("GUIDEREC_NEO4J_USERNAME", os.environ.get("NEO4J_USERNAME"))
    neo4j_password = os.environ.get("GUIDEREC_NEO4J_PASSWORD", os.environ.get("NEO4J_PASSWORD"))
    gemini_api_key = os.environ.get("KYEONGCHAN_GEMINI_API_KEY")
    store_retriever_rev_emb_k = 30
    store_retriever_rev_emb_k_grp = 6
    recomm_candidates_num = 6
    recomm_select_k = 2

CONFIG = Config()