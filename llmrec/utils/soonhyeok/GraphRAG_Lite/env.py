import os

from dotenv import load_dotenv
load_dotenv()


class Env:
    neo4j_url = os.environ["NEO4J_URI"]
    neo4j_user = os.environ["NEO4J_USERNAME"]
    neo4j_password = os.environ["NEO4J_PASSWORD"]
    os.environ["SOONHYEOK_GEMINI_API_KEY"] =os.getenv("SOONHYEOK_GEMINI_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")     
    tavily_api_key = os.environ["SOONHYEOK_TAVILY_API_KEY"]
    gemini_api_key = os.environ["SOONHYEOK_GEMINI_API_KEY"]
    openai_key = os.environ["OPENAI_API_KEY"] 
    recomm_candidates_num = 3
    recomm_select_k = 2
