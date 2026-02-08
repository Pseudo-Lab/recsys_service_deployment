import os
from langchain_google_genai import ChatGoogleGenerativeAI
from guiderec_config import CONFIG


# def get_llm_model():
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash",
#         temperature=0,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#         api_key=CONFIG.gemini_api_key
#     )
#     return llm

from langchain.chat_models import ChatOpenAI

def get_llm_model():
    llm = ChatOpenAI(
        model="gpt-4.1-2025-04-14",
        temperature=0,
        api_key=os.environ["KYEONGCHAN_OPENAI_API_KEY"]
    )
    return llm
