import os
from langchain_google_genai import ChatGoogleGenerativeAI
from ..env import Env


def get_llm_model():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=Env.gemini_api_key
    )
    return llm

