import os

from langchain.chat_models import ChatOpenAI


def get_model_kyeongchan():
    return ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=os.environ.get("OPENAI_API_KEY"))


kyeongchan_model = get_model_kyeongchan()
