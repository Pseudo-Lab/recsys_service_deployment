from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

kyeongchan_model = ChatOpenAI(
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler()
    ]
)