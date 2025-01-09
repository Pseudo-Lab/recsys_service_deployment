from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from openai import OpenAI


class FirstPageAgent:
    def __init__(self):
        load_dotenv(".env.dev")
        self.llm = ChatOpenAI(model_name="gpt-4o-mini")

    def welcome_message(self):
        """
        Stream an openai completion back to the client.
        Docs: https://platform.openai.com/docs/api-reference/streaming
        """
        openai_client = OpenAI()
        stream = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an reception agent greeting user visits website called 'PseudoRec'. You can only speak in Korean.",
                },
                {
                    "role": "user",
                    "content": """Warmly welcome a new user and ask to rate the given movies to provide accurate recommendations. Use H1 heading with first line. Use more colorful and rich expressions than example.
                    
                    example)
                    # **PseudoRec에 오신 것을 환영합니다!**
                    
                    영화 추천을 받고 정확한 추천을 받기 위해 제안된 영화에 평점을 매겨주세요 🎥✨. 
                    많이 선택할수록 취향에 맞는 영화를 찾아드리는데 도움이 됩니다.""",
                },
            ],
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                # We tidy the content for showing in the browser:
                content = content.replace("\n", "<br>")
                content = content.replace(",", ", ")
                content = content.replace(".", ". ")

                yield f"data: {content}\n\n"  # We yield the content in the text/event-stream format.
