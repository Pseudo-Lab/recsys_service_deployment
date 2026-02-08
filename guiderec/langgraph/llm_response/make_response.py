from llm_response.get_llm_model import get_llm_model

llm = get_llm_model()


def get_llm_response(query, response):
    """
    선진버전 : 1020
    """

    prompt_template = f"""
    You're a chatbot that suggests stores based on user requests.

    TASK:
    1. Consider the user's needs.
    2. Suggest up to three suitable restaurants from the provided context.
    3. Ensure each recommendation truly fits the user's needs; if not don't recommend.
    4. Justify each suggestion without quoting reviews directly.
    5. Include each restaurant's image using markdown and change width of 200px (ignore if NaN), rating (ignore if NaN), and menu (ignore if NaN).
    6. Mention 'Nearby tourist attractions' for each restaurant.

    QUESTION: {query}

    CONTEXT: {response}

    ANSWER:"
    """

    ai_msg = llm.invoke(prompt_template)

    return ai_msg.content