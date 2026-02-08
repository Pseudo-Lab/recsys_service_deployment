"""
Author : Youngsang Jeong
"""

from llm_response.get_llm_model import get_llm_model


llm = get_llm_model()

def analyze_user_input(user_input):
    instruction_prompt = f"""
    You are a sophisticated AI that analyzes user input to determine the type of question and extract hidden meanings. And you are a context expert. You need to write a prompt that identifies and responds to the following text. Your prompt should include underlying high-context elements not revealed in the text.

    TASK:
    1. Classify the user input as either a '추천형질문' or a '검색형질문'.
    2. Break down the user input into a list of insights, including implicit meanings not directly stated in the text.

    EXAMPLE:
    USER INPUT: '제주시 한림읍에 있는 카페 중 30대 이용 비중이 가장 높은 곳은?'

    OUTPUT:
    ('검색형질문', ['제주시 한림읍에 위치한 카페', '30대 이용자 비율', '이용자 통계 또는 이용 비율을 제공하는 자료', '카페의 인기 및 트렌드', '연령대별 고객 선호도를 반영한 카페 평가', '30대가 주로 찾는 장소의 특징(분위기, 메뉴, 위치 등)'])

    USER INPUT: '연인과 함께 시간을 보낼 수 있는 양식집 추천해줘'

    OUTPUT:
    ('추천형질문', ['연인이 함께 시간을 보내기 위해 중요한 요소에는 분위기, 음식의 퀄리티, 편안한 대화가 가능한 환경 등이 있습니다.', '이러한 요소들을 고려할 때, 고급스러우면서도 따뜻한 분위기를 갖추고 있고, 프라이빗한 공간이 있는 양식 레스토랑을 추천해주세요.', '장소의 분위기나 음식의 스타일이 연인과의 추억을 더 특별하게 만들어줄 수 있도록 세심하게 골라주세요.'])

    USER INPUT: {user_input}

    OUTPUT: 
    """

    analysis_result = llm.invoke(instruction_prompt)
    
    return analysis_result.content

def handle_user_query(user_input):
    question_type = analyze_user_input(user_input)
    
    # ... existing code to handle the question type ...
    return question_type

if __name__ == "__main__":

    def test_handle_user_query():
        test_inputs = [
            "20대 친구들과 함께 방문할 만한 맛집을 알려줘",
            "제주시 한림읍에 있는 카페 중 30대 이용 비중이 가장 높은 곳은?",
            "제주시 노형동에 있는 단품요리 전문점 중 이용 건수가 상위 10%에 속하고 현지인 이용 비중이 가장 높은 곳은?",
            "부모님과 함께 즐길 수 있는 카페 알려줘",
        ]

        for user_input in test_inputs:
            result = handle_user_query(user_input)
            print(f"Input: {user_input}\nOutput: {result}\n")
    
    test_handle_user_query()