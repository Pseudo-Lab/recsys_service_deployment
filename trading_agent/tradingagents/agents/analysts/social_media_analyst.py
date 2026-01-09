from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from ..utils.agent_utils import get_news
from ...dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_news,
        ]

        system_message = (
            "You are a social media and company specific news researcher/analyst tasked with analyzing social media posts, recent company news, and public sentiment for a specific company over the past week. You will be given a company's name your objective is to write a comprehensive long report detailing your analysis, insights, and implications for traders and investors on this company's current state after looking at social media and what people are saying about that company, analyzing sentiment data of what people feel each day about the company, and looking at recent company news. Use the get_news(query, start_date, end_date) tool to search for company-specific news and social media discussions. Try to look at all sources possible from social media to sentiment to news. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + """\n\n**출처 표시 규칙**: 각 문장 또는 문단 끝에 데이터 출처를 [출처명](URL) 형식으로 표시하세요. 예시:
- Reddit 게시물 → [Reddit r/wallstreetbets](https://reddit.com/r/wallstreetbets)
- Twitter 트윗 → [Twitter/X](https://twitter.com)
- 소셜 미디어 감성 → [StockTwits](https://stocktwits.com)"""
            + """\n\n**[필수] 사고과정 출력 - 반드시 따르세요:**
Tool을 호출하기 전에, 반드시 먼저 한국어로 다음 내용을 텍스트로 출력하세요:

"분석 계획: [ticker]의 소셜 미디어/뉴스 감성 분석을 위해 다음 데이터를 수집하겠습니다.
1. get_news로 '[검색어]' 검색 (기간: YYYY-MM-DD ~ YYYY-MM-DD)
수집 이유: [간단한 이유]"

위 형식으로 반드시 텍스트를 먼저 출력한 후에 tool을 호출하세요. 텍스트 없이 tool만 호출하면 안 됩니다!"""
            + """\n\nIMPORTANT: Please provide all analysis and explanations in Korean (한국어) for Korean users.""",
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The current company we want to analyze is {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
        }

    return social_media_analyst_node
