from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from ..utils.agent_utils import get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement, get_insider_sentiment, get_insider_transactions
from ...dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
            + """\n\n**출처 표시 규칙**: 각 문장 또는 문단 끝에 데이터 출처를 [출처명](URL) 형식으로 표시하세요. 예시:
- SEC 공시자료 → [SEC EDGAR](https://www.sec.gov/edgar)
- 재무제표 → [Yahoo Finance](https://finance.yahoo.com)
- 기업 정보 → [Alpha Vantage](https://www.alphavantage.co)
- 분기별 실적 → [Nasdaq](https://nasdaq.com)"""
            + """\n\n**[필수] 사고과정 출력 - 반드시 따르세요:**
Tool을 호출하기 전에, 반드시 먼저 한국어로 다음 내용을 텍스트로 출력하세요:

"분석 계획: [ticker]의 펀더멘털 분석을 위해 다음 재무 데이터를 수집하겠습니다.
1. get_fundamentals로 기업 개요 조회
2. get_balance_sheet로 대차대조표 조회
3. get_income_statement로 손익계산서 조회
4. get_cashflow로 현금흐름표 조회
분석 목적: [간단한 설명]"

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
                    "For your reference, the current date is {current_date}. The company we want to look at is {ticker}",
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
