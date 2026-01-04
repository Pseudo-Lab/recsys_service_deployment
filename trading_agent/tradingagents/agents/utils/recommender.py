
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from threading import Lock

class RecommendedTickers(BaseModel):
    tickers: List[str] = Field(description="List of 3 recommended stock tickers (e.g. ['AMD', 'TSM', 'INTC'])")
    reasoning: str = Field(description="Brief explanation of why these were chosen based on the profile")

class TickerRecommender:
    def __init__(self, llm_model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.7)
        self.parser = JsonOutputParser(pydantic_object=RecommendedTickers)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Senior Portfolio Strategist. Your goal is to recommend 3 related or complementary stock tickers based on a Target Ticker and the User's Investment Profile.\n"
                       "1. Identify the sector/industry of the Target Ticker.\n"
                       "2. Find 3 other companies that would make sense for the user to investigate next.\n"
                       "   - Competitors (if they like the sector)\n"
                       "   - Supply chain partners (upstream/downstream)\n"
                       "   - Similar growth/value plays in adjacent sectors\n"
                       "3. STRICTLY ADHERE to the User's Profile (Risk Tolerance, Favorites, Avoids).\n"
                       "   - If they avoid 'Semiconductors', do NOT recommend chips, even if the target is NVDA. Find a related tech play instead.\n"
                       "4. Return exactly 3 valid US Market Tickers.\n"
                       "{format_instructions}"),
            ("user", "Target Ticker: {ticker}\nUser Profile: {profile_summary}")
        ])

    def get_recommendations(self, ticker: str, profile_summary: str) -> RecommendedTickers:
        try:
            print(f"DEBUG: Generating recommendations for {ticker}...")
            chain = self.prompt | self.llm | self.parser
            result = chain.invoke({
                "ticker": ticker,
                "profile_summary": profile_summary,
                "format_instructions": self.parser.get_format_instructions()
            })
            return RecommendedTickers(**result)
        except Exception as e:
            print(f"ERROR in Recommender: {e}")
            # Fallback
            return RecommendedTickers(tickers=[], reasoning="Error generating recommendations.")

# Global instance
recommender = TickerRecommender()
