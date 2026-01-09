"""
Test script to check what information is available when LLM calls tools
"""
import os
from dotenv import load_dotenv
load_dotenv('.env.dev')

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler

@tool
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a given symbol."""
    return f"The current price of {symbol} is $150.00"

@tool
def get_company_info(symbol: str) -> str:
    """Get company information for a given symbol."""
    return f"{symbol} is a technology company founded in 1975."

class VerboseCallbackHandler(BaseCallbackHandler):
    """Callback handler to capture all LLM interactions"""

    def on_llm_start(self, serialized, prompts, **kwargs):
        print("\n" + "="*50)
        print("[LLM START]")
        print(f"Prompts: {prompts[:500]}...")  # First 500 chars

    def on_llm_end(self, response, **kwargs):
        print("\n" + "="*50)
        print("[LLM END]")
        print(f"Response type: {type(response)}")
        if hasattr(response, 'generations'):
            for gen in response.generations:
                for g in gen:
                    print(f"\n[Generation Content]:")
                    if hasattr(g, 'message'):
                        msg = g.message
                        print(f"  - content: {msg.content[:500] if msg.content else 'None'}...")
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            print(f"  - tool_calls: {msg.tool_calls}")
                        if hasattr(msg, 'additional_kwargs'):
                            print(f"  - additional_kwargs: {msg.additional_kwargs}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print("\n" + "-"*50)
        print("[TOOL START]")
        print(f"  Tool: {serialized.get('name', 'Unknown')}")
        print(f"  Input: {input_str}")

    def on_tool_end(self, output, **kwargs):
        print("[TOOL END]")
        print(f"  Output: {output[:200]}...")

def test_tool_reasoning():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    tools = [get_stock_price, get_company_info]
    llm_with_tools = llm.bind_tools(tools)

    callback = VerboseCallbackHandler()

    print("\n" + "="*70)
    print("TEST 1: Standard tool call (no reasoning)")
    print("="*70)

    response = llm_with_tools.invoke(
        "Analyze TSLA stock. First get the price, then get company info.",
        config={"callbacks": [callback]}
    )

    print(f"\nContent: '{response.content}'")
    print(f"Tool calls: {response.tool_calls}")

    print("\n" + "="*70)
    print("TEST 2: With explicit reasoning request")
    print("="*70)

    # Request reasoning before tool calls
    response2 = llm_with_tools.invoke(
        """Analyze TSLA stock.

IMPORTANT: Before calling any tools, first explain your thought process:
1. What information do you need?
2. Which tools will you use and why?
3. What is your analysis plan?

After explaining, call the necessary tools.""",
        config={"callbacks": [callback]}
    )

    print(f"\nContent (Reasoning): '{response2.content}'")
    print(f"Tool calls: {response2.tool_calls}")

    print("\n" + "="*70)
    print("TEST 3: Using parallel_tool_calls=False to get reasoning")
    print("="*70)

    # Some models output reasoning when not doing parallel tool calls
    llm_sequential = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_sequential_tools = llm_sequential.bind_tools(tools, parallel_tool_calls=False)

    response3 = llm_sequential_tools.invoke(
        "Think step by step. What tools do you need to analyze TSLA? Explain your reasoning, then call the tools.",
        config={"callbacks": [callback]}
    )

    print(f"\nContent: '{response3.content}'")
    print(f"Tool calls: {response3.tool_calls}")

if __name__ == "__main__":
    test_tool_reasoning()
