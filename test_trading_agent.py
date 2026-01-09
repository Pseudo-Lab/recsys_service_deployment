"""
Test script for Trading Agent to debug indicator fetching issues
"""
import os
import sys
import django
from datetime import datetime

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from trading_agent.tradingagents.graph.trading_graph import TradingAgentsGraph
from trading_agent.tradingagents.default_config import DEFAULT_CONFIG

def test_trading_agent():
    """Test Trading Agent with NVDA stock"""
    print("=" * 80)
    print("TRADING AGENT TEST")
    print("=" * 80)

    ticker = "NVDA"
    date_str = "2025-12-24"

    print(f"\nTesting with ticker: {ticker}")
    print(f"Date: {date_str}")
    print(f"Config: {DEFAULT_CONFIG}")
    print("\n" + "=" * 80)

    try:
        # Initialize TradingAgents with debug mode
        print("\n[1] Initializing TradingAgentsGraph...")
        ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())
        print("✓ TradingAgentsGraph initialized successfully")

        # Run the analysis
        print(f"\n[2] Running analysis for {ticker} on {date_str}...")
        final_state, decision = ta.propagate(ticker, date_str)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nDecision: {decision}")
        print(f"\nFinal Trade Decision: {final_state.get('final_trade_decision', 'N/A')}")
        print(f"\nTrader Investment Plan:\n{final_state.get('trader_investment_plan', 'N/A')}")

        return final_state, decision

    except Exception as e:
        import traceback
        print("\n" + "=" * 80)
        print("ERROR OCCURRED")
        print("=" * 80)
        print(f"\nError: {e}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    test_trading_agent()
