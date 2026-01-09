"""
Tests for state management in trading agents.
These tests ensure that history is properly maintained as lists, not strings.
"""
import pytest
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from trading_agent.tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from trading_agent.tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from trading_agent.tradingagents.agents.risk_mgmt.conservative_debator import create_safe_debator
from trading_agent.tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from trading_agent.tradingagents.agents.risk_mgmt.aggresive_debator import create_risky_debator


class MockLLM:
    """Mock LLM for testing"""
    def invoke(self, prompt):
        class MockResponse:
            content = "Test response from mock LLM"
        return MockResponse()


class MockMemory:
    """Mock memory for testing"""
    def get_memories(self, situation, n_matches=2):
        return [{"recommendation": "Test memory 1"}, {"recommendation": "Test memory 2"}]


def test_bull_researcher_history_is_list():
    """Test that bull_researcher properly maintains history as a list"""
    llm = MockLLM()
    memory = MockMemory()

    bull_node = create_bull_researcher(llm, memory)

    # Initial state matching views.py - NO count field initially
    state = {
        "investment_debate_state": {
            "history": [],
            "bull_history": [],
            "bear_history": [],
            "current_response": "Test bear response",
            "judge_decision": "",
        },
        "market_report": "Test market report",
        "sentiment_report": "Test sentiment",
        "news_report": "Test news",
        "fundamentals_report": "Test fundamentals",
    }

    # Execute the bull node
    result = bull_node(state)

    # Verify that history is still a list after concatenation
    assert isinstance(result["investment_debate_state"]["history"], list)
    assert isinstance(result["investment_debate_state"]["bull_history"], list)
    assert len(result["investment_debate_state"]["history"]) == 1
    assert len(result["investment_debate_state"]["bull_history"]) == 1
    # Verify count was initialized from 0 (when missing)
    assert result["investment_debate_state"]["count"] == 1

    # Execute again with non-empty history to test concatenation
    result2 = bull_node({**state, "investment_debate_state": result["investment_debate_state"]})
    assert isinstance(result2["investment_debate_state"]["history"], list)
    assert len(result2["investment_debate_state"]["history"]) == 2
    # Verify count increments properly
    assert result2["investment_debate_state"]["count"] == 2


def test_bear_researcher_history_is_list():
    """Test that bear_researcher properly maintains history as a list"""
    llm = MockLLM()
    memory = MockMemory()

    bear_node = create_bear_researcher(llm, memory)

    # Initial state matching views.py - NO count field initially
    state = {
        "investment_debate_state": {
            "history": [],
            "bear_history": [],
            "bull_history": [],
            "current_response": "Test bull response",
            "judge_decision": "",
        },
        "market_report": "Test market report",
        "sentiment_report": "Test sentiment",
        "news_report": "Test news",
        "fundamentals_report": "Test fundamentals",
    }

    result = bear_node(state)

    assert isinstance(result["investment_debate_state"]["history"], list)
    assert isinstance(result["investment_debate_state"]["bear_history"], list)
    assert len(result["investment_debate_state"]["history"]) == 1
    # Verify count was initialized properly
    assert result["investment_debate_state"]["count"] == 1


def test_conservative_debator_history_is_list():
    """Test that conservative debator properly maintains history as a list"""
    llm = MockLLM()

    safe_node = create_safe_debator(llm)

    # Initial state matching views.py - NO count field initially
    state = {
        "risk_debate_state": {
            "history": [],
            "safe_history": [],
            "risky_history": [],
            "neutral_history": [],
            "current_risky_response": "Test risky",
            "current_neutral_response": "Test neutral",
            "judge_decision": "",
        },
        "market_report": "Test market report",
        "sentiment_report": "Test sentiment",
        "news_report": "Test news",
        "fundamentals_report": "Test fundamentals",
        "trader_investment_plan": "Test plan",
    }

    result = safe_node(state)

    assert isinstance(result["risk_debate_state"]["history"], list)
    assert isinstance(result["risk_debate_state"]["safe_history"], list)
    assert len(result["risk_debate_state"]["history"]) == 1
    assert len(result["risk_debate_state"]["safe_history"]) == 1
    # Verify count was initialized properly
    assert result["risk_debate_state"]["count"] == 1


def test_neutral_debator_history_is_list():
    """Test that neutral debator properly maintains history as a list"""
    llm = MockLLM()

    neutral_node = create_neutral_debator(llm)

    # Initial state matching views.py - NO count field initially
    state = {
        "risk_debate_state": {
            "history": [],
            "neutral_history": [],
            "risky_history": [],
            "safe_history": [],
            "current_risky_response": "Test risky",
            "current_safe_response": "Test safe",
            "judge_decision": "",
        },
        "market_report": "Test market report",
        "sentiment_report": "Test sentiment",
        "news_report": "Test news",
        "fundamentals_report": "Test fundamentals",
        "trader_investment_plan": "Test plan",
    }

    result = neutral_node(state)

    assert isinstance(result["risk_debate_state"]["history"], list)
    assert isinstance(result["risk_debate_state"]["neutral_history"], list)
    assert len(result["risk_debate_state"]["history"]) == 1
    # Verify count was initialized properly
    assert result["risk_debate_state"]["count"] == 1


def test_risky_debator_history_is_list():
    """Test that risky debator properly maintains history as a list"""
    llm = MockLLM()

    risky_node = create_risky_debator(llm)

    # Initial state matching views.py - NO count field initially
    state = {
        "risk_debate_state": {
            "history": [],
            "risky_history": [],
            "safe_history": [],
            "neutral_history": [],
            "current_safe_response": "Test safe",
            "current_neutral_response": "Test neutral",
            "judge_decision": "",
        },
        "market_report": "Test market report",
        "sentiment_report": "Test sentiment",
        "news_report": "Test news",
        "fundamentals_report": "Test fundamentals",
        "trader_investment_plan": "Test plan",
    }

    result = risky_node(state)

    assert isinstance(result["risk_debate_state"]["history"], list)
    assert isinstance(result["risk_debate_state"]["risky_history"], list)
    assert len(result["risk_debate_state"]["history"]) == 1
    # Verify count was initialized properly
    assert result["risk_debate_state"]["count"] == 1


def test_multiple_rounds_maintains_list_type():
    """Test that multiple rounds of debate maintain list type"""
    llm = MockLLM()

    safe_node = create_safe_debator(llm)
    neutral_node = create_neutral_debator(llm)
    risky_node = create_risky_debator(llm)

    # Initial state matching views.py - NO count field initially
    state = {
        "risk_debate_state": {
            "history": [],
            "risky_history": [],
            "safe_history": [],
            "neutral_history": [],
            "current_risky_response": "",
            "current_safe_response": "",
            "current_neutral_response": "",
            "judge_decision": "",
        },
        "market_report": "Test market report",
        "sentiment_report": "Test sentiment",
        "news_report": "Test news",
        "fundamentals_report": "Test fundamentals",
        "trader_investment_plan": "Test plan",
    }

    # Round 1: Safe speaks
    result1 = safe_node({**state, "risk_debate_state": {**state["risk_debate_state"]}})
    assert isinstance(result1["risk_debate_state"]["history"], list)
    assert len(result1["risk_debate_state"]["history"]) == 1
    assert result1["risk_debate_state"]["count"] == 1  # Count initialized

    # Round 2: Neutral speaks
    result2 = neutral_node({**state, "risk_debate_state": result1["risk_debate_state"]})
    assert isinstance(result2["risk_debate_state"]["history"], list)
    assert len(result2["risk_debate_state"]["history"]) == 2
    assert result2["risk_debate_state"]["count"] == 2  # Count incremented

    # Round 3: Risky speaks
    result3 = risky_node({**state, "risk_debate_state": result2["risk_debate_state"]})
    assert isinstance(result3["risk_debate_state"]["history"], list)
    assert len(result3["risk_debate_state"]["history"]) == 3
    assert result3["risk_debate_state"]["count"] == 3  # Count incremented again

    # Verify no TypeError or KeyError was raised during concatenation
    assert True  # If we got here, no errors occurred
