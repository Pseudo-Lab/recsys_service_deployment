# Token Usage Tracker for TradingAgents
# Tracks token usage and calculates costs for different LLM providers

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import threading


# Model pricing (per 1K tokens, in USD)
# Updated as of January 2025
MODEL_PRICING = {
    # OpenAI Models
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-2024-11-20": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "o1": {"input": 0.015, "output": 0.06},
    "o1-mini": {"input": 0.003, "output": 0.012},
    "o1-preview": {"input": 0.015, "output": 0.06},
    "o3-mini": {"input": 0.00110, "output": 0.0044},
    "o4-mini": {"input": 0.00110, "output": 0.0044},  # Assuming same as o3-mini

    # OpenAI Embeddings
    "text-embedding-3-small": {"input": 0.00002, "output": 0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0},
    "text-embedding-ada-002": {"input": 0.0001, "output": 0},

    # Anthropic Models
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet-latest": {"input": 0.003, "output": 0.015},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},

    # Google Models
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},

    # Default fallback
    "default": {"input": 0.001, "output": 0.002},
}


@dataclass
class TokenUsage:
    """Token usage for a single LLM call"""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    agent_name: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def calculate_cost(self) -> float:
        """Calculate cost in USD"""
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING["default"])
        input_cost = (self.input_tokens / 1000) * pricing["input"]
        output_cost = (self.output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost


class TokenUsageTracker:
    """Tracks token usage across multiple LLM calls"""

    def __init__(self):
        self._usages: list[TokenUsage] = []
        self._lock = threading.Lock()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0

    def add_usage(self, model: str, input_tokens: int, output_tokens: int, agent_name: str = ""):
        """Add a new token usage record"""
        usage = TokenUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            agent_name=agent_name
        )

        with self._lock:
            self._usages.append(usage)
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_cost += usage.calculate_cost()

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_cost_krw(self) -> float:
        """Convert USD to KRW (approximate rate)"""
        return self._total_cost * 1450  # Approximate USD/KRW rate

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all token usage"""
        with self._lock:
            # Group by model
            by_model: Dict[str, Dict[str, int]] = {}
            for usage in self._usages:
                if usage.model not in by_model:
                    by_model[usage.model] = {"input": 0, "output": 0, "cost": 0}
                by_model[usage.model]["input"] += usage.input_tokens
                by_model[usage.model]["output"] += usage.output_tokens
                by_model[usage.model]["cost"] += usage.calculate_cost()

            # Group by agent
            by_agent: Dict[str, Dict[str, Any]] = {}
            for usage in self._usages:
                agent = usage.agent_name or "unknown"
                if agent not in by_agent:
                    by_agent[agent] = {"input": 0, "output": 0, "cost": 0, "calls": 0}
                by_agent[agent]["input"] += usage.input_tokens
                by_agent[agent]["output"] += usage.output_tokens
                by_agent[agent]["cost"] += usage.calculate_cost()
                by_agent[agent]["calls"] += 1

            return {
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "total_tokens": self.total_tokens,
                "total_cost_usd": round(self._total_cost, 6),
                "total_cost_krw": round(self.total_cost_krw, 2),
                "num_calls": len(self._usages),
                "by_model": by_model,
                "by_agent": by_agent,
            }

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current stats for streaming updates"""
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self._total_cost, 6),
            "cost_krw": round(self.total_cost_krw, 2),
            "num_calls": len(self._usages),
        }

    def reset(self):
        """Reset all tracking data"""
        with self._lock:
            self._usages.clear()
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._total_cost = 0.0
