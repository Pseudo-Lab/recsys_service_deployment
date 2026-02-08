"""
LangGraph 구조 출력 스크립트
trading_agent의 그래프 노드와 엣지 구조를 시각화합니다.

의존성 없이 독립적으로 실행 가능합니다.
"""

from typing import TypedDict, List, Any, Annotated
from operator import add

try:
    from langgraph.graph import END, StateGraph, START
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    print("Warning: langgraph not installed. Using static structure only.")


# 간단한 State 정의 (실제 AgentState 구조 모사)
class SimpleState(TypedDict):
    messages: List[Any]


def create_mock_graph():
    """
    실제 GraphSetup과 동일한 구조로 그래프를 생성하되,
    실제 LLM/Tool 없이 mock으로 대체
    """
    selected_analysts = ["market", "social", "news", "fundamentals"]

    # Mock 노드 함수들
    def mock_node(state):
        return state

    # 간단한 condition 함수들
    def should_continue_analyst(state):
        return "clear"  # 항상 clear로

    def should_continue_debate(state):
        return "Research Manager"

    def should_continue_risk(state):
        return "Risk Judge"

    workflow = StateGraph(SimpleState)

    # Analyst 노드 추가
    for analyst_type in selected_analysts:
        workflow.add_node(f"{analyst_type.capitalize()} Analyst", mock_node)
        workflow.add_node(f"Msg Clear {analyst_type.capitalize()}", mock_node)
        workflow.add_node(f"tools_{analyst_type}", mock_node)

    # 기타 노드들
    workflow.add_node("Bull Researcher", mock_node)
    workflow.add_node("Bear Researcher", mock_node)
    workflow.add_node("Research Manager", mock_node)
    workflow.add_node("Trader", mock_node)
    workflow.add_node("Risky Analyst", mock_node)
    workflow.add_node("Neutral Analyst", mock_node)
    workflow.add_node("Safe Analyst", mock_node)
    workflow.add_node("Risk Judge", mock_node)

    # 엣지 정의
    # START -> 첫번째 analyst
    first_analyst = selected_analysts[0]
    workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

    # Analyst 시퀀스 연결
    for i, analyst_type in enumerate(selected_analysts):
        current_analyst = f"{analyst_type.capitalize()} Analyst"
        current_tools = f"tools_{analyst_type}"
        current_clear = f"Msg Clear {analyst_type.capitalize()}"

        # Conditional edges (tools or clear)
        workflow.add_conditional_edges(
            current_analyst,
            should_continue_analyst,
            {"tools": current_tools, "clear": current_clear},
        )
        workflow.add_edge(current_tools, current_analyst)

        # 다음 analyst 또는 Bull Researcher로 연결
        if i < len(selected_analysts) - 1:
            next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
            workflow.add_edge(current_clear, next_analyst)
        else:
            workflow.add_edge(current_clear, "Bull Researcher")

    # Researcher debate loop
    workflow.add_conditional_edges(
        "Bull Researcher",
        should_continue_debate,
        {"Bear Researcher": "Bear Researcher", "Research Manager": "Research Manager"},
    )
    workflow.add_conditional_edges(
        "Bear Researcher",
        should_continue_debate,
        {"Bull Researcher": "Bull Researcher", "Research Manager": "Research Manager"},
    )

    workflow.add_edge("Research Manager", "Trader")
    workflow.add_edge("Trader", "Risky Analyst")

    # Risk analyst debate loop
    workflow.add_conditional_edges(
        "Risky Analyst",
        should_continue_risk,
        {"Safe Analyst": "Safe Analyst", "Risk Judge": "Risk Judge"},
    )
    workflow.add_conditional_edges(
        "Safe Analyst",
        should_continue_risk,
        {"Neutral Analyst": "Neutral Analyst", "Risk Judge": "Risk Judge"},
    )
    workflow.add_conditional_edges(
        "Neutral Analyst",
        should_continue_risk,
        {"Risky Analyst": "Risky Analyst", "Risk Judge": "Risk Judge"},
    )

    workflow.add_edge("Risk Judge", END)

    return workflow


def print_graph_structure(workflow: StateGraph):
    """그래프 구조를 출력"""
    print("=" * 60)
    print("LangGraph Structure - Trading Agents")
    print("=" * 60)

    # 노드 출력
    print("\n[NODES]")
    print("-" * 40)
    nodes = list(workflow.nodes.keys())
    for i, node in enumerate(nodes, 1):
        print(f"  {i:2}. {node}")
    print(f"\n  Total: {len(nodes)} nodes")

    # 일반 엣지 출력
    print("\n[EDGES (Direct)]")
    print("-" * 40)
    if hasattr(workflow, '_edges'):
        for source, target in workflow._edges.items():
            print(f"  {source} -> {target}")

    # Conditional edges
    print("\n[EDGES (Conditional)]")
    print("-" * 40)
    if hasattr(workflow, '_branches'):
        for source, branches in workflow._branches.items():
            for branch_name, branch in branches.items():
                if hasattr(branch, 'ends') and branch.ends:
                    targets = list(branch.ends.values())
                    print(f"  {source} -> {targets}")

    print("\n" + "=" * 60)


def print_graph_ascii():
    """ASCII 형태로 그래프 흐름 출력"""
    print("\n[GRAPH FLOW - ASCII]")
    print("=" * 60)

    flow = """
    START
      |
      v
    +-----------------------------------------------------------+
    |                    ANALYST PHASE                          |
    |  +----------------+  +----------------+  +---------------+|
    |  |Market Analyst  |->|Social Analyst  |->|News Analyst   ||
    |  +-------+--------+  +-------+--------+  +-------+-------+|
    |          |                   |                   |        |
    |     [tools]             [tools]             [tools]       |
    |          |                   |                   |        |
    |  +-------v--------+  +-------v--------+  +-------v-------+|
    |  |Msg Clear Market|  |Msg Clear Social|  |Msg Clear News ||
    |  +----------------+  +----------------+  +-------+-------+|
    |                                                  |        |
    |  +--------------------+                          |        |
    |  |Fundamentals Analyst|<-------------------------+        |
    |  +---------+----------+                                   |
    |            |                                              |
    |       [tools]                                             |
    |            |                                              |
    |  +---------v--------------+                               |
    |  |Msg Clear Fundamentals  |                               |
    |  +---------+--------------+                               |
    +------------|----------------------------------------------+
                 |
                 v
    +-----------------------------------------------------------+
    |                    RESEARCH PHASE                         |
    |     +----------------+      +----------------+            |
    |     |Bull Researcher |<---->|Bear Researcher |  (debate)  |
    |     +-------+--------+      +----------------+            |
    |             |                                             |
    |             v                                             |
    |     +------------------+                                  |
    |     |Research Manager  |                                  |
    |     +--------+---------+                                  |
    +--------------|--------------------------------------------+
                   |
                   v
    +-----------------------------------------------------------+
    |                    TRADING PHASE                          |
    |             +----------------+                            |
    |             |    Trader      |                            |
    |             +-------+--------+                            |
    +---------------------|-------------------------------------+
                          |
                          v
    +-----------------------------------------------------------+
    |                    RISK PHASE                             |
    |  +---------------+  +---------------+  +---------------+  |
    |  |Risky Analyst  |<-|Neutral Analyst|<-|Safe Analyst   |  |
    |  |               |->|               |->|               |  |
    |  +---------------+  +---------------+  +---------------+  |
    |              (risk debate loop)                           |
    |                        |                                  |
    |                        v                                  |
    |               +----------------+                          |
    |               |  Risk Judge    |                          |
    |               +-------+--------+                          |
    +------------------------|----------------------------------+
                             |
                             v
                           END
    """
    print(flow)


def try_mermaid_export(compiled_graph):
    """Mermaid 다이어그램으로 export 시도"""
    print("\n[MERMAID DIAGRAM]")
    print("=" * 60)
    try:
        graph = compiled_graph.get_graph()
        mermaid = graph.draw_mermaid()
        print(mermaid)
        print("\n(Copy above to https://mermaid.live for visualization)")
    except Exception as e:
        print(f"Mermaid export not available: {e}")
        print("(Install with: pip install 'langgraph[all]' or 'pygraphviz')")


def print_static_structure():
    """langgraph 없이도 정적 구조 출력"""
    print("=" * 60)
    print("LangGraph Structure - Trading Agents (Static)")
    print("=" * 60)

    nodes = [
        "Market Analyst", "tools_market", "Msg Clear Market",
        "Social Analyst", "tools_social", "Msg Clear Social",
        "News Analyst", "tools_news", "Msg Clear News",
        "Fundamentals Analyst", "tools_fundamentals", "Msg Clear Fundamentals",
        "Bull Researcher", "Bear Researcher", "Research Manager",
        "Trader",
        "Risky Analyst", "Neutral Analyst", "Safe Analyst", "Risk Judge"
    ]

    print("\n[NODES]")
    print("-" * 40)
    for i, node in enumerate(nodes, 1):
        print(f"  {i:2}. {node}")
    print(f"\n  Total: {len(nodes)} nodes")

    print("\n[EDGES]")
    print("-" * 40)
    edges = [
        ("START", "Market Analyst"),
        ("Market Analyst", "tools_market | Msg Clear Market"),
        ("tools_market", "Market Analyst"),
        ("Msg Clear Market", "Social Analyst"),
        ("Social Analyst", "tools_social | Msg Clear Social"),
        ("tools_social", "Social Analyst"),
        ("Msg Clear Social", "News Analyst"),
        ("News Analyst", "tools_news | Msg Clear News"),
        ("tools_news", "News Analyst"),
        ("Msg Clear News", "Fundamentals Analyst"),
        ("Fundamentals Analyst", "tools_fundamentals | Msg Clear Fundamentals"),
        ("tools_fundamentals", "Fundamentals Analyst"),
        ("Msg Clear Fundamentals", "Bull Researcher"),
        ("Bull Researcher", "Bear Researcher | Research Manager"),
        ("Bear Researcher", "Bull Researcher | Research Manager"),
        ("Research Manager", "Trader"),
        ("Trader", "Risky Analyst"),
        ("Risky Analyst", "Safe Analyst | Risk Judge"),
        ("Safe Analyst", "Neutral Analyst | Risk Judge"),
        ("Neutral Analyst", "Risky Analyst | Risk Judge"),
        ("Risk Judge", "END"),
    ]
    for src, dst in edges:
        print(f"  {src} -> {dst}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Trading Agents - LangGraph Structure Visualizer")
    print("=" * 60)

    if HAS_LANGGRAPH:
        try:
            workflow = create_mock_graph()
            print_graph_structure(workflow)

            # Compile the graph
            compiled = workflow.compile()

            # Mermaid 다이어그램 시도
            try_mermaid_export(compiled)

        except Exception as e:
            import traceback
            print(f"\nError creating graph: {e}")
            traceback.print_exc()
            print("\nFalling back to static structure...")
            print_static_structure()
    else:
        print_static_structure()

    # ASCII 흐름도는 항상 출력
    print_graph_ascii()

    print("\nDone!")
