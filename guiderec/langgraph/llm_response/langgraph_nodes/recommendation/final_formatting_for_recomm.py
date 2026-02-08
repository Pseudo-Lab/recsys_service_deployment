from typing import List
from llm_response.langgraph_graph_state import GraphState
from llm_response.utils.recomm_final_formatting.star_formatting import (
    get_ratings_str_for_node,
)
from collections import defaultdict


REVIEW_HTML = """<div style="background-color: #f9f9f9; padding: 10px 15px; border-left: 4px solid #FFA500; margin-bottom: 20px;">
        <p style="font-style: italic; margin: 0;">
            <strong>리뷰: "{review}"</strong>
        </p>
    </div>"""


def get_image_html_str(node):
    image_html = '<div style="display: flex; gap: 10px;">'
    if node["image_url_naver"]:
        image_html += f"""\n<img src="{node['image_url_naver']}" alt="Naver Image" style="width: 200px; height: 200px; object-fit: cover; border-radius: 10%;">"""
    if node["image_url_kakao"]:
        image_html += f"""\n<img src="{node['image_url_kakao']}" alt="Naver Image" style="width: 200px; height: 200px; object-fit: cover; border-radius: 10%;">"""
    if node["image_url_google"]:
        image_html += f"""\n<img src="{node['image_url_google']}" alt="Naver Image" style="width: 200px; height: 200px; object-fit: cover; border-radius: 10%;">"""
    image_html += "</div>"
    return image_html


def final_formatting_for_recomm(graphdb_driver, state: GraphState):
    print(f"Final formatting for recomm".ljust(100, "="))
    state["final_answer"] = ""

    pk_store_cypher = """MATCH (s:STORE) WHERE s.pk = {pk} RETURN s"""
    print(
        f"state['selected_recommendations']['recommendations'] : {state['selected_recommendations']['recommendations']}"
    )
    pk_lst = [
        str(r["pk"]) for r in state["selected_recommendations"]["recommendations"]
    ]
    retrieved_stores_nodes = []

    for pk in pk_lst:
        result = graphdb_driver.execute_query(pk_store_cypher.format(pk=pk))
        retrieved_stores_nodes.append(result.records[0]["s"])

    state["final_answer"] += "\n"
    for rank, (node, pk_desc) in enumerate(
        zip(
            retrieved_stores_nodes, state["selected_recommendations"]["recommendations"]
        ),
        start=1,
    ):
        state["final_answer"] += f"#### {rank}. {node['MCT_NM']}" + "\n"
        image_html = get_image_html_str(node)
        state["final_answer"] += image_html + "\n"
        state["final_answer"] += "\n"
        state["final_answer"] += get_ratings_str_for_node(node) + "\n"
        state["final_answer"] += REVIEW_HTML.format(review=pk_desc["review"]) + "\n"
        state["final_answer"] += "\n" + pk_desc["desc"] + "\n"
        state["final_answer"] += "\n" + "---" + "\n"

    state["final_answer"] += (
        state["selected_recommendations"]["decorational_mention_end"] + "\n"
    )
    return state
