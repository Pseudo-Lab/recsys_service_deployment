from llm_response.langgraph_graph_state import GraphState
from neo4j.exceptions import ServiceUnavailable, CypherSyntaxError

def retrieve_for_search_cypher(graphdb_driver, state:GraphState):
    print(f"[Search] Cypher result : neo4j 추출 결과".ljust(100, '='))
    try:
        records, summary, keys = graphdb_driver.execute_query(state['t2c_for_search'])
        print(f"records : \n")

        record_dict_lst = []
        for r in records:
            record_dict_lst.append(dict(r))
            if 's' in r.keys():
                print(f"{dict(r['s'])}")
                print()
        state['record_dict_lst'] = record_dict_lst
    except ServiceUnavailable:
        state['record_dict_lst'] = """Retrieval failed due to a temporary database connection issue.
        Please try again later or provide a custom message apologizing for the unavailability of information."""
    except CypherSyntaxError:
        state["record_dict_lst"] = """A syntax error occurred in text-to-Cypher. Provide guidance kindly based on follows.
        'We performed a database search for the requested query, but there are no results due to an error in the extraction process.'"""
    print(f"record_dict_lst : {record_dict_lst}")
    return state
