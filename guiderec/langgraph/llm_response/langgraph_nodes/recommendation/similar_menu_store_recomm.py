from collections import defaultdict
from llm_response.langgraph_graph_state import GraphState

def similar_menu_store_recomm(graphdb_driver, state:GraphState):
    state['sim_recomm_pks'] = defaultdict(list)

    similar_menu_cypher = """// 1. 최신 날짜를 가진 STORE만 추출
    CALL {{
    MATCH (s:STORE)
    WITH s.MCT_NM AS name, max(s.OP_YMD) AS latest_date
    RETURN collect({{name: name, date: latest_date}}) AS valid_pairs
    }}

    // 2. 기준 pk에 해당하는 최신 가게들만 필터
    WITH valid_pairs
    MATCH (s1:STORE)
    WHERE s1.pk = {pk}
    AND {{name: s1.MCT_NM, date: s1.OP_YMD}} IN valid_pairs

    // 3. s1과 공통 메뉴를 가진 최신 가게 s2 추출
    MATCH (s1)-[:HAS_VISIT_KEYWORD]->(m:menu)<-[:HAS_VISIT_KEYWORD]-(s2:STORE)
    WHERE s1 <> s2
    AND {{name: s2.MCT_NM, date: s2.OP_YMD}} IN valid_pairs
    WITH s1, s2, collect(m) AS shared_menus, count(m) AS shared_count
    ORDER BY shared_count DESC
    LIMIT 5

    // 4. 공유된 메뉴와 관계를 다시 매칭해 시각화
    UNWIND shared_menus AS menu
    MATCH (s1)-[r1:HAS_VISIT_KEYWORD]->(menu)<-[r2:HAS_VISIT_KEYWORD]-(s2)
    RETURN DISTINCT s1, s2, menu, r1, r2
    """
    for rec in state['selected_recommendations']['recommendations']:
        sims = graphdb_driver.execute_query(similar_menu_cypher.format(pk=rec['pk']))
        for triplet in sims.records:
            state['sim_recomm_pks'][rec['pk']].append((triplet['menu']['name'], triplet['s2']))

    for rec_pk in state['sim_recomm_pks'].keys():
        print(f"rec pk : {rec_pk}")
        for sim in state['sim_recomm_pks'][rec_pk]:
            print(f"{sim[0]} - {sim[1]['MCT_NM']}({sim[1]['pk']})")
            
    return state