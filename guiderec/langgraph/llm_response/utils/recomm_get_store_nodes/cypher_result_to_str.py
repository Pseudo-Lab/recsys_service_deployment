from graphrag.graph_search import retrieve_top_k_stores_by_review_graph_embedding, process_review_node
from concurrent.futures import ThreadPoolExecutor, as_completed

def retrieve_top_k_reviews(store_pk, query_embedding, driver, k=2):
    """
    특정 STORE 노드에 연결된 리뷰 중 유사한 TOP-K 리뷰를 반환합니다.
    """
    query = """
    MATCH (s:STORE {pk: $store_pk})-[:HAS_REVIEW]->(r:Review)
    WHERE r.textEmbedding IS NOT NULL
    RETURN r.text AS text, gds.similarity.cosine(r.textEmbedding, $query_embedding) AS similarity
    ORDER BY similarity DESC
    LIMIT $k
    """
    with driver.session() as session:
        result = session.run(
            query, store_pk=store_pk, query_embedding=query_embedding, k=k
        )
        return [
            {"text": record["text"], "similarity": record["similarity"]}
            for record in result
        ]
    

def get_cypher_result_to_str(candidates_2nd, query_embedding, graphdb_driver, k=2):
    cypher_result_str = ''
    for r in candidates_2nd:
        r_keys = r.keys()
        one_record_str = ''
        for key in r_keys:
            one_record_str += f"{key} : {str(r[key])[:100]}\n"
            if key == 'pk':
                reviews = retrieve_top_k_reviews(r[key], query_embedding, graphdb_driver, k=k)
                if reviews:
                    reviews_lst = [f"{ri}. {review['text'][:100]}".strip() for ri, review in enumerate(reviews, start=1)]
                    one_record_str += "Review : \n" + '\n'.join(reviews_lst) + "\n"
        cypher_result_str += one_record_str + '\n'
    return cypher_result_str


def get_candidate_str(candidates, query_embedding, graphdb_driver, subtype, rev_num, grp_num, review_k):          
    drop_dup = []
    for r in candidates:
        if len(drop_dup) == rev_num:
            break
        if r.metadata['pk'] not in [d.metadata['pk'] for d in drop_dup]:
            drop_dup.append(r)
    if subtype == 'purpose_and_visit_with' :
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_review_node, review, 1)
                for review in candidates
            ]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        for grp_store in results[:grp_num]:
            print("grp_store : ", grp_store)
            drop_dup.append(grp_store)
        
    candidates_str = ''
    for d in drop_dup:
        # 가게명
        candidates_str += f"가게명 : {d.metadata['storeName']}\n"
        # pk
        candidates_str += f"pk : {d.metadata['pk']}\n"
        # 리뷰
        reviews = retrieve_top_k_reviews(d.metadata['pk'], query_embedding, graphdb_driver, k=review_k)
        if reviews:
            reviews_lst = [f"{ri}. {review['text'][:100]}".strip() for ri, review in enumerate(reviews, start=1)]
            candidates_str += "Review : \n" + '\n'.join(reviews_lst) + "\n"
        # 평점
        ratings_lst = []
        for platform in ['naver', 'kakao', 'google']:
            if (platform in d.metadata['store_Rating']) and (d.metadata['store_Rating'][platform] is not None):
                pf_rating = d.metadata['store_Rating'][platform]
            else:
                continue
            if platform in d.metadata['reviewCount'] and (d.metadata['reviewCount'][platform] is not None):
                pf_rc = d.metadata['reviewCount'][platform]
            else:
                continue
            ratings_lst.append(f"{platform} {pf_rating}({pf_rc}명)")
        rating_str = ', '.join(ratings_lst)
        candidates_str += f"평점 : {rating_str}\n"
        # 주소
        if 'store_Addr' in d.metadata:
            candidates_str += f"주소 : {d.metadata['store_Addr']}\n"
        # 음식 유형
        if 'store_Type' in d.metadata:
            candidates_str += f"음식 유형 : {d.metadata['store_Type']}\n"
        # 방문 목적 top 3 
        if 'purpose' in d.metadata:
            candidates_str += f"방문 목적 top 3 : {d.metadata['purpose']}\n"
        # 대기 시간 통계
        if 'wait_time' in d.metadata:
            wait_time_str = d.metadata['wait_time'].replace('{', '').replace('}', '').replace('"', '')
            candidates_str += f"대기 시간 통계 : {wait_time_str}\n"
        # 예약 필요 여부
        if 'use_how' in d.metadata:
            use_how_str = d.metadata['use_how'].replace('{', '').replace('}', '').replace('"', '')
            candidates_str += f"예약 필요 여부 통계 : {use_how_str}\n"
        # 메뉴
        if 'menu' in d.metadata:
            candidates_str += f"메뉴 : {d.metadata['menu'][:100]}\n"

        candidates_str += '\n'
    return candidates_str