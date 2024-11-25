def retrieve_top_k_synopsis(movie_id, query_embedding, driver, k=1):
    """
    특정 STORE 노드에 연결된 리뷰 중 유사한 TOP-K 리뷰를 반환합니다.
    """
    query = """
    MATCH (m:Movie {id: $movie_id})-[:HAS_SYNOPSIS]->(s:Synopsis)
    WHERE s.textEmbedding IS NOT NULL
    WITH s.seqId AS seq_Id, gds.similarity.cosine(s.textEmbedding, $query_embedding) AS similarity 
    ORDER BY similarity DESC LIMIT $k
    MATCH (m:Movie {id: $movie_id})-[:HAS_SYNOPSIS]->(synopsis:Synopsis)
    WHERE synopsis.seqId IN [seq_Id - 1 , seq_Id, seq_Id + 1]
    WITH  synopsis
    ORDER BY synopsis.seqId ASC
    WITH REDUCE(output = '', text IN COLLECT(synopsis.text) | output + ' ' + text) AS combinedText
    RETURN combinedText AS synopsis
    """
    with driver.session() as session:
        result = session.run(
            query, movie_id=movie_id, query_embedding=query_embedding, k=k
        )
        record = result.single()
        return {"synopsis": record["synopsis"]}
    
def retrieve_movie_subgraph(movie_id, seq_id, driver):
    """
    특정 STORE 노드에 연결된 리뷰 중 유사한 TOP-K 리뷰를 반환합니다.
    """
    query = """
    MATCH (movie:Movie{id: $movie_id})
    MATCH (movie)-[:HAS_SYNOPSIS]->(synopsis:Synopsis)
    WHERE synopsis.seqId IN [$seq_id - 1 , $seq_id, $seq_id + 1]
    WITH  synopsis
    ORDER BY synopsis.seqId ASC
    WITH REDUCE(output = '', text IN COLLECT(synopsis.text) | output + ' ' + text) AS combinedText
    MATCH (movie:Movie{id: $movie_id})-[:ACTED_BY]->(actor:Actor)
    MATCH (movie)-[:DIRECTED_BY]->(director:Director)
    RETURN 
        movie.title AS MovieTitle, 
        director.name AS director, 
        actor.name AS actor, 
        combinedText AS synopsis
    """

    with driver.session() as session:
        result = session.run(
            query, movie_id=movie_id, seq_id=seq_id
        )
        record = result.single()
        print("result : ", result)
        return {
        "MovieTitle": record["MovieTitle"], "director" : record["director"] if record["director"] else "정보 없음", "actor" : record["actor"] if record["actor"] else ["정보 없음"] , "synopsis" : record["synopsis"] if record["synopsis"] else "정보 없음"}

    

def get_cypher_result_to_str(top_sim_pks, query_embedding, graphdb_driver):
    cypher_result_str = ''
    for r in top_sim_pks:
        r_keys = r.keys()
        one_record_str = ''
        for key in r_keys:
            one_record_str += f"{key} : {str(r[key])[:100]}\n"
        retri_syn = retrieve_top_k_synopsis(r['id'],query_embedding, graphdb_driver, k=1)
        if retri_syn :
            syn_lst = [f"synopsis : {retri_syn['synopsis'][:1000]}".strip()]
            one_record_str += "\n" + '\n'.join(syn_lst) + "\n"
        # one_record_str += f"image_url : https://search.naver.com/search.naver?ssc=tab.image.all&where=image&sm=tab_jum&query={r['MovieTitle']}\n"

        cypher_result_str += one_record_str + '\n'
    return cypher_result_str


def get_candidate_str(candidates, graphdb_driver, rev_num,):          
    drop_dup = []
    for r in candidates:
        if len(drop_dup) == rev_num:
            break
        if r.metadata['id'] not in [d.metadata['id'] for d in drop_dup]:
            print("r.metadata['id'] : ", r.metadata['id'])
            drop_dup.append(r)
    
    print("drop_dup : ", drop_dup)
    candidates_str = ''
    for d in drop_dup:
        print("d : ", d)
        # 영화명
        # 영화 포스터 URL
        
        print('check')
        movie_infos = retrieve_movie_subgraph(d.metadata['id'], d.metadata['SysSeqId'], graphdb_driver)
        print("movie_infos : ", movie_infos)
        if movie_infos:
            print("movie_infos.keys() : ", movie_infos.keys())
            info_lst = [f"{i} : {movie_infos[i][:1000]}".strip() for i in movie_infos.keys()]
            candidates_str += "\n" + '\n'.join(info_lst) + "\n"
        # candidates_str += f"image_url : https://search.naver.com/search.naver?ssc=tab.image.all&where=image&sm=tab_jum&query={d.metadata['MovieTitle']}\n"
        # # 평점
        # ratings_lst = []
        # for platform in ['naver', 'kakao', 'google']:
        #     if (platform in d.metadata['store_Rating']) and (d.metadata['store_Rating'][platform] is not None):
        #         pf_rating = d.metadata['store_Rating'][platform]
        #     else:
        #         continue
        #     if platform in d.metadata['reviewCount'] and (d.metadata['reviewCount'][platform] is not None):
        #         pf_rc = d.metadata['reviewCount'][platform]
        #     else:
        #         continue
        #     ratings_lst.append(f"{platform} {pf_rating}({pf_rc}명)")
        # rating_str = ', '.join(ratings_lst)
        # candidates_str += f"평점 : {rating_str}\n"
        # # 주소
        # if 'store_Addr' in d.metadata:
        #     candidates_str += f"주소 : {d.metadata['store_Addr']}\n"
        # # 음식 유형
        # if 'store_Type' in d.metadata:
        #     candidates_str += f"음식 유형 : {d.metadata['store_Type']}\n"
        # # 방문 목적 top 3 
        # if 'purpose' in d.metadata:
        #     candidates_str += f"방문 목적 top 3 : {d.metadata['purpose']}\n"
        # # 대기 시간 통계
        # if 'wait_time' in d.metadata:
        #     wait_time_str = d.metadata['wait_time'].replace('{', '').replace('}', '').replace('"', '')
        #     candidates_str += f"대기 시간 통계 : {wait_time_str}\n"
        # # 예약 필요 여부
        # if 'use_how' in d.metadata:
        #     use_how_str = d.metadata['use_how'].replace('{', '').replace('}', '').replace('"', '')
        #     candidates_str += f"예약 필요 여부 통계 : {use_how_str}\n"
        # # 메뉴
        # if 'menu' in d.metadata:
        #     candidates_str += f"메뉴 : {d.metadata['menu'][:100]}\n"
        print("candidates_str : ", candidates_str)
        candidates_str += '\n'
    return candidates_str