from neo4j import GraphDatabase
import os
from guiderec_config import CONFIG
import timeit


graphdb_driver = GraphDatabase.driver(uri=CONFIG.neo4j_url,
                                      auth=(
                                          CONFIG.neo4j_user,
                                          CONFIG.neo4j_password
                                          )
                                        )


def get_ratings_str(d):
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
    return rating_str


class DotDict(dict):
    """딕셔너리 키를 속성처럼 접근할 수 있도록 하는 클래스"""
    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]


def get_init_recomm_query():
    example_query1 = "60대 부모님과 가기 좋은 애월읍 흑돼지 맛집 추천해줘"
    example_query2 = "애월읍에서 60대가 가장 많이 가는 카페는?"
    return example_query1, example_query2
