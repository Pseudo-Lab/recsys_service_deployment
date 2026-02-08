# 기존버전
retrievalQuery = """
MATCH (node)<-[:HAS_REVIEW]-(store)
RETURN node.text AS text,
       store AS store,
       score,
       {
         reviewText: node.text,
         storeName: store.MCT_NM,
         storeType: store.MCT_TYPE,	
         storeAddress: store.ADDR,
         storeImage: store.image_url,
         storeRating: store.rating,
         score: score
       } AS metadata
"""

# 수정버전(선진) 1020
retrievalQuery_v2 = """
MATCH (node)<-[:HAS_REVIEW]-(store)
RETURN node.text AS text,
       store AS store,
       score,
       {
         pk : store.pk,
         reviewText: node.text,
         storeName: store.MCT_NM,
         store_Type: store.MCT_TYPE,
         store_Image: {kakao: store.image_url_kakao, google: store.image_url_google},
         store_Rating: {kakao: store.rating_kakao, google: store.rating_google},
         reviewCount: {kakao: store.rating_count_kakao, google: store.rating_count_google}
       } AS metadata
"""

retrievalQuery_v3 = """
MATCH (node)<-[:HAS_REVIEW]-(store)
RETURN node.text AS text,
       store AS store,
       score,
       {
         pk : store.pk,
         reviewText: node.text,
         storeName: store.MCT_NM,
         store_Type: store.MCT_TYPE,
         store_Addr: store.ADDR,
         store_Image: {naver: store.image_url_naver, kakao: store.image_url_kakao, google: store.image_url_google},
         store_Rating: {naver: store.rating_naver, kakao: store.rating_kakao, google: store.rating_google},
         reviewCount: {naver: store.rating_count_naver, kakao: store.rating_count_kakao, google: store.rating_count_google},
         purpose: store.purpose,
         use_how: store.use_how,
         viwit_with: store.visit_with,
         wait_time: store.wait_time,
         menu : store.menu
       } AS metadata
"""


retrievalQuery_grpEmb = """
MATCH (node)<-[:HAS_REVIEW]-(store)
RETURN node.text AS text,
       store AS store,
       score,
       {
         pk: store.pk,
         reviewText: node.text,
         storeName: store.MCT_NM,
         store_Type: store.MCT_TYPE,
         store_Image: {kakao: store.image_url_kakao, google: store.image_url_google, naver: store.image_url_naver},
         store_Rating: {kakao: store.rating_kakao, google: store.rating_google, naver: store.rating_naver},
         reviewCount: {kakao: store.rating_count_kakao, google: store.rating_count_google, naver: store.rating_count_naver},
         purpose: store.purpose,
         use_how: store.use_how,
         visit_with: store.visit_with,
         wait_time: store.wait_time,
         menu: store.menu,
         graphEmbedding: node.GraphEmbedding
       } AS metadata
"""