def get_ratings_str_for_node(node):
    """
    node 구하는 법
    result = driver.execute_query(pk_store_cypher.format(pk=pk))
    node = result.records[0]['s']
    """
    ratings_html = '<div style="display: flex; align-items: center; margin-bottom: 10px;">'
    platforms = ['naver', 'kakao', 'google']
    platform_name_tags = ['<span style="font-weight: bold; color: #1EC800;">Naver</span> :', 
                          '<span style="font-weight: bold; color: #FEE500;">Kakao</span> :', 
                          '<span style="font-weight: bold; color: #4285F4;">G</span><span style="font-weight: bold; color: #EA4335;">o</span><span style="font-weight: bold; color: #FBBC05;">o</span><span style="font-weight: bold; color: #4285F4;">g</span><span style="font-weight: bold; color: #34A853;">l</span><span style="font-weight: bold; color: #EA4335;">e</span></span>: '
                          ]
    for platform, platform_name_tag in zip(platforms, platform_name_tags):
        pf_rating = node[f'rating_{platform}']
        if pf_rating:
            pass
        else:
            continue
        pf_rc = node[f'rating_count_{platform}']
        if pf_rc:
            pass
        else:
            continue
        
        ratings_html += f"""    <div style="margin-right: 20px;">
        {platform_name_tag}
        <span style="font-weight: bold; font-size: 1.1em;">⭐{pf_rating}</span> ({pf_rc}명)
    </div>"""
        
    ratings_html += '</div>'

    return ratings_html