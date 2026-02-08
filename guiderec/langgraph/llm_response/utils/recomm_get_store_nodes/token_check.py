from prompt.final_selecting_for_recomm import FINAL_SELECTING_FOR_RECOMM_v2

def token_check(candidate_str, state, llm, placeholder):
    candidates_lst = candidate_str.strip().split('\n\n')
    num_candidates = len(candidates_lst)
    
    while True:
        prompt = FINAL_SELECTING_FOR_RECOMM_v2.format(
            query=state['query'], 
            intent=state['rewritten_query'],
            candidates='\n\n'.join(candidates_lst[:num_candidates])
        )
        
        num_tokens = llm.get_num_tokens(prompt)
        
        # 현재 후보 수와 토큰 수를 출력합니다.
        print(f"Current number of candidates: {num_candidates}, Token count: {num_tokens}")
        placeholder.markdown(
                    f"> {num_candidates}개의 후보 탐색 중...",
                    unsafe_allow_html=True,
                )
        
        if num_tokens <= 5000:
            break
        
        num_candidates -= 1

    return '\n\n'.join(candidates_lst[:num_candidates])