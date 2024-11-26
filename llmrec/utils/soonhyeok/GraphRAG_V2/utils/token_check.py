from ..prompt.selecting_recomm import SELECTING_FOR_RECOMM

def token_check(candidate_str, state, llm):
    candidates_lst = candidate_str.strip().split('\n\n')
    num_candidates = len(candidates_lst)
    
    while True:
        prompt = SELECTING_FOR_RECOMM.format(
            query=state['query'], 
            intent=state['intent'],
            candidates='\n\n'.join(candidates_lst[:num_candidates])
        )
        
        num_tokens = llm.get_num_tokens(prompt)
        
        # 현재 후보 수와 토큰 수를 출력합니다.
        print(f"Current number of candidates: {num_candidates}, Token count: {num_tokens}")
        
        if num_tokens <= 5000:
            break
        
        num_candidates -= 1

    return '\n\n'.join(candidates_lst[:num_candidates])