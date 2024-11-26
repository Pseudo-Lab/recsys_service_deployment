from ..langgraph_state import Soonhyeok_GraphState
from ...prompt.selecting_recomm import SELECTING_FOR_RECOMM
from pprint import pprint


def selecting_for_recomm(llm, state: Soonhyeok_GraphState):
   print("Selecting for recomm".ljust(100, '='))
   if state['query_type']=='general':
      pass
   else :
      prompt = SELECTING_FOR_RECOMM.format(
      query=state['query'], 
      intent=state['intent'],
      candidates=state['candidate_str']
      )
      
      print(f"Candidates str : \n{state['candidate_str']}\n")  
      
      response = llm.invoke(prompt)
      if response.content == '' : 
         state['tavily_search_num'] = 6
      else :
         print("response : ", response)
         print(f"Response Metadata: {response.usage_metadata}")
         print(f"input tokens : {response.usage_metadata['input_tokens']:,}")
         state["selected_recommendations"] = eval(
            response.content.replace("```", "").replace("json", "").replace("null", "None").strip()
         )
         pprint(f"Seleted : \n{state['selected_recommendations']}")
         # state["final_answer"] = state["selected_recommendations"]
      
   return state
