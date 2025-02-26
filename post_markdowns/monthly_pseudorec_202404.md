




# 이남준 ‘’

# Multi Agent Collaboration for RecSys

안녕하세요, AI 엔지니어 이남준입니다.

가짜연구소 추천시스템 팀에서는 열심히 ‘출석’을 담당하고 있습니다. 많이 기여하지 못한 채 팀원 분들의 열정으로 학습하신 내용에 같이 고민만 해드리곤 했는데, 이제 정말 지식 공유를 하고자 이렇게 월간 슈도렉을 작성해봅니다.

이번 월간 슈도렉에서 전달하고자 하는 내용은 Multi Agent를 활용한 추천 Ideation정도로 봐주시면 좋겠습니다.

LLM의 기본 이론적 원리에 대한 내용보다는 prompt와 함께 아키텍쳐를 어떻게 활용했는지를 담아보았고, 이걸 바탕으로 어떻게 추천에 활용할 수 있을지에 대한 인사이트를 공유하는 포스팅이라고 생각해주시면 좋겠습니다!

## Multi Agent Collaboration

멀티 에이전트란 무엇인가?? 

하나의 일을 하는 ‘노드(일꾼)’라고 생각해주시면 됩니다. 우리는 이러한 노드들에 각각의 역할을 가지고 있다 생각하면 됩니다. 

일반적으로 LLM을 사용해 어떤 일을 처리하라고 하는 것이 Single Agent라고 생각하면 됩니다.

이러한 Single Agent만으로도 GPT3.5가 나온 이래로 정말 혁신적인 변화들이 생겨왔습니다만, 금방 적응되어 퀄리티의 한계가 보이기 시작했고, 더 나은 퍼포먼스를 가져오는 것을 바라게 됩니다.

### Content Making Task

회사에서 개인화된 콘텐츠를 생성해서 제공하는 기능을 만들었던 것을 예시로 들겠습니다. 

사용자의 입력에 따라 연구 활동의 가이드라인을 콘텐츠 형태로 만들어주는 서비스를 가정하겠습니다.

‘반도체 공학’ 분야의 ‘환경 오염’과 관련된 연구 관련 가이드라인 콘텐츠를 만들 때, 아래와 같은 기능 요소(Task)를 필요로 합니다.

- 반도체 공학과 환경 오염의 관계 및 주요 연구 키워드 (정보 수집1)
- 반도체 공학과 환경 오염 관련 연구 주제 (생성1)
- 연구 주제 관련 가이드라인 초안 (생성2)
- 관련 연구 (정보 수집2)
- 콘텐츠 형식 맞추기 (취합 정리)

뭐 프롬프트 스킬 상 더 디테일하게 연구 목적이 무엇인지 등도 주면 좋겠지만, 대략적으로는 위와 같습니다.

### Content Making w/ Single Agent

위와 같은 다양한 요소를 필요로 하는 반면, 우리의 3.5털보씨(gpt-3.5-turbo)는 다소 이렇게 다양한 Task를 한 prompt에 주면 모든일을 처리하지 못합니다.

어떤 정보를 가져오면, 어떤 생성이 안되고. 이를 계속해서 반복하며 마치 한계 Task를 가진 것처럼 몇 가지 Task 내에서 Trade off되는 것 같은 현상을 볼 수 있습니다. (예시 넣을게요)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/ca8f784c-ae7c-46dd-9f31-026559e1bf54/Untitled.png)

그러다 보니 누락되는 포인트가 생기고, 그에 따라 콘텐츠 품질이 떨어지는 현상이 발생합니다.

물론, 항상 n개씩 누락되는 것은 아니었지만, 직접 평가 상 1개 이상 누락 확률이 100회 기준 약 50%로 서비스로써 사용하기엔 부적합한 수준이었습니다.

### Content Making w/ Multi Agent

그러다 문득 발견하게 된 Multi Agent관련 내용들을 접하면서 머리를 한대 맞은 듯 ‘아! 왜 일을 한번에 다 시키려고 했을까?’하는 생각이 들었습니다.

멀티 에이전트에 대해선 여러 연구와 특정 기업의 사례를 볼 수 있었고, 바~로 아이디어를 착안해 작업해봤습니다.

나눠놓은 각 Task들을 오롯이 하나의 노드(agent)들이 담당해서 수행하는 것입니다.

다양한 자료들에서 일반적으로 Multi Agent를 다룰 때 Supervisor 또는 Manager Agent가 특정 상황에 맞춰 각각의 필요한 Agent들에게 업무를 할당하는 구조입니다만, 저는 모든 Task가 포함되는 것을 목표로 했기 때문에 각 역할에 따른 Agent를 두고, 최종 Agent가 취합정리하는 것을 타깃으로 진행했습니다.

해당 아키텍처를 도식으로 표현하면 아래와 같습니다.

![다소 근본없는 귀여운 멀티 에이젼 아키](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/f6fbad7b-629c-4c27-a58e-e364279ee5ca/Untitled.png)

다소 근본없는 귀여운 멀티 에이젼 아키

위와 같이 진행하게 되면 순서대로,

1. [정보 수집1 agent]사용자 input에 따라 정보를 수집하고 기타 작업을 합니다(CoT with 내부 DB)

2. [생성1 agent] 1번에서 받은 정보를 기반으로 연구하기 좋은 주제를 생성합니다. 여기서 몇가지 조건들이 있는데 목표 output이 명확하기에 수행을 잘합니다.

3-1. [생성2 agent] 2번에서 생성된 주제를 바탕으로 가이드라인의 초안을 작성합니다.

3-2. [정보 수집2 agent] 2번에서 생성된 주제에 대한 가이드라인의 풍부한 내용을 위해 관련 선행 연구자료나 인용할 수 있는 논문을 외부로부터 찾아옵니다. (요약을 할 수도 있습니다)

4. [취합정리 agent] 3번에서 만들어진 가이드라인 초안과 수집된 연구 자료를 짜깁기하여 JSON format으로 변환합니다. 이 작업은 서비스에서 사용자에게 최종적으로 제공되기 위함입니다.

이렇게 진행하다보니 각각의 업무에 하나의 에이전트들이 할당되어 처리를 하니 내용 누락 확률이 현저히 떨어지는 것을 볼 수 있었습니다. (100회 기준 3%)

물론 이 외에 내부적으로 데이터 퀄리티와 프롬프트 퀄리티, 아키텍처나 기능 플로우 자체로도 손봐야 할 것들이 산더미였지만, multi agent의 활용을 통해 single agent의 한계를 조금이나마 극복하지 않았나 싶습니다.

관련 예시를 좀 넣고 싶었는데, 직접 다른 예시를 만들어야 하는 상황이라 추후에 넣겠습니다.

## Multi Agent in RecSys

그렇다면 추천시스템을 연구하는 월간슈도렉에서 Multi Agent 얘기를 왜 하는가?

사실 이 마지막 파트를 위해 서론이 좀 길어졌습니다만, 위 작업을 하면서 이를 충분히 슈도렉의 영화 추천에서 활용할 수 있겠다고 생각하게 되었고, 이에 대한 가설들을 쭉쭉 뽑아내고 있는 상황입니다.

여러 리서치를 통해 대화형 추천을 목표로 하고 있지만, 천천히 다양한 실험을 통해 단계적으로 결과를 만들어 볼 계획입니다. 

### Structured RecSys Flow

그래서 우선 대화형을 배제하고, 대략적으로 생각한 흐름을 구조화 해보았습니다 (당연히 실험들과 함께 추후엔 바뀌겠지만!)

1. 유저로부터 쿼리를 받는다 
2. 시놉시스로부터 쿼리를 해결하기 위한 필요한 내용을 추출한다.
3. 유저 인터랙션 데이터를 참고한다.
4. 답변을 내놓는다.

정말 수많은 방법들이 생각났지만, 간략히 위와 같이 정리해볼 수 있습니다. 위와 같이 구조화 된 흐름을 곧 agent들로 쪼개어 볼 방법들을 고안해보면 다음과 같습니다.

1. 유저 쿼리를 통해 필요한 정보를 파악하는 info agent
2. info agent로부터 정보를 받아 Task를 분배하는 Leader agent
3. 시놉시스 agents (당연히 해당 정보를 따로 뽑아낼 수 있으면 뽑아서 RAG 하는게 나을 것)
    1. 연기자, 감독 정보를 뽑는 agent
    2. 내용을 요약 수집하는 agent
    3. etc.
4. 유저 상호작용을 보고 좋아하는 장르나 패턴을 파악해 알려주는 Interaction agent 

간략히 생각해보았을 때 위와 같은 구조로 agent를 짜볼 수도 있을 것 같다 생각했습니다. 물론 허황된 꿈일 수 있겠지만, 실험을 해보고자 하는 부분입니다. (당장 Embedding에서 막혔지만, 도와주세요 대표님, 도와주세요 openai)

여튼, 위와 같이 하나의 cycle을 만들어내고(매우 느리겠지만 여튼) 다음 cycle을 만들어 대화형으로 추천을 진행해 나가는것도 하나의 방법이 될 수도 있고, 다른 방법도 사용될 수 있을 것 같습니다.

## 마치며

항상 추천에 대한 관심을 갖고 있다보니, Multi Agent 아키텍쳐를 활용하면서도 추천에 적용해보는 것을 생각해왔고, 그 내용을 월간 슈도렉에 기입하게 되었네요.

지극히 개인적인 ideation을 공유해봤는데, 이 아이디어를 빠르게 실험해보고 개선하면서 다음 월간 슈도렉에서 결과를 보여드릴 수 있으면 좋겠습니다.

아 또한, 6월 슈도콘에서도 반드시 선보여야죠 암암.

다음에 실험 결과와 함께 인사이트를 공유해보도록 하겠습니다! 긴 글을 봐주셔서 감사합니다 :)