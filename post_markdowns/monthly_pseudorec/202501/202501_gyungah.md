# **📌 챗봇 시스템 아키텍처 설계 방법**

1. **챗봇의 핵심 목적 정의**
2. **대화 흐름(Chat Flow) 설계 및 구현 방법 (To be continued…)**


🔗 <a href="https://read.engineerscodex.com/p/7-simple-habits-of-the-top-1-of-engineers " target="_blank">**7 simple habits of the best engineers I know ↗**</a>



위에 글을 읽으며, 챗봇을 어떻게 구성해야 할지 고민하게 되었습니다.

챗봇을 개발할 때는 다양한 요청 사항을 고려해야 하며, 반드시 따라야 할 프로세스가 있습니다. 하지만 현재 챗봇 시스템 아키텍처를 설계하는 방법에 대한 체계적인 가이드가 많지 않은 듯합니다.

7 Simple Habits에서 언급된 것처럼, 개발에는 코드 스타일, 표준, 원칙이 존재하며, 이를 준수하며 올바르게 구현하는 방법이 있습니다. 하지만 챗봇 아키텍처 설계 과정에 대한 명확한 프로세스를 찾기 어렵다는 점이 고민이었습니다.

이에 저는 직접 챗봇을 개발하면서, 그리고 다양한 자료를 읽으며 이 흐름을 정리하고자 합니다.

"빠르게 개발하려면 천천히 코딩하라"는 말처럼, 초기에는 표준, 테스트, 원칙을 정립하기 위해 자료 조사와 경험을 바탕으로 프로세스를 체계화했습니다. 이렇게 하면 단기적으로는 시간이 걸릴 수 있지만, 장기적으로는 더 적은 수정으로 빠르고 안정적인 개발이 가능하다고 생각합니다. 

본 글 또한 완벽하지 않기에 더 좋은 제안이나 리뷰를 달아주시면 감사하겠습니다.

# **1. 챗봇의 주요 목적 정의**

챗봇의 목적을 정의하는 것은 챗봇 개발의 첫 단계이자 가장 중요한 과정입니다. 명확한 목적은 챗봇의 기능, 사용자 경험, 기술적 구현 방향을 결정하며, 성공적인 개발과 운영을 위한 기반이 됩니다. 

목적에서 정의해야 하는 것은 챗봇의 목표와 범위를 정의해야합니다. 이를 통해 다음 3가지의 효과를 볼 수 있습니다.

- **사용자 기대 충족**: 챗봇이 특정 문제를 해결하거나 작업을 수행하도록 설계되면 사용자 만족도가 높아집니다.
- **비즈니스 목표와 정렬**: 챗봇의 기능은 비즈니스 목표(예: 고객 지원, 판매 증대, 사용자 참여 강화)와 일치할 수 있습니다.
- **효율적인 개발**: 명확한 목적은 불필요한 기능 추가를 방지하고 프로젝트 범위를 제한하여 효율적인 개발을 돕습니다.

**그렇다면 이제 `챗봇 목적 정의 프로세스`를 확인해 보겠습니다.**

## **(1) 비즈니스 목표 설정**

- **핵심 질문**:
    1. 이 챗봇의 주요 역할은 무엇인가? (예: 고객 지원, 판매 지원, 정보 제공)
    2. 어떤 문제를 해결하려고 하는가? (예: 고객 대기 시간 단축, 반복적인 질문 처리)
    3. 기대하는 결과는 무엇인가? (예: CSAT(고객 만족도) 향상, 비용 절감, 운영 효율성, 개인화 추천)
- **실제 사례**:
    - *Salesforce*: "24시간 7일 고객 지원 제공 및 상담원 부담 감소".
    - *Sephora*: "밀레니얼 세대 뷰티 쇼핑 지원 및 제품 추천"
    - *upGrad:*
        - 리드 생성
        - 중요한 고려 단계에 있는 웹사이트 방문자에게 영업 지원을 제공
        - 지능적인 응답
        - 문의에 답변하고, 답변을 개인화
        - 방문자를 등록으로 유도

## **(2) 사용자 정의 및 챗봇 페르소나 설정**

- **사용자 분석**:
    - 사용자 연령대, 직업, 기술 숙련도, 주요 관심사 등을 파악합니다.
    - 사용자의 사용 빈도, 사용 시간, 환경에 대한 것도 조사합니다.
    - 이는 챗봇 개성 (페르소나)를 정의하는데 도움을 줍니다.
    - 예: IT 지원용 챗봇 → 기술에 익숙한 전문가(공식적) / 전자상거래용 챗봇 → 일반 소비자(캐주얼).
- **구체적 질문**:
    - 누가 이 챗봇을 사용할 것인가?
        - 다국어 지원이 필요한가?
    - 어느 분야와 산업에서 활용하려고 하는가?
    - 챗봇 페르소나 특징은 무엇인가? (공식적, 캐주얼, 재치 있는 등)
        - 봇에 페르소나와 스토리를 부여합니다.
- **실제 사례**:
    - Sephora는 밀레니얼 세대를 대상으로 캐주얼하고 대화형 톤을 사용.

<div class="custom-class">
    <p>
    💡 <strong>페르소나 분석 및 행동 패턴 도출 프로세스</strong>
    </p>
    <p style='padding:10px 20px;'>
    페르소나(Persona) 분석은 사용자의 행동 패턴을 분석하여 제품/서비스의 방향성을 설정하는 과정이다. 이를 통해 이용자의 니즈와 목표를 구체적으로 이해하고 맞춤형 전략을 수립할 수 있다.
    </p>
    <p>
    <strong>1. 행동 변수 정의 및 데이터 수집</strong>
    </p>
    <p style='padding:10px 20px;'>사용자 행동 패턴을 구별하는 주요 요소(행동 변수)를 선정한다. 수집 항목은 동기, 목적, 주요 작업(Task), 사용 빈도 및 기간, 사용 능력(숙련도), 환경 및 심리적 요소(멘탈 모델, 태도), 인구통계학적 특성(연령, 직업, 소득 수준 등)이며, 일반적으로 15~30개 변수를 도출한다.</p>

    <p><strong>2. 행동 변수별 사용자의 위치 분석</strong></p>
    
    <p style='padding:10px 20px;'>행동 변수 간 관계를 파악하고 개별 사용자의 위치를 매핑하여 유사한 특성을 가진 그룹을 식별한다.</p>
    
    <p><strong>3. 주요 행동 패턴 도출 및 그룹화</strong></p>
    
    <p style='padding:10px 20px;'>6-8개 주요 행동 패턴 그룹을 도출한다. 사용자 조사 규모에 따라 차별적인 행동 패턴이 보이는 그룹이 2-4개 정도 발견될 수 있다.</p>
    
    <p><strong>4. 페르소나 정의 및 목표 설정</strong></p>
    
    <p style='padding:10px 20px;'>행동 패턴을 기반으로 사용자 그룹을 페르소나로 정의하고, 각 페르소나의 목표와 니즈를 3~5개 설정한다.</p>
    
    <p><strong>5. 페르소나 상세 설명 작성</strong></p>
    
    <p style='padding:10px 20px;'>각 페르소나가 실존하는 인물처럼 보이도록 이름과 사진을 추가하고, 행동 특성, 선호도, 가치관을 구체적으로 설명한다. 다른 페르소나와 차별되는 점을 명확히 하고 감정과 동기까지 고려한다.</p>
    
    <p><strong>6. 페르소나 유형 및 우선순위 결정</strong></p>
    
    <p style='padding:10px 20px;'>비즈니스 목표에 따라 가장 중요한 페르소나를 선정하고, 각 페르소나의 비중과 전략적 중요도를 고려해 우선순위를 결정한다. 필요 시 보조 페르소나(Secondary Persona)를 설정한다.</p>
</div>

## **(3) 사용 시나리오 작성**

- **목표 기반 시나리오**:
    - 사용자가 챗봇과 상호작용할 구체적인 시나리오를 작성합니다.
    - 예상되는 질문을 직접 생성하고, 이를 맥락별로 분류하여 어떤 유형의 질문이 많은지 분석합니다.
- **실제 사례**:
    - *Tidio*: 음식 배달 챗봇 → "사용자가 메뉴를 선택하면 추천 옵션을 제공".

## **(4) 기능 우선순위 설정**

- 핵심 기능과 부가 기능을 구분합니다.
    - 핵심 기능: 반드시 필요한 기능 (예: FAQ 응답).
    - 부가 기능: 추가적으로 제공할 수 있는 기능 (예: 유머 응답).
- **실제 사례**:
    - 의료 분야 챗봇 → "예약 스케줄링"은 필수, "건강 팁 제공"은 부가 기능.

<div class="custom-class">
    <div style="font-weight: bold; font-size: 18px; margin-bottom: 10px;">💡 질문지를 작성하고 분류 프로세스 (3-4)</div>
    
    <div style="font-weight: bold; font-size: 18px; margin-top: 15px;">(1) 예상되는 질문 나열</div>
    <p style="margin-left: 10px;">먼저, 사용자가 챗봇에게 물어볼 가능성이 높은 질문들을 나열합니다.</p>
    <p style="margin-left: 10px;">- 예시 (전자상거래 챗봇):</p>
    <ul style="margin-left: 20px;">
        <li>내 주문 상태는 어떻게 되나요?</li>
        <li>이 제품의 재고가 있나요?</li>
        <li>환불 절차는 어떻게 되나요?</li>
        <li>배송비는 얼마인가요?</li>
        <li>특정 제품 추천해 줄 수 있나요?</li>
    </ul>
    
    <div style="font-weight: bold; font-size: 18px; margin-top: 15px;">(2) 질문 분류 기준</div>
    <ul style="margin-left: 20px;">
        <li><b>의도(Intent) 기반 분류:</b>
            <ul>
                <li>정보 요청 (FAQ): "배송비는 얼마인가요?"</li>
                <li>문제 해결 (Troubleshooting): "제품이 작동하지 않아요."</li>
                <li>추천 요청 (Recommendation): "어떤 제품이 가장 인기 있나요?"</li>
            </ul>
        </li>
        <li><b>복잡성 기반 분류:</b>
            <ul>
                <li>단순 질문: "환불 정책은 무엇인가요?"</li>
                <li>복잡한 질문: "내 상황에 맞는 최적의 보험 상품은 무엇인가요?"</li>
            </ul>
        </li>
        <li><b>시간성 기반 분류:</b>
            <ul>
                <li><b>긴급한 질문/비긴급한 질문:</b>
                    <ul>
                        <li>긴급한 질문: "내 비행기 예약이 취소되었나요?"</li>
                        <li>비긴급한 질문: "다음 주 날씨는 어떤가요?"</li>
                    </ul>
                </li>
                <li><b>질문 시간대:</b>
                    <ul>
                        <li>사용자가 주로 어떤 시간대에 질문할 가능성이 높은지 파악</li>
                    </ul>
                </li>
                <li><b>최신성:</b>
                    <ul>
                        <li>최신 정보 필요 여부 또는 특정 이벤트 정보 필요 여부 파악</li>
                        <ul>
                            <li>최신 정보: "최근 작업 중 오류가 있는 부분이 있는가?"</li>
                            <li>특정 이벤트 정보: "내가 구입한 물건에 대한 질문이 있다."</li>
                        </ul>
                    </ul>
                </li>
            </ul>
        </li>
        <li><b>사용자 맥락(Context) 기반 분류:</b>
            <ul>
                <li>개인 정보 필요 여부: </li>
                <ul>
                    <li>개인 정보 필요: "내 계정 정보를 알려주세요."</li>
                    <li>개인 정보 불필요: "오늘의 환율은 얼마인가요?"</li>
                </ul>
                <li>사용자 경험 수준:
                    <ul>
                        <li>사용자가 기술적으로 숙련된지, 초보자인지에 따라 대화 스타일과 답변 수준을 조정합니다.</li>
                        <li>예: 초보자 → 간단하고 친절한 답변 / 전문가 → 기술적이고 상세한 답변.</li>
                    </ul>
                </li>
            </ul>
        </li>
    </ul>
    
    <div style="font-weight: bold; font-size: 18px; margin-top: 15px;">(3) 우선순위 결정</div>
    <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
        <thead>
            <tr style="background-color: #f4f4f4;">
                <th style="border: 1px solid #ddd; padding: 8px;">질문 유형</th>
                <th style="border: 1px solid #ddd; padding: 8px;">빈도</th>
                <th style="border: 1px solid #ddd; padding: 8px;">중요도</th>
                <th style="border: 1px solid #ddd; padding: 8px;">우선순위</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">배송 상태 문의</td>
                <td style="border: 1px solid #ddd; padding: 8px;">높음</td>
                <td style="border: 1px solid #ddd; padding: 8px;">높음</td>
                <td style="border: 1px solid #ddd; padding: 8px;">★★★★☆</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">환불 절차 문의</td>
                <td style="border: 1px solid #ddd; padding: 8px;">중간</td>
                <td style="border: 1px solid #ddd; padding: 8px;">높음</td>
                <td style="border: 1px solid #ddd; padding: 8px;">★★★☆☆</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">제품 추천 요청</td>
                <td style="border: 1px solid #ddd; padding: 8px;">낮음</td>
                <td style="border: 1px solid #ddd; padding: 8px;">중간</td>
                <td style="border: 1px solid #ddd; padding: 8px;">★★☆☆☆</td>
            </tr>
        </tbody>
    </table>

    <div style="font-weight: bold; font-size: 16px; margin-top: 15px;">그리고 필요한 기능들에 있어서 우선순위를 정하세요. </div>
    <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
        <thead>
            <tr style="background-color: #f4f4f4;">
                <th style="border: 1px solid #ddd; padding: 8px;">기능 유형</th>
                <th style="border: 1px solid #ddd; padding: 8px;">빈도</th>
                <th style="border: 1px solid #ddd; padding: 8px;">중요도</th>
                <th style="border: 1px solid #ddd; padding: 8px;">우선순위</th>
                <th style="border: 1px solid #ddd; padding: 8px;">설명 (예시)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">최신성</td>
                <td style="border: 1px solid #ddd; padding: 8px;">높음</td>
                <td style="border: 1px solid #ddd; padding: 8px;">높음</td>
                <td style="border: 1px solid #ddd; padding: 8px;">★★★★☆</td>
                <td style="border: 1px solid #ddd; padding: 8px;">대부분의 대화가 최신 일정에 대한 질문</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">전문가 수준</td>
                <td style="border: 1px solid #ddd; padding: 8px;">중간</td>
                <td style="border: 1px solid #ddd; padding: 8px;">높음</td>
                <td style="border: 1px solid #ddd; padding: 8px;">★★★☆☆</td>
                <td style="border: 1px solid #ddd; padding: 8px;">전문가를 위한 답변</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">복잡한 질문</td>
                <td style="border: 1px solid #ddd; padding: 8px;">낮음</td>
                <td style="border: 1px solid #ddd; padding: 8px;">중간</td>
                <td style="border: 1px solid #ddd; padding: 8px;">★★☆☆☆</td>
                <td style="border: 1px solid #ddd; padding: 8px;">특정 기계를 어떻게 조치해야하는지에 대한 질문</td>
            </tr>
        </tbody>
    </table>


</div>

## **(5) 기대 성과 측정**

- 성공 여부를 평가할 KPI(Key Performance Indicator)를 설정합니다.
    - 예:
        - 응답 시간 단축(평균 응답 시간), 고객 만족도(CSAT), 전환율(Conversion Rate), 비용 절감, 운영 효율성, 개인화 추천

이렇게 해서 챗봇 주요 목표 정의를 하는 방법에 대해 알아보았습니다. 다음에는 “**Chat Flow 설계 및 방법”**에 대해 나눠보도록 하겠습니다!

 

# 참고

🔗 <a href="https://www.tidio.com/blog/chatbot-flowchart/ " target="_blank">**Chatbot Flowchart Examples & Decision Tree Diagram ↗**</a>

🔗 <a href="https://www.salesforce.com/service/customer-service-chatbot/chatbot-best-practices/ 
" target="_blank">**The Top Chatbot Best Practices for Service | Salesforce US ↗**</a>

🔗 <a href="https://www.engati.com/blog/design-chatbot-flow-chart" target="_blank">**How to Design Chatbot Conversational Flow with Examples (2024) | Engati ↗**</a>

🔗 <a href="https://brunch.co.kr/@kja0717/17" target="_blank">**사용자 이해: 페르소나, 사용자 여정 맵, 고객 세분화 ↗**</a>