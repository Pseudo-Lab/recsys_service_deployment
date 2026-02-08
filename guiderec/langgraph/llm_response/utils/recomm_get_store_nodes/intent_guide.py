class IntentGuide:
    def __init__(self):
        self.guide = """
        <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);">
        <h5 style="font-size: 16px; margin-bottom: 10px;">🔍 질문 의도를 다음과 같이 파악했어요 🤖</h5>
        <ul style="list-style-type: none; padding-left: 0;">
        """

    def add(self, string):
        self.guide += string

    def close(self):
        self.guide += f"""</ul>
        </div>"""

    def close_with_num_candidates(self, num_canidates):
        self.guide += f"""</ul>
        <h5 style="font-size: 16px;">⏳ 질문 조건을 만족하는 {num_canidates}개의 후보 중에서 최적의 추천 결과 선별 중...</h5>
        </div>"""