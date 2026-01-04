import os
import json
from datetime import datetime, timedelta
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import TradingAgents components
from .tradingagents.graph.trading_graph import TradingAgentsGraph
from .tradingagents.default_config import DEFAULT_CONFIG


@require_http_methods(["GET"])
def trading_agent_index(request):
    """Main trading agent page"""
    context = {
        'user': request.user,
    }
    return render(request, 'trading_agent/index.html', context)


@require_http_methods(["POST"])
@csrf_exempt
def analyze_stock(request):
    """
    Analyze a stock with the TradingAgents system (STREAMING)
    POST /trading_agent/analyze/
    Body: {"ticker": "AAPL", "date": "2024-12-27"}
    """
    try:
        data = json.loads(request.body)
        ticker = data.get('ticker', '').upper()
        date_str = data.get('date', '')

        if not ticker or not date_str:
            return JsonResponse({
                'status': 'error',
                'message': 'ticker and date are required'
            }, status=400)

        def event_stream():
            """Generator function that yields progress updates"""
            try:
                # Initialize
                yield f"data: {json.dumps({'type': 'progress', 'step': '초기화 중...', 'progress': 5})}\n\n"
                yield f"data: {json.dumps({'type': 'log', 'message': f'{ticker} 분석 시작 (날짜: {date_str})', 'level': 'start', 'task_id': 'init'})}\n\n"
                yield f"data: {json.dumps({'type': 'log', 'message': f'{ticker} 분석 시작', 'level': 'complete', 'task_id': 'init'})}\n\n"

                # Get user profile if authenticated
                user_profile = None
                if request.user.is_authenticated:
                    yield f"data: {json.dumps({'type': 'log', 'message': '사용자 프로필 로드', 'level': 'start', 'task_id': 'profile'})}\n\n"
                    user_profile = get_user_investment_profile(request.user)
                    yield f"data: {json.dumps({'type': 'log', 'message': '사용자 프로필 로드 완료', 'level': 'complete', 'task_id': 'profile'})}\n\n"

                # Create TradingAgents with debug mode to get streaming updates
                yield f"data: {json.dumps({'type': 'log', 'message': '멀티 에이전트 시스템 초기화 중...', 'level': 'start', 'task_id': 'graph_init'})}\n\n"
                ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())
                yield f"data: {json.dumps({'type': 'log', 'message': 'TradingAgents 그래프 초기화 완료', 'level': 'complete', 'task_id': 'graph_init'})}\n\n"

                yield f"data: {json.dumps({'type': 'progress', 'step': '시장 데이터 수집 중...', 'progress': 15})}\n\n"
                yield f"data: {json.dumps({'type': 'log', 'message': '데이터 소스에서 시장 데이터 수집 중...', 'level': 'start', 'task_id': 'market_data'})}\n\n"

                # Track progress through the graph execution
                from .tradingagents.agents.utils.agent_states import AgentState

                init_agent_state = {
                    "company_of_interest": ticker,
                    "trade_date": date_str,
                    "messages": [],
                    "market_report": "",
                    "sentiment_report": "",
                    "news_report": "",
                    "fundamentals_report": "",
                    "investment_debate_state": {
                        "bull_history": [],
                        "bear_history": [],
                        "history": [],
                        "current_response": "",
                        "judge_decision": "",
                    },
                    "trader_investment_plan": "",
                    "risk_debate_state": {
                        "risky_history": [],
                        "safe_history": [],
                        "neutral_history": [],
                        "history": [],
                        "judge_decision": "",
                    },
                    "investment_plan": "",
                    "final_trade_decision": "",
                }

                args = {"config": {"recursion_limit": 100}}
                trace = []
                current_progress = 15

                # Track what we've already reported
                reported_market = False
                reported_sentiment = False
                reported_news = False
                reported_fundamentals = False

                for chunk in ta.graph.stream(init_agent_state, **args):
                    # Check if chunk has data updates and report them
                    chunk_data = list(chunk.values())[0] if chunk else {}

                    # Report market data collection completion
                    if not reported_market and chunk_data.get('market_report'):
                        market_report = chunk_data.get('market_report', '')
                        yield f"data: {json.dumps({'type': 'log', 'message': '시장 데이터 수집 완료', 'level': 'complete', 'task_id': 'market_data'})}\n\n"
                        yield f"data: {json.dumps({'type': 'intermediate', 'report_type': 'market', 'content': market_report})}\n\n"
                        reported_market = True
                        current_progress = 30
                        yield f"data: {json.dumps({'type': 'progress', 'step': '시장 데이터 분석 중...', 'progress': current_progress})}\n\n"

                    # Report sentiment data collection
                    if not reported_sentiment and chunk_data.get('sentiment_report'):
                        sentiment_report = chunk_data.get('sentiment_report', '')
                        yield f"data: {json.dumps({'type': 'log', 'message': '소셜 미디어 감성 분석 완료', 'level': 'complete', 'task_id': 'sentiment'})}\n\n"
                        yield f"data: {json.dumps({'type': 'intermediate', 'report_type': 'sentiment', 'content': sentiment_report})}\n\n"
                        reported_sentiment = True
                        current_progress = 45
                        yield f"data: {json.dumps({'type': 'progress', 'step': '감성 데이터 처리 중...', 'progress': current_progress})}\n\n"

                    # Report news data collection
                    if not reported_news and chunk_data.get('news_report'):
                        news_report = chunk_data.get('news_report', '')
                        yield f"data: {json.dumps({'type': 'log', 'message': '뉴스 데이터 수집 완료', 'level': 'complete', 'task_id': 'news'})}\n\n"
                        yield f"data: {json.dumps({'type': 'intermediate', 'report_type': 'news', 'content': news_report})}\n\n"
                        reported_news = True
                        current_progress = 55
                        yield f"data: {json.dumps({'type': 'progress', 'step': '뉴스 분석 중...', 'progress': current_progress})}\n\n"

                    # Report fundamentals data collection
                    if not reported_fundamentals and chunk_data.get('fundamentals_report'):
                        fundamentals_report = chunk_data.get('fundamentals_report', '')
                        yield f"data: {json.dumps({'type': 'log', 'message': '기업 펀더멘털 분석 완료', 'level': 'complete', 'task_id': 'fundamentals'})}\n\n"
                        yield f"data: {json.dumps({'type': 'intermediate', 'report_type': 'fundamentals', 'content': fundamentals_report})}\n\n"
                        reported_fundamentals = True
                        current_progress = 65
                        yield f"data: {json.dumps({'type': 'progress', 'step': '펀더멘털 데이터 처리 중...', 'progress': current_progress})}\n\n"

                    # Update progress based on which node is executing
                    if 'market_analyst' in str(chunk):
                        if not reported_market:
                            current_progress = 20
                            yield f"data: {json.dumps({'type': 'progress', 'step': '기술적 지표 계산 중...', 'progress': current_progress})}\n\n"
                            yield f"data: {json.dumps({'type': 'log', 'message': '기술적 지표 분석 시작...', 'level': 'info'})}\n\n"
                    elif 'news_analyst' in str(chunk):
                        current_progress = 40
                        if not reported_sentiment:
                            yield f"data: {json.dumps({'type': 'log', 'message': '소셜 미디어 감성 분석', 'level': 'start', 'task_id': 'sentiment'})}\n\n"
                        if not reported_news:
                            yield f"data: {json.dumps({'type': 'log', 'message': '뉴스 데이터 수집', 'level': 'start', 'task_id': 'news'})}\n\n"
                    elif 'fundamentals_analyst' in str(chunk):
                        current_progress = 50
                        if not reported_fundamentals:
                            yield f"data: {json.dumps({'type': 'log', 'message': '기업 펀더멘털 분석', 'level': 'start', 'task_id': 'fundamentals'})}\n\n"
                    elif 'bull_researcher' in str(chunk):
                        current_progress = 70
                        yield f"data: {json.dumps({'type': 'progress', 'step': '투자 토론 진행 중...', 'progress': current_progress})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': 'Bull Analyst: 낙관적 분석 진행 중...', 'level': 'start', 'task_id': 'bull_research'})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': 'Bull 리서치 완료', 'level': 'complete', 'task_id': 'bull_research'})}\n\n"
                    elif 'bear_researcher' in str(chunk):
                        current_progress = 73
                        yield f"data: {json.dumps({'type': 'progress', 'step': '투자 토론 진행 중...', 'progress': current_progress})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': 'Bear Analyst: 비관적 분석 진행 중...', 'level': 'start', 'task_id': 'bear_research'})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': 'Bear 리서치 완료', 'level': 'complete', 'task_id': 'bear_research'})}\n\n"
                    elif 'investment_judge' in str(chunk):
                        current_progress = 76
                        yield f"data: {json.dumps({'type': 'log', 'message': '투자 전략 토론 진행', 'level': 'start', 'task_id': 'investment_debate'})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': '투자 전략 토론 완료', 'level': 'complete', 'task_id': 'investment_debate'})}\n\n"
                    elif 'trader' in str(chunk):
                        current_progress = 80
                        yield f"data: {json.dumps({'type': 'progress', 'step': '트레이더 투자 계획 수립 중...', 'progress': current_progress})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': 'Trader: 투자 계획 작성 중...', 'level': 'start', 'task_id': 'trader_decision'})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': '트레이더 투자 계획 수립 완료', 'level': 'complete', 'task_id': 'trader_decision'})}\n\n"
                    elif 'risk_manager' in str(chunk):
                        current_progress = 85
                        yield f"data: {json.dumps({'type': 'progress', 'step': '리스크 평가 중...', 'progress': current_progress})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': 'Risk Manager: 리스크 분석 시작...', 'level': 'start', 'task_id': 'risk_debate'})}\n\n"
                    elif 'safe_debator' in str(chunk):
                        current_progress = 87
                    elif 'neutral_debator' in str(chunk):
                        current_progress = 89
                    elif 'risky_debator' in str(chunk):
                        current_progress = 91
                    elif 'risk_judge' in str(chunk):
                        current_progress = 93
                        yield f"data: {json.dumps({'type': 'log', 'message': '리스크 관리 토론 완료', 'level': 'complete', 'task_id': 'risk_debate'})}\n\n"
                    elif 'final_decision' in str(chunk):
                        current_progress = 97
                        yield f"data: {json.dumps({'type': 'progress', 'step': '최종 의사결정 도출 중...', 'progress': current_progress})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': '최종 투자 결정 도출 중...', 'level': 'start', 'task_id': 'final_plan'})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': '최종 투자 계획 수립 완료', 'level': 'complete', 'task_id': 'final_plan'})}\n\n"

                    trace.append(chunk)

                final_state = trace[-1] if trace else init_agent_state
                yield f"data: {json.dumps({'type': 'log', 'message': 'All agents completed execution', 'level': 'complete'})}\n\n"

                # Process the decision
                yield f"data: {json.dumps({'type': 'log', 'message': 'Processing final trading signal...', 'level': 'info'})}\n\n"
                decision = ta.process_signal(final_state.get("final_trade_decision", ""))
                yield f"data: {json.dumps({'type': 'log', 'message': f'Decision: {decision}', 'level': 'complete'})}\n\n"

                yield f"data: {json.dumps({'type': 'progress', 'step': '분석 완료!', 'progress': 100})}\n\n"

                # Calculate accuracy
                yield f"data: {json.dumps({'type': 'log', 'message': 'Calculating backtest accuracy...', 'level': 'info'})}\n\n"
                accuracy_info = calculate_accuracy(ticker, date_str, decision)
                if accuracy_info.get('calculable'):
                    return_pct = accuracy_info.get('return_pct', 'N/A')
                    yield f"data: {json.dumps({'type': 'log', 'message': f'Backtest complete: {return_pct} return', 'level': 'complete'})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'log', 'message': 'Backtest unavailable for this date', 'level': 'warning'})}\n\n"

                # Send final result
                result = {
                    'type': 'result',
                    'status': 'success',
                    'ticker': ticker,
                    'date': date_str,
                    'decision': decision,
                    'report': final_state.get('trader_investment_plan', 'No report available.'),
                    'accuracy': accuracy_info,
                    'full_state': {
                        'sentiment': final_state.get('sentiment_report', ''),
                        'fundamentals': final_state.get('fundamentals_report', ''),
                        'technical': final_state.get('market_report', ''),
                        'risk': final_state.get('risk_debate_state', {}).get('judge_decision', '')
                    }
                }
                yield f"data: {json.dumps(result)}\n\n"

            except Exception as e:
                import traceback
                traceback.print_exc()  # Print full traceback to server logs
                error_result = {
                    'type': 'error',
                    'status': 'error',
                    'message': str(e)
                }
                yield f"data: {json.dumps(error_result)}\n\n"

        response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'
        return response

    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


def get_user_investment_profile(user):
    """
    Get user's investment profile
    This will be implemented after we extend the User model
    """
    # TODO: Implement this based on extended User model
    return {
        'risk_tolerance': 'moderate',
        'investment_horizon': 'medium-term',
        'preferred_sectors': [],
        'avoided_sectors': []
    }


def calculate_accuracy(ticker, date_str, decision):
    """Calculate accuracy based on actual price movement"""
    try:
        import yfinance as yf

        analysis_date = datetime.strptime(date_str, "%Y-%m-%d")
        end_date = analysis_date + timedelta(days=7)

        stock = yf.Ticker(ticker)
        hist = stock.history(start=analysis_date, end=end_date)

        if len(hist) < 2:
            return {
                "calculable": False,
                "message": "데이터가 충분하지 않습니다 (미래 날짜이거나 거래일 부족)"
            }

        start_price = round(hist.iloc[0]['Close'], 2)
        end_price = round(hist.iloc[-1]['Close'], 2)
        return_pct = f"{round(((end_price - start_price) / start_price) * 100, 2)}%"

        # Determine if the decision was accurate
        price_change = ((end_price - start_price) / start_price) * 100
        is_accurate = False

        if "BUY" in decision and price_change > 0:
            is_accurate = True
        elif "SELL" in decision and price_change < 0:
            is_accurate = True
        elif "HOLD" in decision and abs(price_change) < 2:
            is_accurate = True

        explanation = f"{ticker}의 {analysis_date.strftime('%Y-%m-%d')} 가격은 ${start_price}였고, {len(hist)-1}일 후 ${end_price}로 변화했습니다."

        if is_accurate:
            explanation += f" AI의 {decision} 판단이 정확했습니다."
        else:
            explanation += f" AI의 {decision} 판단과 실제 움직임이 다릅니다."

        return {
            "calculable": True,
            "start_price": start_price,
            "end_price": end_price,
            "return_pct": return_pct,
            "is_accurate": is_accurate,
            "explanation": explanation
        }
    except Exception as e:
        return {
            "calculable": False,
            "message": f"백테스트 오류: {str(e)}"
        }


@require_http_methods(["GET", "POST"])
def user_profile_api(request):
    """
    API for managing user investment profile
    GET: Retrieve current profile
    POST: Update profile based on natural language input
    """
    if not request.user.is_authenticated:
        return JsonResponse({
            'status': 'error',
            'message': 'Authentication required'
        }, status=401)

    if request.method == "GET":
        profile = get_user_investment_profile(request.user)
        return JsonResponse(profile)

    elif request.method == "POST":
        # TODO: Implement profile update with LLM
        data = json.loads(request.body)
        text_input = data.get('text', '')

        # For now, return the current profile
        # Later we'll use profile_manager.update_profile_from_text(text_input)
        profile = get_user_investment_profile(request.user)
        return JsonResponse(profile)
