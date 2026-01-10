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
                from langchain_core.callbacks import BaseCallbackHandler

                # Tool execution logger with queue for streaming
                tool_events = []

                class ToolLoggingCallback(BaseCallbackHandler):
                    def on_tool_start(self, serialized, input_str, **kwargs):
                        tool_name = serialized.get("name", "Unknown")

                        # Parse input to get meaningful parameters
                        params = ""
                        if input_str:
                            try:
                                # input_str might be a string representation of dict, parse it
                                if isinstance(input_str, str):
                                    import ast
                                    try:
                                        input_dict = ast.literal_eval(input_str)
                                    except:
                                        input_dict = {}
                                elif isinstance(input_str, dict):
                                    input_dict = input_str
                                else:
                                    input_dict = {}

                                # Extract key parameters based on tool type
                                if tool_name == 'get_indicators' and input_dict:
                                    indicator = input_dict.get('indicator', '')
                                    symbol = input_dict.get('symbol', '')
                                    params = f"{symbol}/{indicator}"
                                elif tool_name == 'get_stock_data' and input_dict:
                                    symbol = input_dict.get('symbol', input_dict.get('ticker', ''))
                                    params = symbol
                                elif tool_name == 'get_news' and input_dict:
                                    ticker = input_dict.get('ticker', '')
                                    start = input_dict.get('start_date', '')
                                    end = input_dict.get('end_date', '')
                                    params = f"{ticker} ({start}~{end})"
                                elif tool_name == 'get_global_news' and input_dict:
                                    curr_date = input_dict.get('curr_date', '')
                                    days = input_dict.get('look_back_days', 7)
                                    params = f"{curr_date}, {days}일"
                                elif tool_name in ['get_fundamentals', 'get_balance_sheet', 'get_cashflow', 'get_income_statement'] and input_dict:
                                    ticker = input_dict.get('ticker', '')
                                    freq = input_dict.get('freq', '')
                                    params = f"{ticker}" + (f" ({freq})" if freq else "")
                                elif tool_name in ['get_net_liquidity_tool', 'get_macro_indicators_tool']:
                                    params = ""
                                else:
                                    params = ""
                            except:
                                params = ""

                        # Map tool names to data sources
                        source_map = {
                            'get_stock_data': ('Yahoo Finance', 'https://finance.yahoo.com'),
                            'get_indicators': ('Alpha Vantage', 'https://www.alphavantage.co'),
                            'get_news': ('Google News', 'https://news.google.com'),
                            'get_global_news': ('Reuters/Bloomberg', 'https://reuters.com'),
                            'get_fundamentals': ('Yahoo Finance', 'https://finance.yahoo.com'),
                            'get_balance_sheet': ('SEC EDGAR', 'https://www.sec.gov/edgar'),
                            'get_cashflow': ('SEC EDGAR', 'https://www.sec.gov/edgar'),
                            'get_income_statement': ('SEC EDGAR', 'https://www.sec.gov/edgar'),
                            'get_net_liquidity_tool': ('FRED', 'https://fred.stlouisfed.org'),
                            'get_macro_indicators_tool': ('FRED', 'https://fred.stlouisfed.org'),
                        }

                        source_info = source_map.get(tool_name, (tool_name, None))
                        tool_events.append({
                            'type': 'tool_call',
                            'tool_name': tool_name,
                            'source': source_info[0],
                            'url': source_info[1],
                            'params': params
                        })

                    def on_tool_end(self, output, **kwargs):
                        pass

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
                        "count": 0,
                    },
                    "trader_investment_plan": "",
                    "risk_debate_state": {
                        "risky_history": [],
                        "safe_history": [],
                        "neutral_history": [],
                        "history": [],
                        "current_risky_response": "",
                        "current_safe_response": "",
                        "current_neutral_response": "",
                        "latest_speaker": "",
                        "judge_decision": "",
                        "count": 0,
                    },
                    "investment_plan": "",
                    "final_trade_decision": "",
                }

                tool_callback = ToolLoggingCallback()
                args = {
                    "config": {"recursion_limit": 100, "callbacks": [tool_callback]}
                }
                trace = []
                current_progress = 15

                # Accumulated state to collect all reports
                accumulated_state = {
                    'market_report': '',
                    'sentiment_report': '',
                    'news_report': '',
                    'fundamentals_report': '',
                    'trader_investment_plan': '',
                    'final_trade_decision': '',
                    'risk_debate_state': {'judge_decision': ''},
                    'investment_debate_state': {'judge_decision': ''},
                }

                # Track what we've already reported
                reported_market = False
                reported_sentiment = False
                reported_news = False
                reported_fundamentals = False

                for chunk in ta.graph.stream(init_agent_state, **args):
                    # Stream any pending tool events
                    while tool_events:
                        event = tool_events.pop(0)
                        yield f"data: {json.dumps(event)}\n\n"

                    # Check if chunk has data updates and report them
                    chunk_data = list(chunk.values())[0] if chunk else {}
                    chunk_key = list(chunk.keys())[0] if chunk else ""

                    # Capture LLM reasoning from messages (when LLM outputs thought process before tool calls)
                    if 'messages' in chunk_data:
                        messages = chunk_data['messages']
                        for msg in messages:
                            # Debug: log message structure
                            has_content = hasattr(msg, 'content') and msg.content
                            has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
                            print(f"DEBUG MSG: chunk_key={chunk_key}, has_content={has_content}, has_tool_calls={has_tool_calls}")
                            if has_content:
                                print(f"DEBUG MSG CONTENT: {str(msg.content)[:200]}...")

                            # Check if this message has content (reasoning) and also has tool calls
                            if has_content and has_tool_calls:
                                reasoning_content = msg.content
                                # Determine which analyst this is based on chunk key
                                analyst_type = 'analysis'
                                if 'market' in chunk_key.lower():
                                    analyst_type = 'market'
                                elif 'social' in chunk_key.lower() or 'sentiment' in chunk_key.lower():
                                    analyst_type = 'sentiment'
                                elif 'news' in chunk_key.lower():
                                    analyst_type = 'news'
                                elif 'fundamental' in chunk_key.lower():
                                    analyst_type = 'fundamentals'
                                print(f"DEBUG: Sending reasoning for {analyst_type}")
                                yield f"data: {json.dumps({'type': 'reasoning', 'analyst': analyst_type, 'content': reasoning_content})}\n\n"

                    # Update accumulated state with any new data
                    for key in ['market_report', 'sentiment_report', 'news_report', 'fundamentals_report',
                                'trader_investment_plan', 'final_trade_decision']:
                        if chunk_data.get(key):
                            accumulated_state[key] = chunk_data[key]
                    if chunk_data.get('risk_debate_state', {}).get('judge_decision'):
                        accumulated_state['risk_debate_state']['judge_decision'] = chunk_data['risk_debate_state']['judge_decision']
                    if chunk_data.get('investment_debate_state', {}).get('judge_decision'):
                        accumulated_state['investment_debate_state']['judge_decision'] = chunk_data['investment_debate_state']['judge_decision']

                    # Report market data collection completion
                    if not reported_market and chunk_data.get('market_report'):
                        market_report = chunk_data.get('market_report', '')
                        yield f"data: {json.dumps({'type': 'log', 'message': '시장 데이터 수집 완료', 'level': 'complete', 'task_id': 'market_data'})}\n\n"
                        yield f"data: {json.dumps({'type': 'intermediate', 'report_type': 'market', 'content': market_report})}\n\n"
                        reported_market = True
                        current_progress = 30
                        yield f"data: {json.dumps({'type': 'progress', 'step': '소셜 미디어 감성 분석 중...', 'progress': current_progress})}\n\n"
                        # Start next task
                        yield f"data: {json.dumps({'type': 'log', 'message': '소셜 미디어 감성 분석', 'level': 'start', 'task_id': 'sentiment'})}\n\n"

                    # Report sentiment data collection
                    if not reported_sentiment and chunk_data.get('sentiment_report'):
                        sentiment_report = chunk_data.get('sentiment_report', '')
                        yield f"data: {json.dumps({'type': 'log', 'message': '소셜 미디어 감성 분석 완료', 'level': 'complete', 'task_id': 'sentiment'})}\n\n"
                        yield f"data: {json.dumps({'type': 'intermediate', 'report_type': 'sentiment', 'content': sentiment_report})}\n\n"
                        reported_sentiment = True
                        current_progress = 45
                        yield f"data: {json.dumps({'type': 'progress', 'step': '뉴스 데이터 분석 중...', 'progress': current_progress})}\n\n"
                        # Start next task
                        yield f"data: {json.dumps({'type': 'log', 'message': '최신 뉴스 분석', 'level': 'start', 'task_id': 'news'})}\n\n"

                    # Report news data collection
                    if not reported_news and chunk_data.get('news_report'):
                        news_report = chunk_data.get('news_report', '')
                        yield f"data: {json.dumps({'type': 'log', 'message': '최신 뉴스 분석 완료', 'level': 'complete', 'task_id': 'news'})}\n\n"
                        yield f"data: {json.dumps({'type': 'intermediate', 'report_type': 'news', 'content': news_report})}\n\n"
                        reported_news = True
                        current_progress = 55
                        yield f"data: {json.dumps({'type': 'progress', 'step': '기업 재무 데이터 분석 중...', 'progress': current_progress})}\n\n"
                        # Start next task
                        yield f"data: {json.dumps({'type': 'log', 'message': '기업 재무 데이터 분석', 'level': 'start', 'task_id': 'fundamentals'})}\n\n"

                    # Report fundamentals data collection
                    if not reported_fundamentals and chunk_data.get('fundamentals_report'):
                        fundamentals_report = chunk_data.get('fundamentals_report', '')
                        yield f"data: {json.dumps({'type': 'log', 'message': '기업 재무 데이터 분석 완료', 'level': 'complete', 'task_id': 'fundamentals'})}\n\n"
                        yield f"data: {json.dumps({'type': 'intermediate', 'report_type': 'fundamentals', 'content': fundamentals_report})}\n\n"
                        reported_fundamentals = True
                        current_progress = 65
                        yield f"data: {json.dumps({'type': 'progress', 'step': '강세/약세 분석가 토론 중...', 'progress': current_progress})}\n\n"
                        # Start bull research
                        yield f"data: {json.dumps({'type': 'log', 'message': '강세 분석가 리서치', 'level': 'start', 'task_id': 'bull_research'})}\n\n"

                    # Update progress based on which node is executing
                    chunk_str = str(chunk).lower()
                    node_keys = [k.lower() for k in chunk.keys()] if chunk else []

                    # Check for Bull Researcher
                    if 'bull' in chunk_str or any('bull' in k for k in node_keys):
                        # Show bull argument as intermediate result
                        bull_history = chunk_data.get('investment_debate_state', {}).get('bull_history', [])
                        if bull_history:
                            latest_bull = bull_history[-1] if isinstance(bull_history, list) else str(bull_history)
                            yield f"data: {json.dumps({'type': 'intermediate', 'report_type': 'bull_debate', 'content': latest_bull})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': '강세 분석가 리서치 완료', 'level': 'complete', 'task_id': 'bull_research'})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': '약세 분석가 리서치', 'level': 'start', 'task_id': 'bear_research'})}\n\n"
                        current_progress = 70
                        yield f"data: {json.dumps({'type': 'progress', 'step': '약세 분석가 토론 중...', 'progress': current_progress})}\n\n"

                    # Check for Bear Researcher
                    if 'bear' in chunk_str or any('bear' in k for k in node_keys):
                        # Show bear argument as intermediate result
                        bear_history = chunk_data.get('investment_debate_state', {}).get('bear_history', [])
                        if bear_history:
                            latest_bear = bear_history[-1] if isinstance(bear_history, list) else str(bear_history)
                            yield f"data: {json.dumps({'type': 'intermediate', 'report_type': 'bear_debate', 'content': latest_bear})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': '약세 분석가 리서치 완료', 'level': 'complete', 'task_id': 'bear_research'})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': '투자 전략 토론', 'level': 'start', 'task_id': 'investment_debate'})}\n\n"
                        current_progress = 75
                        yield f"data: {json.dumps({'type': 'progress', 'step': '투자 전략 토론 중...', 'progress': current_progress})}\n\n"

                    # Check for Research Manager
                    if 'research manager' in chunk_str or any('manager' in k for k in node_keys):
                        # Show investment judge decision
                        judge_decision = chunk_data.get('investment_debate_state', {}).get('judge_decision', '')
                        if judge_decision:
                            yield f"data: {json.dumps({'type': 'intermediate', 'report_type': 'investment_decision', 'content': judge_decision})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': '투자 전략 토론 완료', 'level': 'complete', 'task_id': 'investment_debate'})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': '트레이더 의사결정', 'level': 'start', 'task_id': 'trader_decision'})}\n\n"
                        current_progress = 80
                        yield f"data: {json.dumps({'type': 'progress', 'step': '트레이더 투자 계획 수립 중...', 'progress': current_progress})}\n\n"

                    # Check for Trader
                    if 'trader' in chunk_str or any('trader' in k for k in node_keys):
                        # Show trader investment plan
                        trader_plan = chunk_data.get('trader_investment_plan', '')
                        if trader_plan:
                            yield f"data: {json.dumps({'type': 'intermediate', 'report_type': 'trader_plan', 'content': trader_plan})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': '트레이더 의사결정 완료', 'level': 'complete', 'task_id': 'trader_decision'})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': '리스크 관리 토론', 'level': 'start', 'task_id': 'risk_debate'})}\n\n"
                        current_progress = 85
                        yield f"data: {json.dumps({'type': 'progress', 'step': '리스크 평가 중...', 'progress': current_progress})}\n\n"

                    # Check for Risk Judge
                    if 'risk judge' in chunk_str or 'risk_judge' in chunk_str or any('judge' in k for k in node_keys):
                        # Show risk judge decision
                        risk_decision = chunk_data.get('risk_debate_state', {}).get('judge_decision', '')
                        if risk_decision:
                            yield f"data: {json.dumps({'type': 'intermediate', 'report_type': 'risk_decision', 'content': risk_decision})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': '리스크 관리 토론 완료', 'level': 'complete', 'task_id': 'risk_debate'})}\n\n"
                        yield f"data: {json.dumps({'type': 'log', 'message': '최종 투자 계획 수립', 'level': 'start', 'task_id': 'final_plan'})}\n\n"
                        current_progress = 95
                        yield f"data: {json.dumps({'type': 'progress', 'step': '최종 의사결정 도출 중...', 'progress': current_progress})}\n\n"

                    trace.append(chunk)

                yield f"data: {json.dumps({'type': 'log', 'message': '최종 투자 계획 수립 완료', 'level': 'complete', 'task_id': 'final_plan'})}\n\n"

                # Process the decision using accumulated state (no log messages)
                decision = ta.process_signal(accumulated_state.get("final_trade_decision", ""))

                yield f"data: {json.dumps({'type': 'progress', 'step': '분석 완료!', 'progress': 100})}\n\n"

                # Calculate accuracy (no log messages)
                accuracy_info = calculate_accuracy(ticker, date_str, decision)

                # Send final result using accumulated state
                result = {
                    'type': 'result',
                    'status': 'success',
                    'ticker': ticker,
                    'date': date_str,
                    'decision': decision,
                    'report': accumulated_state.get('trader_investment_plan', 'No report available.'),
                    'accuracy': accuracy_info,
                    'full_state': {
                        'sentiment': accumulated_state.get('sentiment_report', ''),
                        'fundamentals': accumulated_state.get('fundamentals_report', ''),
                        'technical': accumulated_state.get('market_report', ''),
                        'risk': accumulated_state.get('risk_debate_state', {}).get('judge_decision', '')
                    }
                }
                yield f"data: {json.dumps(result)}\n\n"

            except GeneratorExit:
                # Client disconnected (stop button clicked)
                print(f"[STOP SIGNAL] Client disconnected from analysis stream for {ticker} on {date_str}")
                # Clean exit - generator will terminate here
                return
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
    Get user's investment profile from database
    """
    from .models import InvestmentProfile

    try:
        profile = InvestmentProfile.objects.get(user=user)
        return profile.to_dict()
    except InvestmentProfile.DoesNotExist:
        # Return default profile if not exists
        return {
            'risk_tolerance': 'moderate',
            'investment_horizon': 'medium-term',
            'preferred_sectors': [],
            'avoided_sectors': [],
            'raw_text': None,
            'chat_history': [],
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
    from .models import InvestmentProfile

    if not request.user.is_authenticated:
        return JsonResponse({
            'status': 'error',
            'message': 'Authentication required'
        }, status=401)

    if request.method == "GET":
        profile = get_user_investment_profile(request.user)
        return JsonResponse({'profile': profile})

    elif request.method == "POST":
        data = json.loads(request.body)
        text_input = data.get('text', '')

        # Get or create profile
        profile, created = InvestmentProfile.objects.get_or_create(
            user=request.user,
            defaults={
                'risk_tolerance': 'moderate',
                'investment_horizon': 'medium-term',
                'preferred_sectors': [],
                'avoided_sectors': [],
            }
        )

        # Add to chat history
        chat_history = profile.chat_history or []
        chat_history.append({
            'role': 'user',
            'content': text_input,
            'timestamp': datetime.now().isoformat()
        })

        # Update raw_text with latest input
        profile.raw_text = text_input
        profile.chat_history = chat_history

        # Simple keyword-based profile extraction (can be enhanced with LLM later)
        text_lower = text_input.lower()

        # Risk tolerance detection
        if any(word in text_lower for word in ['공격적', '고위험', '적극적', 'aggressive', 'high risk']):
            profile.risk_tolerance = 'aggressive'
        elif any(word in text_lower for word in ['보수적', '안정적', '저위험', 'conservative', 'safe', 'low risk']):
            profile.risk_tolerance = 'conservative'
        elif any(word in text_lower for word in ['중립', '보통', 'moderate', 'balanced']):
            profile.risk_tolerance = 'moderate'

        # Investment horizon detection
        if any(word in text_lower for word in ['단기', '1년 이내', 'short-term', 'short term']):
            profile.investment_horizon = 'short-term'
        elif any(word in text_lower for word in ['장기', '5년', '10년', 'long-term', 'long term']):
            profile.investment_horizon = 'long-term'
        elif any(word in text_lower for word in ['중기', '2년', '3년', 'medium-term', 'medium term']):
            profile.investment_horizon = 'medium-term'

        # Investment style detection
        if any(word in text_lower for word in ['성장', 'growth', '성장주']):
            profile.investment_style = 'growth'
        elif any(word in text_lower for word in ['가치', 'value', '가치주', '저평가']):
            profile.investment_style = 'value'
        elif any(word in text_lower for word in ['배당', 'income', 'dividend', '인컴']):
            profile.investment_style = 'income'
        elif any(word in text_lower for word in ['균형', 'balanced', '혼합']):
            profile.investment_style = 'balanced'

        # Custom instructions - save the text if it contains specific preferences
        custom_keywords = ['위주', '중시', '선호', '싫어', '피하고', '원해', '원함', '좋아',
                          'ESG', '변동성', '안전', '수익률', '리스크', '분산', '집중']
        if any(kw in text_lower for kw in custom_keywords):
            # Append to existing custom instructions or set new
            if profile.custom_instructions:
                profile.custom_instructions = f"{profile.custom_instructions}; {text_input}"
            else:
                profile.custom_instructions = text_input

        # Sector detection
        sector_keywords = {
            'tech': ['기술', '테크', 'tech', 'technology', 'IT', '반도체', 'AI', '인공지능'],
            'healthcare': ['헬스케어', '의료', '바이오', 'healthcare', 'biotech', 'pharma'],
            'finance': ['금융', '은행', 'finance', 'banking', 'fintech'],
            'energy': ['에너지', '친환경', 'energy', 'renewable', 'solar', 'green'],
            'consumer': ['소비재', '유통', 'consumer', 'retail'],
        }

        preferred = list(profile.preferred_sectors) if profile.preferred_sectors else []
        avoided = list(profile.avoided_sectors) if profile.avoided_sectors else []

        for sector, keywords in sector_keywords.items():
            if any(kw in text_lower for kw in keywords):
                if any(neg in text_lower for neg in ['싫', '피하', 'avoid', 'hate', '안좋', '제외']):
                    if sector not in avoided:
                        avoided.append(sector)
                    if sector in preferred:
                        preferred.remove(sector)
                else:
                    if sector not in preferred:
                        preferred.append(sector)
                    if sector in avoided:
                        avoided.remove(sector)

        profile.preferred_sectors = preferred
        profile.avoided_sectors = avoided

        # Add system response to chat history
        response_msg = f"프로필이 업데이트되었습니다. 투자 성향: {profile.risk_tolerance}, 투자 기간: {profile.investment_horizon}"
        if profile.investment_style:
            response_msg += f", 투자 스타일: {profile.investment_style}"
        if preferred:
            response_msg += f", 선호 섹터: {', '.join(preferred)}"
        if avoided:
            response_msg += f", 비선호 섹터: {', '.join(avoided)}"
        if profile.custom_instructions:
            response_msg += f", 커스텀 지시: {profile.custom_instructions}"

        chat_history.append({
            'role': 'assistant',
            'content': response_msg,
            'timestamp': datetime.now().isoformat()
        })
        profile.chat_history = chat_history

        profile.save()

        return JsonResponse({'profile': profile.to_dict()})


@require_http_methods(["POST"])
@csrf_exempt
def save_analysis(request):
    """
    Save analysis results to database
    POST /trading_agent/api/save_analysis/
    Body: {"ticker": "AAPL", "date": "2024-12-27", "decision": "BUY", "report": "...", "accuracy": {...}, "full_state": {...}}
    """
    if not request.user.is_authenticated:
        return JsonResponse({
            'status': 'error',
            'message': 'Authentication required'
        }, status=401)

    try:
        from .models import AnalysisHistory

        data = json.loads(request.body)
        ticker = data.get('ticker', '').upper()
        date_str = data.get('date', '')
        decision = data.get('decision', '')
        report = data.get('report', '')
        accuracy = data.get('accuracy', {})
        full_state = data.get('full_state', {})

        if not ticker or not date_str or not decision:
            return JsonResponse({
                'status': 'error',
                'message': 'ticker, date, and decision are required'
            }, status=400)

        analysis_date = datetime.strptime(date_str, "%Y-%m-%d").date()

        # Create analysis history record
        analysis = AnalysisHistory.objects.create(
            user=request.user,
            ticker=ticker,
            analysis_date=analysis_date,
            decision=decision,
            investment_plan=report,
            risk_assessment=full_state.get('risk', ''),
            technical_analysis=full_state.get('technical', ''),
            fundamental_analysis=full_state.get('fundamentals', ''),
            sentiment_analysis=full_state.get('sentiment', ''),
            start_price=accuracy.get('start_price'),
            end_price=accuracy.get('end_price'),
            return_pct=accuracy.get('return_pct', ''),
            is_accurate=accuracy.get('is_accurate')
        )

        return JsonResponse({
            'status': 'success',
            'analysis_id': analysis.id,
            'message': '분석 결과가 저장되었습니다.'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_analysis_history(request):
    """
    Get user's analysis history
    GET /trading_agent/api/history/
    """
    if not request.user.is_authenticated:
        return JsonResponse({
            'status': 'error',
            'message': 'Authentication required'
        }, status=401)

    try:
        from .models import AnalysisHistory

        analyses = AnalysisHistory.objects.filter(user=request.user).order_by('-created_at')

        history = []
        for analysis in analyses:
            history.append({
                'id': analysis.id,
                'ticker': analysis.ticker,
                'analysis_date': analysis.analysis_date.strftime('%Y-%m-%d'),
                'decision': analysis.decision,
                'return_pct': analysis.return_pct,
                'is_accurate': analysis.is_accurate,
                'created_at': analysis.created_at.strftime('%Y-%m-%d %H:%M:%S')
            })

        return JsonResponse({
            'status': 'success',
            'history': history
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_analysis_detail(request, analysis_id):
    """
    Get detailed analysis by ID
    GET /trading_agent/api/history/<analysis_id>/
    """
    if not request.user.is_authenticated:
        return JsonResponse({
            'status': 'error',
            'message': 'Authentication required'
        }, status=401)

    try:
        from .models import AnalysisHistory

        analysis = AnalysisHistory.objects.get(id=analysis_id, user=request.user)

        detail = {
            'id': analysis.id,
            'ticker': analysis.ticker,
            'analysis_date': analysis.analysis_date.strftime('%Y-%m-%d'),
            'decision': analysis.decision,
            'investment_plan': analysis.investment_plan,
            'risk_assessment': analysis.risk_assessment,
            'technical_analysis': analysis.technical_analysis,
            'fundamental_analysis': analysis.fundamental_analysis,
            'sentiment_analysis': analysis.sentiment_analysis,
            'start_price': str(analysis.start_price) if analysis.start_price else None,
            'end_price': str(analysis.end_price) if analysis.end_price else None,
            'return_pct': analysis.return_pct,
            'is_accurate': analysis.is_accurate,
            'created_at': analysis.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

        return JsonResponse({
            'status': 'success',
            'analysis': detail
        })

    except AnalysisHistory.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Analysis not found'
        }, status=404)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@require_http_methods(["DELETE"])
@csrf_exempt
def delete_analysis(request, analysis_id):
    """
    Delete analysis by ID
    DELETE /trading_agent/api/history/<analysis_id>/delete/
    """
    if not request.user.is_authenticated:
        return JsonResponse({
            'status': 'error',
            'message': 'Authentication required'
        }, status=401)

    try:
        from .models import AnalysisHistory

        analysis = AnalysisHistory.objects.get(id=analysis_id, user=request.user)
        analysis.delete()

        return JsonResponse({
            'status': 'success',
            'message': '분석 결과가 삭제되었습니다.'
        })

    except AnalysisHistory.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Analysis not found'
        }, status=404)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)
