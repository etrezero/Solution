# 운용보고서 자동화 코멘트 생성기 (Dash UI 통합)

# 표준 라이브러리
import os
import sys
import re
import json
import yaml
import base64
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
from dateutil.relativedelta import relativedelta

# 외부 패키지
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
from openpyxl import Workbook
from sklearn.feature_extraction.text import CountVectorizer

import fitz  # PyMuPDF
import pymysql
import requests
import concurrent.futures
import win32com.client

# 금융 데이터
import yfinance as yf
from pykrx import stock as pykrx

# Dash 관련
import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

# Plotly 시각화
import plotly.graph_objs as go
import plotly.express as px

# 사용자 정의 함수
from get_hostname_from_ip import get_hostname_from_ip
import pythoncom





# 기본 CONFIG 경로 설정
DEFAULT_CONFIG_PATH = r"C:\Covenant\config.yaml"
LOCAL_CONFIG_PATH = r"C:\Covenant\config.yaml"

# 호스트에서 API 키 불러오기 (서버 내부용)
def load_openai_key():
    with open(LOCAL_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["openai_api_key"]

def load_news_api_key():
    with open(LOCAL_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["newsapi"]







cache_price = r'C:\Covenant\cache\Market_Comment_price.pkl'
cache_expiry = timedelta(days=1)



# 주식 코드 목록 설정
code_dict = {
    'ACWI': '글로벌주식',
    'ACWX': '미국외 글로벌주식',
    'BND': '글로벌채권',
    'DIA': '미국주식',
    'VUG': '미국성장주',
    'VTV': '미국가치주',
    'VEA': '선진국주식',
    'VWO': '신흥국주식',
    'MCHI': 'MSCI중국주식',
    'HYG': '미국하이일드채권',
    'GLD': '금',
    'KRW=X': '원화환율',
    '356540.KS': '한국주식',

  # 🔽 추가된 섹터 ETF
    'XLF': '미국금융섹터',
    'IYF': '미국금융지주',
    'XLK': '미국기술섹터',
    'XLY': '미국소비재섹터',
    'XLE': '미국에너지섹터',
    'XLV': '미국헬스케어섹터',
    'XLI': '미국산업섹터',
    'XLP': '미국필수소비재섹터',
    'XLU': '미국유틸리티섹터',
    'XLC': '미국커뮤니케이션섹터',
    'XLB': '미국소재섹터',

}

code = list(code_dict.keys())  # 주식 코드 목록


#? pip install curl_cffi
# 데이터 가져오기 함수
def fetch_data(code, start, end):
    from curl_cffi import requests

    try:
        if isinstance(code, int) or code.isdigit():
            if len(code) == 5:
                code = '0' + code
            df_price = pykrx.get_market_ohlcv_by_date(start, end, code)['종가']
        else:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            session = requests.Session(impersonate="chrome")
            session.verify = False  # SSL 인증서 검증 비활성화
            yf_data = yf.Ticker(code, session=session)
            df_price = yf_data.history(start=start, end=end)['Close']
            df_price = df_price.tz_localize(None)  # 타임존 제거

        df_price = pd.DataFrame(df_price)
        df_price.columns = [code]
        df_price.index = pd.to_datetime(df_price.index).strftime('%Y-%m-%d')  # 인덱스를 문자열 형식으로 변환
        df_price = df_price.sort_index(ascending=True)
        
        return df_price
    
    except Exception as e:
        print(f"Error fetching data for {code}: {e}")
        return None


# 캐시 데이터 로딩 및 데이터 병합 처리 함수
def Func(code, start, end, batch_size=10):
    if os.path.exists(cache_price):
        cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_price))
        if datetime.now() - cache_mtime < cache_expiry:
            with open(cache_price, 'rb') as f:
                print("Loading data from cache...")
                return pickle.load(f)

    data_frames = []
    for i in range(0, len(code), batch_size):
        code_batch = code[i:i + batch_size]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(fetch_data, c, start, end): c for c in code_batch}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    data_frames.append(result)

    price_data = pd.concat(data_frames, axis=1) if data_frames else pd.DataFrame()
    price_data = price_data.sort_index(ascending=True)
    print("price_data=================\n", price_data)

    with open(cache_price, 'wb') as f:
        pickle.dump(price_data, f)
        print("Data cached.")

    return price_data




# 오늘 날짜와 기간 설정

from pandas.tseries.offsets import MonthEnd

# 오늘 기준 지지난달 말, 지난달 말 계산
today = datetime.today()
last_month_end = (today - MonthEnd(1)).date()      # 지난달 말
prev_month_end = (today - MonthEnd(2)).date()      # 지지난달 말


df_price = Func(code, prev_month_end, last_month_end, batch_size=10)
df_price = df_price.ffill()





# 인덱스를 datetime으로 변환 (문자열이면)
df_price.index = pd.to_datetime(df_price.index)
df_price.columns = [code_dict.get(col, col) for col in df_price.columns]
print("df_price===============", df_price)


# 지지난달 말과 지난달 말의 데이터만 필터링
df_month_ends = df_price.loc[df_price.index.isin([prev_month_end, last_month_end])]

# 수익률 계산
월간수익률 = df_month_ends.pct_change().iloc[-1:].copy()  # 마지막 행만 (지난달말 기준 수익률)
월간수익률.index = [last_month_end]  # 인덱스를 날짜로 설정

월간수익률 = (월간수익률 * 100).round(1).astype(str) + '%'

# 결과 출력
print("📈 월간 수익률 ====================", 월간수익률)







# Dash 앱 생성
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Market Comment"


# 레이아웃 정의
app.layout = dbc.Container([
    html.H3("Market Comment 생성기", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label("✅ config.yaml 경로", className="fw-bold mb-1"),
                    dcc.Input(id='config-path', value=DEFAULT_CONFIG_PATH, type='text', style={"width": "100%"}),

                    html.Br(), html.Br(),

                    html.Label("📝 명령어 (예: 시장현황과 시장전망 작성해줘)", className="fw-bold mb-1"),
                    dcc.Textarea(
                        id='nl-command', 
                        value=f"시장현황과 시장전망 작성해줘. news와 PDF의 내용은 경제와 자산군에 대한 설명에 보조적 역할을 하는 수준으로만 활용", 
                        style={'width': '100%', 'height': 100}),

                    html.Br(),
                    dbc.Button("코멘트 생성 실행", id='run-button', color='primary', className='w-100')
                ])
            ])
        ], width=4, style={"height": "300vh", "overflowY": "auto"}),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Market Comment", className="fw-bold"),
                dbc.CardBody([
                    dcc.Loading(
                        html.Div(id='reply-output', style={'whiteSpace': 'pre-wrap', 'minHeight': '400px'}),
                        type='circle'
                    ),

                html.Br(),
                    dbc.Button("📋 코멘트 복사", id="copy-button", color="secondary", className="mt-2"),
                    
                    html.Div(id="dummy-output", style={"display": "none"}),  # 콜백 결과를 받는 더미 Div

                

                ])
            ])
        ], width=7, style={"height": "200vh", "overflowY": "auto"}),



        
    ])
], fluid=True, style={"maxWidth": "70%", "margin": "0 auto"})




# 오류 메시지 파싱 함수
def extract_missing_variable(error_msg):
    match = re.search(r"NameError: name '(\w+)' is not defined", error_msg)
    return match.group(1) if match and match.group(1) not in ['fig', 'df', 'table', 'text'] else None




#? <PDF 요약>============================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
PDF_DIR = Path("assets/Outlook_PDFs")
PDF_DIR.mkdir(parents=True, exist_ok=True)

# Outlook에서 PDF 저장
def fetch_recent_pdf_attachments(days=30, save_dir=PDF_DIR):
    pythoncom.CoInitialize()  # ✅ COM 초기화
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    inbox = outlook.GetDefaultFolder(6)
    messages = inbox.Items
    messages.Sort("[ReceivedTime]", True)
    recent_date = datetime.now() - timedelta(days=days)
    saved_files = []
    for message in messages:
        try:
            if isinstance(message.ReceivedTime, datetime) and message.ReceivedTime.replace(tzinfo=None) < recent_date:
                break
            if message.Class == 43 and message.Attachments.Count > 0:
                received_date = message.ReceivedTime.strftime("%Y-%m-%d")
                for i in range(1, message.Attachments.Count + 1):
                    attachment = message.Attachments.Item(i)
                    if attachment.FileName.lower().endswith(".pdf"):
                        filename = f"{received_date}_{attachment.FileName}"
                        filepath = os.path.join(save_dir, filename)
                        attachment.SaveAsFile(filepath)
                        saved_files.append((filepath, received_date))
        except Exception as e:
            logger.warning(f"⚠️ 메시지 오류: {e}")
    return saved_files



# PDF 텍스트 추출 및 정제
def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)

def filter_relevant_text(text, min_length=30):
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if len(line) < min_length: continue
        if sum(c.isdigit() for c in line) > len(line) * 0.4: continue
        if re.search(r"수익률|1D|3M|YTD|mailto:|@\w+|https?://", line, re.IGNORECASE): continue
        if re.match(r"^[-=•│\d\s]{3,}$", line): continue
        clean_lines.append(line)
    return "\n".join(dict.fromkeys(clean_lines))



# GPT 요약 (requests 방식)
def summarize_text(text):
    api_key = load_openai_key()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    prompt = f"금융시장이나 경제분석에 사용할 수 있는 핵심 요점 15개로 요약해 주세요.\n\n{text[:2000]}"

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json={
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        },
        verify=False
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        return f"Error: {response.status_code}"

#? ==================================================================



#? ====================================================================
def generate_pdf_summary(days=30):
    import hashlib
    """
    최근 Outlook PDF 첨부파일을 수집하고, 텍스트를 추출 및 정제한 후
    OpenAI GPT를 통해 요약하여 PDF_summary 문자열을 반환합니다.
    1일 캐시 사용.
    """
    try:
        # ✅ 캐시 디렉토리 및 키 생성
        CACHE_DIR = Path("cache/pdf_summary_cache")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_expiry = timedelta(days=1)

        # 키 생성 (기간이 달라지면 다르게 캐싱)
        cache_key = f"pdf_summary_{days}"
        cache_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
        cache_path = CACHE_DIR / f"{cache_hash}.txt"

        # ✅ 캐시가 유효하면 반환
        if cache_path.exists():
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if datetime.now() - mtime < cache_expiry:
                with open(cache_path, "r", encoding="utf-8") as f:
                    print("🔁 PDF 요약 캐시 사용")
                    return f.read()

        # ✅ 최신 PDF 첨부파일 가져오기
        pdf_files = fetch_recent_pdf_attachments(days=days)
        if not pdf_files:
            return "최근 30일간 저장된 PDF 첨부파일이 없습니다."

        # ✅ 텍스트 추출 및 정제
        pdf_texts = []
        for path, date in pdf_files:
            try:
                raw_text = extract_text_from_pdf(path)
                clean_text = filter_relevant_text(raw_text)
                if clean_text:
                    pdf_texts.append(clean_text)
            except Exception as e:
                logger.warning(f"⚠️ PDF 처리 오류 ({path}): {e}")

        combined_pdf_text = "\n\n".join(pdf_texts)

        # ✅ GPT 요약
        if combined_pdf_text.strip():
            summary = summarize_text(combined_pdf_text)

            # ✅ 캐시에 저장
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(summary)

            print("✅ PDF 요약 캐시 저장됨")
            return summary
        else:
            return "PDF 텍스트가 충분하지 않아 요약할 수 없습니다."

    except Exception as e:
        logger.error(f"❌ PDF 요약 생성 실패: {e}")
        return f"오류 발생: {str(e)}"
#? ====================================================================







#? <뉴스요약>============================================

def fetch_news_summary(query, start_date=None, end_date=None, page_size=5):
    import hashlib
    import requests
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # ✅ 캐시 설정
    CACHE_DIR = Path("cache/news_cache")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_expiry = timedelta(days=1)

    # ✅ 캐시 키 생성 (query + dates)
    key_string = f"{query}_{start_date}_{end_date}_{page_size}"
    key_hash = hashlib.md5(key_string.encode('utf-8')).hexdigest()
    cache_path = CACHE_DIR / f"{key_hash}.json"

    # ✅ 캐시 유효성 검사
    if cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime < cache_expiry:
            with open(cache_path, "r", encoding="utf-8") as f:
                print(f"🔁 뉴스 캐시 사용: {query}")
                return json.load(f)

    # ✅ API 호출
    api_key = load_news_api_key()
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key
    }

    if start_date:
        params["from"] = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    if end_date:
        params["to"] = pd.to_datetime(end_date).strftime('%Y-%m-%d')

    try:
        res = requests.get(url, params=params, verify=False)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ 뉴스 요청 실패: {e}")
        return []

    articles = res.json().get("articles", [])
    summaries = [
        f"""- {a['title']} ({a['source']['name']})
  📌 {a.get('description', '').strip()}
  🔎 {a.get('content', '').split('…')[0].strip()}"""
        for a in articles if a.get("description")
    ]

    # ✅ 캐시 저장
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"✅ 뉴스 캐시 저장됨: {query}")
    return summaries
# ?==========================================================================



def get_news_digest(start_date=None, end_date=None, include_region=True, include_asset_class=True):
    topics = {}

    if include_region:
        topics.update({
            "미국": "US economy stock market FED inflation interest rate",
            "유럽": "Europe ECB inflation interest rate",
            "일본": "Japan JGB inflation interest rate",
            "이머징": "Emerging Stock Market",
        })

    if include_region or include_asset_class:
        topics["경제지표"] = "economic indicators US Europe Japan Emerging"

    if include_asset_class:
        topics["자산군"] = (
            "US Growth Stocks US Value Stocks US Treasury "
            "US Highyield US Real Estate Gold Oil "
            "Developed Market Emerging Market"
        )

    result = []
    for region, query in topics.items():
        summaries = fetch_news_summary(query, start_date=start_date, end_date=end_date)
        if summaries:
            result.append(f"[{region}]\n" + "\n".join(summaries))

    return "\n\n".join(result)



from datetime import datetime, timedelta

# 예: 지난 30일 뉴스 요약
today = datetime.today()
start = (today - timedelta(days=30)).strftime('%Y-%m-%d')
end = today.strftime('%Y-%m-%d')

news_summary = get_news_digest(start_date=start, end_date=end)
print("news_summary=====================\n", news_summary)











# GPT 기반 운용보고서 생성 함수
def Run_GPT(nl_command, news_summary, PDF_summary, config_path, error_msg=None):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            api_key = yaml.safe_load(f)["openai_api_key"]
    except Exception as e:
        return f"❌ CONFIG 파일 로드 실패: {e}", None

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    prompt = f"""
        - 사용자 명령: {nl_command}
        - news_summary : {news_summary}
        - PDF_summary : {PDF_summary}
        - 시장현황, 시장전망, 운용계획 등의 전체 내용 약 1500~2500자 이내로 코멘트 작성해줘.
        - 각 항목은 반드시 대괄호로 구분된 제목을 포함해줘 (예: [시장현황])
        - 사용자명령, news_summary, PDF_summary를 반영하지만 이 출처는 언급하지마.
        - news_summary, PDF_summary는 경제와 자산군에 대한 설명에 보조적 역할을 하는 수준으로만 활용해주고 주로 ETF관점으로 서술해줘.
        - 경제지표는 구체적인 수치를 언급해줘.
        - 자산군과 국가, region 이외의 주제 외에는 모두 보완적으로만 언급해줘.
        - 시장현황에 지역별/섹터별 {월간수익률}을 일부 언급해줘. 
        - **동일 주제 반복 금지**: 동일한 주제나 내용이 반복되지 않도록 주의해줘.
        - 같은 문장 어미가 반복되지 않도록 어미를 변경해줘.
        - **투자비중 수치 언급 금지**
        - 운용계획은 자산군과 region별 긍정적 부정적 견해로 서술해줘. 
        - 숫자는 소수점 한 자리 (예: 12.1%)로 표기해줘.
    """

    if error_msg:
        missing = extract_missing_variable(error_msg)
        prompt += f"\n# 오류:\n{error_msg}"
        if missing:
            prompt += f"\n💡 '{missing}'는 정의되지 않은 변수입니다. 이를 포함해 다시 생성해주세요."

    res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2000
    }, verify=False)

    if res.status_code != 200:
        return f"# 오류 발생: {res.status_code}\n{res.text}", prompt

    return res.json()['choices'][0]['message']['content'].strip(), prompt

# 콜백 정의
@app.callback(
    Output('reply-output', 'children'),
    Input('run-button', 'n_clicks'),
    State('config-path', 'value'),
    State('nl-command', 'value'),
    prevent_initial_call=True
)
def update_output(n_clicks, config_path, nl_command):
    news_summary = get_news_digest(start_date=start, end_date=end)
    PDF_summary = generate_pdf_summary(days=30)
    reply, _ = Run_GPT(nl_command, news_summary, PDF_summary, config_path)
    return reply


app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return '';
        const text = document.getElementById('reply-output')?.innerText || '';
        if (text) {
            navigator.clipboard.writeText(text).then(function() {
                console.log("📋 복사 완료");
            }).catch(function(err) {
                console.error("❌ 복사 실패", err);
            });
        }
        return '';
    }
    """,
    Output('dummy-output', 'children'),  # 아무 데도 출력하지 않음
    Input('copy-button', 'n_clicks'),
    prevent_initial_call=True
)



# 서버 실행
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)

