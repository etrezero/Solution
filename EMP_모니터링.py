import yfinance as yf
from pykrx import stock as pykrx
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.graph_objs as go
from dash import html, dcc, Input, Output, State
import concurrent.futures
import requests
import dash
import pickle
import os
from openpyxl import Workbook

import numpy as np


from flask import Flask
import socket




cache_price = r'C:\Covenant\cache\EMP_모니터링.pkl'
cache_expiry = timedelta(days=1)



# 주식 코드 목록 설정
code_dict = {
    'code_Asset':
        ['ACWI', 'ACWX', 'BND', 'DIA', 'VUG', 'VTV', 'VEA', 'IDEV', 'VWO', 'MCHI', 'AVEM', 'EEM', 'IEMG', 'HYG', 'GLD', 'KRW=X', '356540.KS'],
   

    'Country':
    ['^KS11',  # KOSPI (한국 종합주가지수)
     '^GSPC',  # S&P 500 (미국 S&P 500 지수)
     '^IXIC',  # NASDAQ (나스닥 종합지수)
     '^DJI',   # Dow Jones (다우존스 산업평균지수)
     '^FTSE',  # FTSE 100 (영국 FTSE 100 지수)
     '^N225',  # Nikkei 225 (일본 니케이 225 지수)
     '^GDAXI', # DAX (독일 DAX 지수)
     '^FCHI',  # CAC 40 (프랑스 CAC 40 지수)
     '^HSI',   # Hang Seng (홍콩 항셍지수)
     '^SSEC',  # Shanghai Composite (중국 상하이 종합지수)
     '^AXJO',  # ASX 200 (호주 ASX 200 지수)
     '^TSX',   # TSX Composite (캐나다 TSX 종합지수)
     'KRW=X'],  # South Korean Won (KRW) to USD exchange rate


    'code_US': 
        ['QQQ', 'SOXX', 'SPY', 'DIA', 'VUG', 'SPYG', 'IWF', 'VTV', 'MGV', 'SPYV', 'IWD', 'MAGS', 'XMAG', 'IWM'],
  
    'code_국내주식': 
        ['069500.KS', '000660.KS', '005930.KS', '035420.KS', '207940.KS', '035720.KS', '068270.KS', '051910.KS', '005380.KS', '006400.KS', '035720.KS'],
 
    'code_국내채권': 
        ['356540.KS', '148070.KS', '273130.KS', '439870.KS', '114460.KS', '365780.KS'],

    'code_Big7': 
        ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META'],
 
    'code_테마': 
        ['NYMT', 'PAVE', 'DAT', 'DRIV', 'MAGS', 'URA', 'CLOU', ],
 
    'code_Sector': 
        ['QQQ', 'SPY', 'XLF', 'IYF', 'XLK', 'XLY', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLC', 'XLB'],
    
    'code_FI': 
        ['BND', 'BNDX', 'HYG', 'XHYC', 'XHYD', '356540.KS', '190620.KS'],
 
    'S자산배분': 
        ['URTH', 'QQQ', 'PPA', 'SOXX', 'IWF', 'INDA', 'IEMG', 'SPY', 'VEA', 'VUG', 'VTI', 'IYF', '069500.KS'],

    'code_Currency' : 
        [
        'KRW=X', 
        'GLD',
        'CNY=X', 
        'EURUSD=X', 
        'JPY=X', 
        'GBPUSD=X', 
        'CHF=X', 'CAD=X', 'AUDUSD=X', 'NZDUSD=X', 'SEK=X', 'NOK=X', 'HKD=X', 'SGD=X'
        ],

    '금리': 
        ['^IRX', '^FVX', '^TNX', '^TYX', ],
        # '^IRX':UST3M, '^TXY':UST2Y, '^FVX':UST5Y, '^TNX':UST10Y,  '^TYX': UST30Y,

}



# 모든 리스트를 하나로 합치고 중복 제거
code = list(set([item for sublist in code_dict.values() for item in sublist]))

print(code)


# 데이터 가져오기 함수
def fetch_data(code, start, end):
    try:
        if isinstance(code, int) or code.isdigit():
            if len(code) == 5:
                code = '0' + code
            df_price = pykrx.get_market_ohlcv_by_date(start, end, code)['종가']
        else:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            session = requests.Session()
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
today = datetime.now()
start = today - timedelta(days=365*1)
end = today


df_price = Func(code, start, end, batch_size=10)
df_price = df_price.ffill()
print("df_price===============", df_price)





#엑셀 저장=======================================================
def save_excel(df, sheetname, index_option=None):
    
    # 파일 경로
    path = rf'C:\Covenant\data\모니터링_EMP.xlsx'

    # 파일이 없는 경우 새 Workbook 생성
    if not os.path.exists(path):
        wb = Workbook()
        wb.save(path)
        print(f"새 파일 '{path}' 생성됨.")
    
    # 인덱스를 날짜로 변환 시도
    try:
        # index_option이 None일 경우 인덱스를 포함하고 날짜 형식으로 저장
        if index_option is None or index_option:  # 인덱스를 포함하는 경우
            df.index = pd.to_datetime(df.index, errors='raise')  # 변환 실패 시 오류 발생
            df.index = df.index.strftime('%Y-%m-%d')  # 벡터화된 방식으로 날짜 포맷 변경
            index = True  # 인덱스를 포함해서 저장
        else:
            index = False  # 인덱스를 제외하고 저장
    except Exception:
        print("Index를 날짜 형식으로 변환할 수 없습니다. 기본 인덱스를 사용합니다.")
        index = index_option if index_option is not None else True  # 변환 실패 시에도 인덱스를 포함하도록 설정

    # DataFrame을 엑셀 시트로 저장
    with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheetname, index=index)  # index 여부 설정
        print(f"'{sheetname}' 저장 완료.")




# Flask 서버 생성
server = Flask(__name__)
app = dash.Dash(__name__, server=server)
app.title = '모니터링_EMP'






# 연도 선택을 위한 Dropdown 옵션 생성
current_year = datetime.today().year
year_options = [{'label': str(year), 'value': year} for year in range(current_year - 10, current_year + 1)]

# 기간에 따른 데이터 범위 설정 함수
def get_period_range(period):
    today = datetime.today()
    if period == '1M':
        return today - relativedelta(months=1), today
    elif period == '3M':
        return today - relativedelta(months=3), today
    elif period == '1Y':
        return today - relativedelta(years=1), today
    elif period == '3Y':
        return today - relativedelta(years=3), today
    elif period == 'YTD':
        return datetime(today.year, 1, 1), today
    else:
        return None, None

period_options = [
    {'label': '1M', 'value': '1M'},
    {'label': '3M', 'value': '3M'},
    {'label': '1Y', 'value': '1Y'},
    {'label': '3Y', 'value': '3Y'},
    {'label': 'YTD', 'value': 'YTD'}
]





@app.callback(
    [Output('stock-dropdown', 'options'),
     Output('stock-dropdown', 'value')],  # 🔹 기본값 설정을 위한 추가 Output
    Input('code-group-dropdown', 'value')
)
def update_stock_options(selected_group):
    if selected_group and selected_group in code_dict:
        options = [{'label': code, 'value': code} for code in code_dict[selected_group]]
        return options, options[0]['value'] if options else None  # 🔹 첫 번째 항목을 기본값으로 설정
    return [], None  # 선택된 그룹이 없을 경우 빈 리스트 반환



@app.callback(
    [Output('cumulative-return-graph', 'figure'),
     Output('fig_Continue-graph', 'figure'),
     Output('fig_trend-graph', 'figure'),
     Output('Digital-cum-graph', 'figure'),
     Output('Continue_return-graph', 'figure'),
     Output('rolling-return-graph', 'figure'),
     Output('Volatility-graph', 'figure'),
     Output('DD_1Y-graph', 'figure')],
    [Input('stock-dropdown', 'value'),
     Input('year-dropdown', 'value')],
    [State('code-group-dropdown', 'value')]  # State로 선택된 그룹 정보를 전달
)
def update_graph(selected_code, selected_year, selected_group):    # 시작 및 끝 날짜 설정
    start = datetime(selected_year, 1, 1).strftime('%Y-%m-%d')
    end = datetime.today().strftime('%Y-%m-%d')

    # 선택된 종목에 대한 데이터를 가져오기
    df_price = fetch_data(selected_code, start, end).dropna()
    if df_price is None or df_price.empty:
        return {}, {}, {}, {}

    # 일간 수익률 계산
    df_price['Daily_Return'] = df_price[selected_code].pct_change().fillna(0)


    # 선택된 종목의 누적 수익률 계산=================================
    df_price['cum'] = (1 + df_price['Daily_Return']).cumprod() - 1
    df_price['cum'] -= df_price['cum'].iloc[0]  # 첫날 수익률을 0으로 설정




    # RSI 계산 함수 추가===========================================
    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


    # RSI 계산 (14일)
    df_price['RSI_14'] = calculate_rsi(df_price[selected_code], window=14)

    # RSI_14가 30 이하일 때 Weight는 0.8
    # RSI_14가 30 초과 70 미만일 때 Weight는 0.9
    # RSI_14가 70 이상일 때 Weight는 1    
    df_price['Weight'] = df_price['RSI_14'].apply(lambda x: 0.5 if x <= 30 else (0.75 if x < 70 else 1))

    # 비중을 반영하여 누적 수익률(MP) 계산
    df_price['MP'] = (1 + df_price['Daily_Return'] * df_price['Weight'].fillna(0)).cumprod() - 1
    df_price['MP'] -= df_price['MP'].iloc[0]  # 첫날 수익률을 0으로 설정

    # save_excel(df_price, 'EMP', index_option=True)



    # 베이스라인 누적수익률 그래프 ======================================
    min_idx = df_price['cum'].idxmin()
    max_idx = df_price['cum'].idxmax()

    # 각 변곡점의 값
    min_value = df_price['cum'].loc[min_idx]
    max_value = df_price['cum'].loc[max_idx]

    # 시작점, 끝점의 값
    start_value = df_price['cum'].iloc[0]
    end_value = df_price['cum'].iloc[-1]

    # 변곡점 순서에 따른 구간 나누기
    if min_idx < max_idx:
        # 최저점이 먼저 오는 경우
        segments = [
            (df_price['cum'].index[0], min_idx, start_value, min_value),
            (min_idx, max_idx, min_value, max_value),
            (max_idx, df_price['cum'].index[-1], max_value, end_value),
        ]
    else:
        # 최고점이 먼저 오는 경우
        segments = [
            (df_price['cum'].index[0], max_idx, start_value, max_value),
            (max_idx, min_idx, max_value, min_value),
            (min_idx, df_price['cum'].index[-1], min_value, end_value),
        ]

    # 베이스라인 생성
    baseline = np.full_like(df_price['cum'].values, fill_value=np.nan, dtype=np.float64)

    for start_seg, end_seg, start_val, end_val in segments:
        start_idx = df_price['cum'].index.get_loc(start_seg)
        end_idx = df_price['cum'].index.get_loc(end_seg) + 1
        slope = (end_val - start_val) / (end_idx - start_idx - 1)
        baseline[start_idx:end_idx] = start_val + slope * np.arange(end_idx - start_idx)

    # 차이 계산
    diff = df_price['cum'].values - baseline
    positive_area = np.where(diff > 0, diff, 0)
    negative_area = np.where(diff < 0, diff, 0)


    # 선택한 종목의 누적 수익률 그래프 생성
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(
        go.Scatter(x=df_price.index, y=df_price['cum'], mode='lines', name=selected_code),
    )
    fig_cumulative.add_trace(
        go.Scatter(x=df_price.index, y=df_price['MP'], mode='lines', name=f'{selected_code} MP RSI(14day)'),
    )
    fig_cumulative.add_trace(
        go.Scatter(x=df_price.index, y=df_price['RSI_14'], mode='lines', name=f'{selected_code} RSI(14-day) ', yaxis='y2'),
    )

    fig_cumulative.update_layout(
        title=f'{selected_code} Return & RSI ({selected_year})',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2=dict(title='RSI', overlaying='y', side='right', range=[0, 300]),
        yaxis={'tickformat': ',.1%'},  # y축 포맷 설정
        xaxis=dict(range=[df_price.index.min(), df_price.index.max()]),
        template='plotly_white',
    )





    # 그래프 생성
    fig_trend = go.Figure()

    # 누적 수익률
    fig_trend.add_trace(
        go.Scatter
        (x=df_price['cum'].index, y=df_price['cum'], mode='lines', name='Cumulative Return'))

    # 베이스라인
    fig_trend.add_trace(
        go.Scatter(x=df_price['cum'].index, y=baseline, mode='lines', name='Baseline', line=dict(dash='dash')))

    # +영역
    fig_trend.add_trace(
        go.Scatter(
        x=df_price['cum'].index, y=positive_area, mode='lines', fill='tozeroy', name='Positive Area',
        line=dict(color='green', width=0)
    ))

    # -영역
    fig_trend.add_trace(go.Scatter(
        x=df_price['cum'].index, y=negative_area, mode='lines', fill='tozeroy', name='Negative Area',
        line=dict(color='red', width=0)
    ))

    # 그래프 레이아웃 설정
    fig_trend.update_layout(
        title=f'{selected_code} Return with Baselines',
        xaxis_title='Date',
        yaxis_title='Return',
        yaxis=dict(tickformat=',.0%'),
        template='plotly_white'
    )






    def Continue_return(df):
        Continue = []
        current_value = 0
        prev_value = 0  # 이전 Daily_Return 값

        for return_value in df['Daily_Return']:
            if (prev_value > 0 and return_value <= 0) or (prev_value <= 0 and return_value > 0):
                # 변화가 있을 때 초기화
                current_value = 0
            elif return_value > 0:
                # +인 경우 누적 증가
                current_value += 1
            elif return_value <= 0:
                # -인 경우 누적 감소
                current_value -= 1

            Continue.append(current_value)
            prev_value = return_value  # 이전 값을 업데이트

        df['Continuation_Return'] = Continue
        return df


    df_price = Continue_return(df_price)

    # 그래프 생성
    fig_Continue = go.Figure()

    # 누적 수익률
    fig_Continue.add_trace(
        go.Scatter(
            x=df_price['cum'].index,
            y=df_price['cum'],
            mode='lines',
            name='Cumulative Return'
        )
    )

    # Continuation_Return (보조축)
    fig_Continue.add_trace(
        go.Scatter(
            x=df_price['Continuation_Return'].index,
            y=df_price['Continuation_Return'],
            mode='lines',
            name='Continuation_Return',
            fill='tozeroy',
            # line=dict(dash='dash'),
            yaxis='y2'  # 보조축으로 설정
        )
    )


    # 그래프 레이아웃 설정
    fig_Continue.update_layout(
        title=f'{selected_code} Return with Continuation',
        xaxis_title='Date',
        yaxis_title='Return',
        yaxis=dict(
            tickformat=',.0%',  # 주축의 포맷 설정 (퍼센트)
            title='Cumulative Return'
        ),
        yaxis2=dict(
            title='Continuation Return',
            overlaying='y',  # 기존 y축과 겹치도록 설정
            side='right',  # 보조축을 오른쪽에 배치
            tickformat=',.0f',
            range=[-5, 15],  # 보조축 범위 설정  # 보조축의 포맷 설정 (정수)
        ),
        template='plotly_white'
    )







    # Digital cum 수익률 그래프 생성============================
    fig_Digital_cum = go.Figure()
    fig_Continue_return = go.Figure()
    fig_Rolling_Return = go.Figure()
    fig_RV_3M = go.Figure()
    fig_DD_1Y = go.Figure()


    # 선택된 그룹 데이터 확인 및 적용
    if selected_group in code_dict:
        for code in code_dict[selected_group]:
            df_temp = fetch_data(code, start, end)
            if df_temp is None or df_temp.empty:
                continue

            df_temp['Daily_Return'] = df_temp[code].pct_change().fillna(0)
            df_temp['Weekly_Return'] = df_temp[code].pct_change(7).fillna(0)
            df_temp['M_Return'] = df_temp[code].pct_change(20).fillna(0)
            

            # 롤링 수익률 그래프 추가==============================
            df_temp['Rolling_Return_3M'] = (df_temp['Daily_Return'].rolling(window=60).sum()).mean()
            

            df_temp_sorted = df_temp['Weekly_Return'].sort_index(ascending=False)
            weekly_returns = df_temp_sorted[::7]
            weekly_returns_reversed = weekly_returns[::-1]
            df_temp['Volatility'] = weekly_returns_reversed.std() * np.sqrt(52)

            # Digital 수익률 그래프 추가===========================
            df_temp['Digital_Return'] = df_temp['M_Return'].apply(lambda x: 1 if x > 0 else 0)
            df_temp['Digital_cum'] = df_temp['Digital_Return'].cumsum().fillna(0)
            

            df_temp['DD_1Y'] = (df_temp[code] / df_temp[code].rolling(window=60).max()) - 1
    
            df_temp = df_temp.iloc[60:]
            
            print("df_temp====================", df_temp)


            fig_Digital_cum.add_trace(
                go.Scatter(x=df_temp.index, y=df_temp['Digital_cum'], mode='lines', name=f'{code}')
            )

            # 마지막 데이터 포인트에 어노테이션 추가
            fig_Digital_cum.add_annotation(
                x=df_temp.index[-1],
                y=df_temp['Digital_cum'].iloc[-1],
                text=f'{code}',  # 항목 이름
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-30,
                font=dict(size=12, color="blue"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="blue",
                borderwidth=1
            )

            fig_Digital_cum.update_layout(
                title=f'Monthly Digital Return (from {selected_year})',
                xaxis=dict(range=[df_temp.index.min(), df_temp.index.max()]),
                yaxis={'tickformat': ',.0f'},  # y축 포맷은 정수로 설정
                xaxis_title='Date',
                yaxis_title='Monthly Digital Return',
                template='plotly_white'
            )




            # Continue_return================================
            df_temp = Continue_return(df_temp)

            fig_Continue_return.add_trace(
                go.Scatter(x=df_temp.index, y=df_temp['Continuation_Return'], mode='lines', name=f'{code}')
            )

            # 마지막 데이터 포인트에 어노테이션 추가
            fig_Continue_return.add_annotation(
                x=df_temp.index[-1],
                y=df_temp['Continuation_Return'].iloc[-1],
                text=f'{code}',  # 항목 이름
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-30,
                font=dict(size=12, color="blue"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="blue",
                borderwidth=1
            )

            fig_Continue_return.update_layout(
                title=f'Daily Continuation Return (from {selected_year})',
                xaxis=dict(range=[df_temp.index.min(), df_temp.index.max()]),
                yaxis={'tickformat': ',.0f'},  # y축 포맷은 정수로 설정
                xaxis_title='Date',
                yaxis_title='Daily Continuation Return',
                template='plotly_white'
            )









            fig_Rolling_Return.add_trace(
                go.Scatter(x=df_temp.index, y=df_temp['Rolling_Return_3M'], mode='lines', name=f'{code}')
            )

            # 마지막 데이터 포인트에 어노테이션 추가
            fig_Rolling_Return.add_annotation(
                x=df_temp.index[-1],
                y=df_temp['Rolling_Return_3M'].iloc[-1],
                text=f'{code}',  # 항목 이름
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-30,
                font=dict(size=12, color="green"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="green",
                borderwidth=1
            )


            # 3개월 롤링 수익률 그래프 레이아웃 업데이트
            fig_Rolling_Return.update_layout(
                title=f'Average 3M Rolling Return (from {selected_year})',
                xaxis=dict(range=[df_temp.index.min(), df_temp.index.max()]),
                yaxis={'tickformat': ',.1%'},  # y축 포맷은 소수점 두 자리까지 백분율
                xaxis_title='Date',
                yaxis_title='3M Rolling Return',
                template='plotly_white'
            )








            fig_RV_3M.add_trace(
                go.Scatter(
                    x=[df_temp['Volatility'].iloc[-1]],  # x축에 도트 위치 지정
                    y=[df_temp['Rolling_Return_3M'].iloc[-1]],  # y축에 도트 위치 지정
                    mode='markers+text',  # 도트와 텍스트 모두 표시
                    name=f'{code}',  # 코드 이름
                    text=[f'{code}'],  # 어노테이션으로 표시할 텍스트
                    textposition='top center',  # 텍스트 위치 (도트 위쪽 중앙)
                    marker=dict(size=10)  # 도트 크기 지정
                )
            )

            fig_RV_3M.update_layout(
                title=f'Return / Vol (from {selected_year})',
                xaxis_title='Volatility',
                yaxis_title='Averaged 3M Return',
                yaxis={'tickformat': ',.0%'},  # y축 포맷은 소수점 두 자리까지 백분율
                xaxis={
                    'tickformat': ',.1%',  # x축 포맷도 백분율로 설정
                    'range': [0, 'auto'],
                },
                template='plotly_white'
            )


            fig_DD_1Y.add_trace(
                go.Scatter(
                    x=df_temp.index, 
                    y=df_temp['DD_1Y'], 
                    mode='lines', 
                    name=f'{code}'
                )
            )
            
            # DD_1Y
            fig_DD_1Y.update_layout(
                title=f'Drawdown(1Y from {selected_year})',
                xaxis_title='Date',
                yaxis_title='Drawdown',
                yaxis={'tickformat': ',.0%'},  # y축 포맷은 소수점 두 자리까지 백분율
                xaxis={},
                template='plotly_white'
            )

            print(df_price)

    return fig_cumulative, fig_trend, fig_Continue, fig_Digital_cum, fig_Continue_return, fig_Rolling_Return, fig_RV_3M, fig_DD_1Y



@app.callback(
    [Output('stock-dropdown-1', 'options'),
     Output('stock-dropdown-2', 'options'),
     Output('stock-dropdown-1', 'value'),
     Output('stock-dropdown-2', 'value')],
    [Input('code-group-dropdown', 'value')]
)
def update_comparison_dropdowns(selected_group):
    if selected_group and selected_group in code_dict:
        options = [{'label': code, 'value': code} for code in code_dict[selected_group]]
        # 첫 번째 값을 기본값으로 설정
        return options, options, options[0]['value'], options[1]['value']
    return [], [], None, None  # 옵션을 비워 반환하고 기본값을 None으로 설정


# 콜백 함수
@app.callback(
    [Output('cumulative-return-graph-comparison', 'figure'),
     Output('excess-return-bar', 'figure'),
     Output('corr-graph', 'figure'),
     ],
    [Input('stock-dropdown-1', 'value'),
     Input('stock-dropdown-2', 'value'),
     Input('period-dropdown', 'value')]
)
def update_graph(stock_1, stock_2, selected_period):
    start, end = get_period_range(selected_period)

    # 첫 번째 종목 데이터 가져오기
    df_1 = fetch_data(stock_1, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    if df_1 is None or df_1.empty:
        return {}, {}

    # 두 번째 종목 데이터 가져오기
    df_2 = fetch_data(stock_2, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    if df_2 is None or df_2.empty:
        return {}, {}

    # 일간 수익률 계산
    df_1['Daily_Return'] = df_1[stock_1].pct_change().fillna(0)
    df_2['Daily_Return'] = df_2[stock_2].pct_change().fillna(0)

    # 누적 수익률 계산
    df_1['cum_return'] = (1 + df_1['Daily_Return']).cumprod() - 1
    df_2['cum_return'] = (1 + df_2['Daily_Return']).cumprod() - 1

    # 초과 수익률 계산
    df_1['excess_return'] = df_1['cum_return'] - df_2['cum_return']


    # RSI 계산 함수 추가===========================================
    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


    
    df_1['RSI_14'] = calculate_rsi(df_1['cum_return'], window=14)
    df_1['Weight'] = df_1['RSI_14'].apply(lambda x: 0.5 if x <= 30 else (0.75 if x < 70 else 1))

    # 비중을 반영하여 누적 수익률(MP) 계산
    df_1['MP'] = (1 + df_1['Daily_Return'] * df_1['Weight'].fillna(0)).cumprod() - 1
    df_1['MP'] -= df_1['MP'].iloc[0]  # 첫날 수익률을 0으로 설정


    df_2['RSI_14'] = calculate_rsi(df_2['cum_return'], window=14)
    df_2['Weight'] = df_2['RSI_14'].apply(lambda x: 0.8 if x <= 30 else (0.9 if x < 70 else 1))

    # 비중을 반영하여 누적 수익률(MP) 계산
    df_2['MP'] = (1 + df_2['Daily_Return'] * df_2['Weight'].fillna(0)).cumprod() - 1
    df_2['MP'] -= df_2['MP'].iloc[0]  # 첫날 수익률을 0으로 설정



    # df_1['MP_EXR'] = df_1['MP']- df_2['MP']
    df_1['MP_EXR'] = df_1['MP']- df_2['cum_return']
    df_1['MP_EXR'] -= df_1['MP_EXR'].iloc[0]






    # 그래프 생성 - 누적 수익률
    fig_2cumulative = go.Figure()

    fig_2cumulative.add_trace(
        go.Scatter(
            x=df_1.index, y=df_1['cum_return'], mode='lines', name=f'{stock_1}')
    )

    fig_2cumulative.add_trace(
        go.Scatter(
            x=df_2.index, y=df_2['cum_return'], mode='lines', name=f'{stock_2}')
    )

    fig_2cumulative.update_layout(
        title=f'Cumulative Returns ({selected_period})',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        yaxis={'tickformat': ',.1%'},  # y축 포맷 설정
        xaxis=dict(range=[df_1.index.min(), df_1.index.max()]),
        template='plotly_white',
    )

    # 그래프 생성 - 초과 수익률
    fig_excess = go.Figure()
    fig_excess.add_trace(
        go.Bar(
            x=df_1.index, y=df_1['excess_return'], 
            name=f'Excess Return ({stock_1}-{stock_2})')
    )
    fig_excess.add_trace(
        go.Scatter(
            x=df_1.index, y=df_1['MP_EXR'], 
            name=f'MP Excess Return ({stock_1} RSI-{stock_2})')
    )
    
    fig_excess.update_layout(
        title=f'Excess Return ({selected_period})',
        xaxis_title='Date',
        yaxis_title='Excess Return',
        yaxis={'tickformat': ',.1%'},  # y축 포맷을 백분율로 설정
        xaxis=dict(range=[df_1.index.min(), df_1.index.max()]),
        template='plotly_white',
    )




    # 30일간 1주일 상관관계
    df_1['W_Return'] = df_1[stock_1].pct_change(7).fillna(0)
    df_2['W_Return'] = df_2[stock_2].pct_change(7).fillna(0)
    df_corr = df_1['W_Return'].rolling(window=30).corr(df_2['W_Return'])
    print("df_corr====================", df_corr)



    # 상관관계 그래프 생성
    fig_corr = go.Figure()
    fig_corr.add_trace(
        go.Scatter(x=df_corr.index, y=df_corr, mode='lines', name='Correlation'))
    
    fig_corr.update_layout(
        title=f'Correlation_Weekly(R)_window(30) ({selected_period})',
        xaxis_title='Date',
        yaxis_title='Excess Return',
        yaxis={'tickformat': ',.1f'},  # y축 포맷을 백분율로 설정
        xaxis=dict(range=[df_corr.index.min(), df_corr.index.max()]),
        template='plotly_white',
    )


    return fig_2cumulative, fig_excess, fig_corr




















# 레이아웃 정의
app.layout = html.Div(
    style={
        'margin': 'auto', 
        'width': '75%', 
        'height': '100vh', 
        'display': 'flex', 
        'flexDirection': 'column'},
    children=[
        html.H1(f"Covenant EMP Monitoring {datetime.today().strftime('%Y-%m-%d')}", style={'textAlign': 'center'}),
       

        # 딕셔너리 선택 Dropdown
        html.Label("Select Code Group"),
        dcc.Dropdown(
            id='code-group-dropdown',
            options=[{'label': name, 'value': name} for name in code_dict.keys()],
            value=list(code_dict.keys())[0],
            style={'width': '40%'}
        ),

        # ETF 이름 Dropdown
        html.Label("ETF(Stock) Name"),
        dcc.Dropdown(
            id='stock-dropdown',
            options=[{'label': code, 'value': code} for code in code],
            value='VUG',
            style={'width': '40%'},
        ),

        # 연도 Dropdown
        html.Label("Start Year"),
        dcc.Dropdown(
            id='year-dropdown',
            options=year_options,
            value=current_year-1,
            style={'width': '40%'},
        ),
        
        
        dcc.Graph(id='cumulative-return-graph', style={'width': '70%', 'margin' : 'auto'}),

        dcc.Graph(id='fig_Continue-graph', style={'width': '70%', 'margin' : 'auto'}),

        dcc.Graph(id='fig_trend-graph', style={'width': '70%', 'margin' : 'auto'}),

        dcc.Graph(id='Digital-cum-graph', style={'width': '70%', 'margin' : 'auto'}),
        
        dcc.Graph(id='Continue_return-graph', style={'width': '70%', 'margin' : 'auto'}),
        
        dcc.Graph(id='rolling-return-graph', style={'width': '70%', 'margin' : 'auto'}),

        dcc.Graph(id='Volatility-graph', style={'width': '70%', 'margin' : 'auto'}),

        dcc.Graph(id='DD_1Y-graph', style={'width': '70%', 'margin' : 'auto'}),











        # 두 종목 비교 섹션
        html.Label("Select First Stock"),
        dcc.Dropdown(
            id='stock-dropdown-1',
            options=[],  # 옵션을 빈 리스트로 설정
            value=None,
            style={'width': '40%'},
        ),
        html.Label("Select Second Stock"),
        dcc.Dropdown(
            id='stock-dropdown-2',
            options=[],  # 옵션을 빈 리스트로 설정
            value=None,
            style={'width': '40%'},
        ),
        html.Label("Select Period"),
        dcc.Dropdown(
            id='period-dropdown',
            options=period_options,
            value='1Y',
            style={'width': '40%'},
        ),
        dcc.Loading(
            id="loading-graph-2",
            type="default",
            children=html.Div([
                html.Div([
                    dcc.Graph(id='cumulative-return-graph-comparison', style={'flex': '1'}),
                    dcc.Graph(id='excess-return-bar', style={'flex': '1'}),
                ], style={'display': 'flex', 'flexDirection': 'row'}),
                
                dcc.Graph(id='corr-graph', style={'flex': '1', 'margin': 'auto'})
            ])
        ),

        html.H3(rf"MP Excess Return : RSI_14 30 </br> 이하 : Weight 50%  RSI_14 30~70 : Weight 75%   RSI_14 70 이상 :Weight 100%", style={'textAlign': 'right'}),
        









    ]
)




# 기본 포트 설정 ============================= 여러개 실행시 충돌 방지

DEFAULT_PORT = 8051

def find_available_port(start_port=DEFAULT_PORT, max_attempts=20):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))  # 실제 바인딩을 시도
                return port  # 사용 가능한 포트 반환
            except OSError:
                continue  # 이미 사용 중이면 다음 포트 확인
    raise RuntimeError("사용 가능한 포트를 찾을 수 없습니다.")

port = find_available_port()
print(f"사용 중인 포트: {port}")  # 디버깅용


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=port)

# ==================================================================


