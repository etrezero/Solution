# 표준 라이브러리
import os
import pickle
import socket
from datetime import datetime, timedelta
import concurrent.futures

# 외부 라이브러리
import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from openpyxl import Workbook

# 데이터 소스
import yfinance as yf
from pykrx import stock as pykrx
import FinanceDataReader as fdr

# Dash & Plotly
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go

# Flask 서버
from flask import Flask
import socket





cache_price = r'C:\Covenant\cache\Big7_상관관계.pkl'
cache_expiry = timedelta(days=1)



code = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META']



# 데이터 가져오기 함수
def fetch_data(code, start, end):
    try:
        if isinstance(code, int) or code.isdigit():
            if len(code) == 5:
                code = '0' + code
            df_price = pykrx.get_market_ohlcv_by_date(start, end, code)['종가']
        else:
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
start = today - timedelta(days=365*5)
end = today


df_price = Func(code, start, end, batch_size=10)
df_price = df_price.ffill()
print("df_price===============", df_price)




df_R = df_price.pct_change(periods=1).fillna(0)
df_3M_R = df_price.pct_change(periods=60).fillna(0)


df_mean_R = df_R.mean(axis=1)



df_cum = (1+df_R).cumprod()-1
df_cum_mean = (1+df_mean_R).cumprod()-1


# 상관관계 계산
df_corr_matrix = df_3M_R.corr()

# 상관관계 시계열 계산 함수
def calculate_rolling_correlation(df, window=60):
    corr_dict = {}
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:
                corr_key = f"{col1} vs {col2}"
                corr_dict[corr_key] = df[col1].rolling(window).corr(df[col2])
    return pd.DataFrame(corr_dict)

df_corr = calculate_rolling_correlation(df_3M_R).dropna()

# 모든 쌍의 상관관계 평균 계산
df_corr['average'] = df_corr.mean(axis=1)
df_corr_average = df_corr['average']


#문자형식으로 소수점 4자리로 변환 후 테이블/그래프에 전달
df_corr_matrix = df_corr_matrix.applymap(lambda x: f"{x:.4f}")
df_corr = df_corr.applymap(lambda x: f"{x:.4f}")



print('df_corr_matrix========', df_corr_matrix)
print('df_corr===============', df_corr)
print('df_corr_average=======', df_corr_average)




# Flask 서버 생성
server = Flask(__name__)
app = dash.Dash(__name__, server=server)
app.title = 'Big7_상관관계'



# 대시 앱 레이아웃 설정
app.layout = html.Div(
    style={'width': '65%', 'margin': 'auto'},
    children=[
        dcc.Graph(
            id='Big7 Correlation',
             figure={
                'data': [
                    go.Scatter(
                        x=df_corr.index, 
                        y=df_corr['average'].astype(float), 
                        mode='lines', 
                        name='Average Corr',
                        yaxis='y1',
                        line=dict(color='#3762AF', width=2),
                    ),
                    go.Scatter(
                        x=df_cum_mean.index, 
                        y=df_cum_mean, 
                        mode='lines', 
                        name='Cumulative Return(Average)',
                        yaxis='y2',
                        line=dict(color='#630', width=2),
                    ),
                ], 
                'layout': {
                    'title': 'Big7 Correlation and Return',
                    'xaxis': {
                        'title': 'Date', 
                        'tickformat': '%Y-%m-%d',
                        'tickmode': 'auto', 
                        'nticks': 10,
                        'textangle' : 10,
                    },
                    'yaxis': {
                        'title': 'Correlation', 
                        'tickformat': '.1f',
                        'side': 'left',
                    },
                    'yaxis2': {
                        'title': 'Average Return',
                        'overlaying': 'y',
                        'side': 'right',
                        'tickformat': '.0%',
                    },
                }
            }
        ),



        html.H3("Correlation Matrix 3M"),
        dash_table.DataTable(
            id='corr-matrix-table',
            columns=[{"name": "index", "id": "index"}] + [{"name": i, "id": i} for i in df_corr_matrix.columns],
            data=df_corr_matrix.reset_index().to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'padding': '5px'},
            style_header={
                'backgroundColor': 'white',
                'fontWeight': 'bold'
            },
        ),


        html.Hr(),  # 수평선 추가
        html.H3("Correlation 3M"),
        dash_table.DataTable(
            id='corr-time-series-table',
            columns=[{"name": i, "id": i} for i in df_corr.columns],
            data=df_corr.reset_index().to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'padding': '5px'},
            style_header={
                'backgroundColor': 'white',
                'fontWeight': 'bold'
            },
            page_size=10,
          
        ),






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


