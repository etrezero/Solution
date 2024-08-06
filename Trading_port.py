import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
from tqdm import tqdm
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
from scipy import stats
from scipy.optimize import minimize
import openpyxl
import concurrent.futures

# 추가된 Dash 임포트 (중복 제거 및 필요한 모듈 임포트)
import dash
from dash import dcc, html
import plotly.graph_objs as go

# # Symbol dict 만들기 =================================
# # 시장 리스트
# market_list = [
#     'ETF/KR',
#     'ETF/US',
#     'S&P500',
#     'KOSPI',
#     'NYSE',
#     'NASDAQ',
#     'KOSDAQ',
#     'SSE',
#     'SZSE',
#     'HKEX',
#     'TSE',
#     'HOSE',
# ]

# # 코드 불러오기
# def fetch_market_symbols(market):
#     try:
#         symbols = fdr.StockListing(market)
#         return symbols
#     except Exception as e:
#         print(f"{market} 목록을 가져오는 중 오류 발생: {e}")
#         return pd.DataFrame()  # 빈 데이터프레임 반환

# def get_symbol(market_list):
#     all_symbols = pd.DataFrame()
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future_to_market = {executor.submit(fetch_market_symbols, market): market for market in market_list}
#         for future in concurrent.futures.as_completed(future_to_market):
#             market = future_to_market[future]
#             try:
#                 symbols = future.result()
#                 all_symbols = pd.concat([all_symbols, symbols], ignore_index=True)
#             except Exception as e:
#                 print(f"{market} 데이터를 병합하는 중 오류 발생: {e}")
#     return all_symbols

# # 심볼 가져오기 및 'USD/KRW' 추가
# symbols = get_symbol(market_list)
# symbols = pd.concat([symbols, pd.DataFrame({'Symbol': ['USD/KRW'], 'Name': ['USD/KRW']})], ignore_index=True)

# # 심볼과 이름의 데이터 프레임 생성
# df_code_dict = pd.DataFrame({'code': symbols['Symbol'], 'Name': symbols['Name']})
# print(df_code_dict.head())

# # 데이터 프레임을 JSON 파일로 저장
# path_code_dict = r'D:\data\code_dict.json'
# df_code_dict.to_json(path_code_dict, orient='table', index=False)
# #====================================================


# JSON 파일 읽어오기
path_code_dict = r'D:\data\code_dict.json'
df_code_dict = pd.read_json(path_code_dict, orient='table')
df_code_dict = pd.DataFrame(df_code_dict)

print("df_code_dict", df_code_dict)


# DataFrame을 딕셔너리로 변환
code_dict = df_code_dict.set_index('code').to_dict()['Name']

# '005930' 키의 값 확인
if '005930' in code_dict:
    samsung_name = code_dict['005930']
    print("005930 값:", samsung_name)
else:
    print("'005930' is not present in code_dict")




# 병렬로 데이터 가져오기 함수
def fetch_data(code):
    try:
        df = fdr.DataReader(code, start, end)['Close']
        df.index = pd.to_datetime(df.index, errors='coerce')  # 잘못된 날짜를 NaT로 변환
        df = df.dropna()  # NaT를 가진 행을 제거
        return code, df
    except Exception as e:
        print(f"Error fetching data for {code}: {e}")
        return code, None

# 데이터프레임 병합 함수
def get_data(codes):
    data = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_data, code): code for code in codes}
        with tqdm(total=len(futures), desc="Fetching data") as pbar:
            for future in concurrent.futures.as_completed(futures):
                code, df = future.result()
                if df is not None:
                    data[code] = df
                pbar.update(100)
    return pd.DataFrame(data)



code_list = [
    'ACWI', 'URTH', 
    'VUG', 'SPYG', 'IWF',
    'VTV', 'SPYV', 'IWD',
    'VEA', 'VWO', 'IAUM',
    'SOXX', 'MAGS',
    '005930', '356540',
    'NVDA',
]


start = '2024-01-01'
end = datetime.today().strftime('%Y%m%d')
df_price = get_data(code_list)


# NaN 값을 포함하는 열의 이름 출력
nan_columns = df_price.columns[df_price.isna().any()].tolist()
print("NaN 값을 포함하는 열:", nan_columns)

df_price = df_price.ffill().bfill()
print('df_price===============', df_price)

# 일간 수익률 계산
df_R = df_price.pct_change().dropna()
print('df_R===============', df_R)

df_cum = (1+df_R).cumprod()-1
print('df_cum===============', df_cum)


# 열 이름을 'Name'으로 대체===========================
def replace_name(df, code_dict):
    new_columns = []
    for col in df.columns:
        if col.isdigit() or col.startswith('0'):  # 열 이름이 숫자이거나 0으로 시작하는 경우
            new_name = code_dict.loc[code_dict['code'] == col, 'Name'].values
            if len(new_name) > 0:
                new_columns.append(new_name[0])
            else:
                new_columns.append(col)  # 매칭되는 이름이 없을 경우 원래 이름 사용
        else:
            new_columns.append(col)  # 조건에 맞지 않는 경우 원래 이름 사용
    df.columns = new_columns
    return df

df_cum = replace_name(df_cum, df_code_dict)
print('df_cum with replaced column names===============', df_cum)
#======================================================




# df_cum에서 각 윈도우의 수익률을 계산합니다.
# 각 윈도우 수익률과 df_cum의 다음 10일 윈도우 수익률 간의 상관관계를 계산합니다.
# 상관관계가 가장 높은 윈도우를 순서대로 출력합니다.

# 윈도우 수익률 계산
window_returns = {}
for window in range(5, 21):
    window_returns[window] = df_cum.pct_change(periods=window).dropna()

# 다음 10일 윈도우 수익률 계산
df_next_10 = df_cum.pct_change(periods=10).shift(-10).dropna()

# 각 윈도우 수익률과 다음 10일 윈도우 수익률 간의 상관관계 계산
correlations = {}
for window, df_window in window_returns.items():
    common_index = df_window.index.intersection(df_next_10.index)
    correlation = df_window.loc[common_index].corrwith(df_next_10.loc[common_index], axis=0)
    correlations[window] = correlation.mean()  # 평균 상관관계를 사용

# 상관관계가 가장 높은 윈도우 순서대로 정렬
sorted_windows = sorted(correlations.items(), key=lambda item: item[1], reverse=True)

# 출력
for window, correlation in sorted_windows:
    print(f"Window: {window}, Correlation: {correlation:.4f}")
    print(window_returns[window].head(), "\n")



# 모든 윈도우의 평균 상관관계 출력
for window, correlation in correlations.items():
    print(f"Window: {window}, Average Correlation: {correlation:.4f}")










# Dash 앱 초기화
app = dash.Dash(__name__)

# 레이아웃 설정
app.layout = html.Div([
    dcc.Graph(
        id='line-graph',
        figure={
            'data': [
                go.Scatter(
                    x=df_cum.index,
                    y=df_cum[col],
                    mode='lines',
                    name=col
                ) for col in df_cum.columns
            ],
            'layout': go.Layout(
                title='Cumulative Returns',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Value'},
                hovermode='closest'
            )
        }
    )
])



if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')


# http://192.168.219.101:8050