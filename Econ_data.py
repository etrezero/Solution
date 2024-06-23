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
import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
from scipy import stats
from scipy.optimize import minimize
import openpyxl




# FRED API 키 설정 (FRED API 키를 등록해야 합니다)
fred_api_key = '9260edd9940ff3b32a0d991334330fcc'
fred = Fred(api_key=fred_api_key)


path1 = r'D:\data\Econ_FRED.pkl'

start0 = (datetime.now() - relativedelta(years=20)).strftime('%Y-%m-%d')
end0 = datetime.now().strftime('%Y-%m-%d')


# # 경제 지표 가져오기===============================

symbol = ['GDP', 'UNRATE', 'CPIAUCSL']

def get_econ_FRED(indicators, start0):
    econ_data = {}
    for indicator in tqdm(indicators, desc="Fetching data"):
        data = fred.get_series(indicator, observation_start=start0)
        econ_data[indicator] = data
    return pd.DataFrame(econ_data)

econ_df = get_econ_FRED(symbol, start0)









#함수로 Long-Term Price 데이터 불러오기 실행 ===============================
econ_df = get_econ_FRED(symbol, start0)

if os.path.exists(path1):
    os.remove(path1)
econ_df.to_pickle(path1)
print(f"econ_df를 {path1}에 저장했습니다.")




#===========================================================================




# # 저장된 마지막 날짜부터 다운받아서 업데이트(병합) 해서 저장하기
# econ_df = pd.read_pickle(path1, orient='columns')
# econ_df = econ_df.fillna(0)

# econ_df.index = pd.to_datetime(econ_df.index)  # 명시적으로 인덱스를 datetime 형식으로 변환
# start1 = econ_df.index[-1] + pd.Timedelta(days=1)
# print(start1)


# update = get_data(symbol, start1, end0)
# econ_df = pd.concat([econ_df, update])
# econ_df.to_pickle(path1)


# 파일로 저장된 Price 데이터로 작업하기
econ_df = pd.read_pickle(path1)
econ_df.index = pd.to_datetime(econ_df.index)

# 인덱스 중복 제거
econ_df = econ_df[~econ_df.index.duplicated(keep='first')]

# 고유한 인덱스를 가진 DataFrame을 pickle 파일로 저장합니다.
econ_df.to_pickle(path1)
print(f"econ_df를 {path1}에 저장했습니다.")

num_columns = econ_df.shape[1]
print(f"Number of columns : {num_columns}")
print(econ_df.head(), econ_df.tail())
#===============================================================





# # Dash 앱 초기화
# app = dash.Dash(__name__)

# # 대시 앱 레이아웃 설정
# app.layout = html.Div(
#     style={'width': '60%', 'margin': 'auto'},
#     children=[
#         html.H3("ETF Selection", style={'textAlign': 'center'}),
       

# dcc.Graph(
#             id='line1',
#             figure={
#                 'data': [
#                     go.Scatter(
#                         x=df_cum.index,
#                         y=df_cum[column],
#                         mode='lines',
#                         name=column
#                     ) for column in df_cum.columns
#                 ],
#                 'layout': {
#                     'title': f'ETF Cumulative Return : {num_columns}종목',
#                     'xaxis': {'title': 'Date'},
#                     'yaxis': {'title': 'Return', 'tickformat': '.0%'},
#                 }
#             },
#             style={'width': '100%', 'margin': 'auto'},
#         ),



#         dcc.Graph(
#             id='line2',
#             figure={
#                   'data': [
#                         go.Scatter(x=port_top10.index, y=port_top10, mode='lines', name='Test1'),
#                         go.Scatter(x=top10_MA.index, y=top10_MA, mode='lines', name='Test1 MA'),
#                         go.Scatter(x=port_mid30.index, y=port_mid30, mode='lines', name='Test2'),
#                         go.Scatter(x=mid30_MA.index, y=mid30_MA, mode='lines', name='Test2 MA'),
                  
#                         go.Scatter(x=Covenant_port.index, y=Covenant_port, mode='lines', name='Covenant'),
#                     ],
#                 'layout': {
#                     'title': 'Covenant 포트폴리오',
#                     'xaxis': {'title': 'Date'},
#                     'yaxis': {'title': 'Return', 'tickformat': '.0%'},
#                 }
#             },
#             style={'width': '100%', 'margin': 'auto'},
#         ),





#         dcc.Graph(
#             id='line4',
#             figure={
#                   'data': [
                        
#                         go.Bar(x=MP1.index, y=MP1, name='Test1 1M'),
#                         go.Bar(x=MP2.index, y=MP2, name='Test2 1M'),
#                     ],
#                 'layout': {
#                     'title': 'Test1 Rolling 1M',
#                     'xaxis': {'title': 'Date'},
#                     'yaxis': {'title': 'Return', 'tickformat': '.0%'},
#                 }
#             },
#             style={'width': '100%', 'margin': 'auto'},
#         ),




#     ]
# )


# if __name__ == '__main__':
#     app.run_server(debug=False, host='0.0.0.0')
