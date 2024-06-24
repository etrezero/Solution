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



path_price = r'D:\data\price.pkl'

# 시장 리스트
list = [
    'ETF/KR',
    # 'ETF/US',
    # 'S&P500',
    # 'KOSPI',
    # 'NYSE',
    # 'NASDAQ'
    # 'KOSDAQ',
    # 'SSE',
    # 'SZSE',
    # 'HKEX',
    # 'TSE',
    # 'HOSE',
]

# 코드 불러오기
def get_symbol(list):
    all_symbol = []
    for market in list:
        try:
            symbols = fdr.StockListing(market)
            all_symbol.extend(symbols['Symbol'].tolist())
        except Exception as e:
            print(f"Error fetching list for {market}: {e}")
    return all_symbol

symbol = get_symbol(list) + ['USD/KRW']


# 시작 및 종료 날짜 설정
start0 = (datetime.today() - relativedelta(years=1)).strftime('%Y-%m-%d')
end0 = datetime.today().strftime('%Y-%m-%d')

# 병렬로 데이터 가져오기 함수
def fetch_data(symbol):
    try:
        df = fdr.DataReader(symbol, start0, end0)['Close']
        df.index = pd.to_datetime(df.index, errors='coerce')  # 잘못된 날짜를 NaT로 변환
        df = df.dropna()  # NaT를 가진 행을 제거
        return symbol, df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return symbol, None

# 데이터프레임 병합 함수
def get_data(symbols):
    data = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_data, symbol): symbol for symbol in symbols}
        with tqdm(total=len(futures), desc="Fetching data") as pbar:
            for future in concurrent.futures.as_completed(futures):
                symbol, df = future.result()
                if df is not None:
                    data[symbol] = df
                pbar.update(100)
    return pd.DataFrame(data)




# 데이터 불러오기====================================
# # df_price = get_data(symbol)


# # 고유한 인덱스를 가진 DataFrame을 Pickle 파일로 저장
# df_price.index = pd.to_datetime(df_price.index)
# df_price = df_price[~df_price.index.duplicated(keep='first')]

# df_price.to_pickle(path_price)
# print(f"df_price를 {path_price}에 저장했습니다.")

# print(df_price)
#=================================================









# 파일로 저장된 Price 데이터로 작업하기========================================================
df_price = pd.read_pickle(path_price)


num_columns = df_price.shape[1]
print(f"Number of columns: {num_columns}")
print(df_price.head())


# 데이터 전처리
df_R = df_price.pct_change(fill_method=None)
df_R = df_R.where(df_R.abs() <= 0.5, 0)
print('df_R=======================', df_R.head())


df_cum = (1 + df_R).cumprod() - 1

# 월간 수익률 계산
df_m = df_price.pct_change(periods=30).dropna()
print('df_m', df_m)

# 월간 수익률 기준으로 정렬
cum_last_m = df_m.sum().sort_values(ascending=False)

# 상위 10% 종목의 일간 수익률
top_line = int(len(cum_last_m) * 0.1)
df_top10 = df_price[cum_last_m[:top_line].index]
port_top10 = df_top10.pct_change(periods=1, fill_method=None).mean(axis=1)
cum_top10 = (1+port_top10).cumprod()-1
# 상위 10% ~ 30% 종목의 일간 수익률
mid_line = int(len(cum_last_m) * 0.3)
df_mid30 = df_price[cum_last_m[top_line:mid_line].index]
port_mid30 = df_mid30.pct_change(periods=1, fill_method=None).mean(axis=1)
cum_mid30 = (1+port_mid30).cumprod()-1

# MP1 and MP2
MP1 = port_top10 
MP2 = port_mid30

# 최적화 함수 (샤프 비율 최대화)
def portfolio_performance(weights, MP1, MP2, risk_free_rate=0.02):
    W_MP1, W_MP2 = weights
    portfolio_return = W_MP1 * MP1 + W_MP2 * MP2
    mean_return = portfolio_return.mean()
    risk = portfolio_return.std()
    sharpe_ratio = (mean_return - risk_free_rate) / risk
    return sharpe_ratio

# 최적화 문제 정의 (샤프 비율 최대화를 위해 음수를 최소화)
def neg_sharpe_ratio(weights, MP1, MP2, risk_free_rate=0.04):
    return -portfolio_performance(weights, MP1, MP2, risk_free_rate)

# 제약 조건 및 초기 가중치 설정
constraints = {'type': 'eq', 'fun': lambda weights: weights[0] + weights[1] - 1}
bounds = [(0, 1), (0, 1)]
initial_weights = [0.5, 0.5]

# 매달 최적의 W_MP1, W_MP2 찾기
monthly_dates = MP1.resample('ME').last().index  # 매월 마지막 날로 재샘플링
results = []

for date in monthly_dates:
    if date not in MP1.index or date not in MP2.index:
        continue
    if pd.isnull(MP1.loc[date]) or pd.isnull(MP2.loc[date]):
        continue
    res = minimize(neg_sharpe_ratio, initial_weights, args=(MP1.loc[:date], MP2.loc[:date]),
                   method='SLSQP', bounds=bounds, constraints=constraints)
    results.append((date, res.x[0], res.x[1]))

Covenant_weight = pd.DataFrame(results, columns=['Date', 'W_MP1', 'W_MP2'])
Covenant_weight.set_index('Date', inplace=True)

# 월중에는 전월말 업데이트된 가중치를 유지
Covenant_weight = Covenant_weight.reindex(MP1.index, method='ffill')

print(Covenant_weight)


# 최적의 투자비중을 사용하여 포트폴리오 생성
Covenant_port = (1+Covenant_weight['W_MP1'] * MP1 + Covenant_weight['W_MP2'] * MP2).cumprod()-1
Covenant_port_1M = Covenant_port.diff(30)


# 엑셀 파일 경로
path_Covenant_weight = r'D:\data\Covenant_weight.xlsx'

if not os.path.exists(path_Covenant_weight):
    with pd.ExcelWriter(path_Covenant_weight, engine='openpyxl') as writer:
        pd.DataFrame().to_excel(writer, sheet_name='Sheet')

# 엑셀 시트에 데이터 저장
with pd.ExcelWriter(path_Covenant_weight, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    Covenant_weight.to_excel(writer, sheet_name='covenant_Weight', index=True)
    print('엑셀저장 완료')


# Dash 앱 초기화
app = Dash(__name__)

# 대시 앱 레이아웃 설정
app.layout = html.Div(
    style={'width': '60%', 'margin': 'auto'},
    children=[
        html.H3("ETF Selection", style={'textAlign': 'center'}),
        dcc.Graph(
            id='line1',
            figure={
                'data': [
                    go.Scatter(
                        x=df_cum.index,
                        y=df_cum[column],
                        mode='lines',
                        name=column
                    ) for column in df_cum.columns
                ],
                'layout': {
                    'title': f'ETF Cumulative Return : {num_columns} 종목',
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': 'Return', 'tickformat': '.0%'},
                }
            },
            style={'width': '100%', 'margin': 'auto'},
        ),
        dcc.Graph(
            id='line2',
            figure={
                'data': [
                    go.Scatter(x=cum_top10.index, y=cum_top10, mode='lines', name='Test1'),
                    go.Scatter(x=cum_mid30.index, y=cum_mid30, mode='lines', name='Test2'),
                    go.Scatter(x=Covenant_port.index, y=Covenant_port, mode='lines', name='Covenant'),
                ],
                'layout': {
                    'title': 'Covenant 포트폴리오',
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': 'Return', 'tickformat': '.0%'},
                }
            },
            style={'width': '100%', 'margin': 'auto'},
        ),
        dcc.Graph(
            id='line3',
            figure={
                'data': [
                    go.Bar(x=Covenant_port_1M.index, y=Covenant_port_1M, name='Covenant Rolling 1M'),
                ],
                'layout': {
                    'title': 'Covenant Rolling 1M',
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': 'Return', 'tickformat': '.0%'},
                }
            },
            style={'width': '100%', 'margin': 'auto'},
        ),
        dcc.Graph(
            id='line4',
            figure={
                'data': [
                    go.Bar(x=MP1.index, y=MP1, name='Test1 1M'),
                    go.Bar(x=MP2.index, y=MP2, name='Test2 1M'),
                ],
                'layout': {
                    'title': 'Test1 Rolling 1M',
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': 'Return', 'tickformat': '.0%'},
                }
            },
            style={'width': '100%', 'margin': 'auto'},
        ),
    ]
)

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')