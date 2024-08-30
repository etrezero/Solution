import requests
import yaml
import json
import numpy as np
import certifi
import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import dash
from dash import dcc, html
import plotly.graph_objs as go
import concurrent.futures
from pykrx import stock as pykrx
from scipy.optimize import minimize


# config.yaml 파일의 경로 설정
config_path = r'C:\Covenant\config.yaml'

# YAML 파일을 읽어 설정값 가져오기
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# API 설정값 로드
# URL_BASE = "https://openapi.koreainvestment.com:9443"

#모의투자
URL_BASE = "https://openapivts.koreainvestment.com:29443"


APP_KEY = config['true_api']['APP_KEY']
APP_SECRET = config['true_api']['APP_SECRET']
CANO = config['true_api']['CANO']
ACNT_PRDT_CD = config['true_api']['ACNT_PRDT_CD']

def get_access_token():
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
    }
    url = f"{URL_BASE}/oauth2/tokenP"
    response = requests.post(url, headers=headers, data=json.dumps(body), verify=certifi.where())
    return response.json()["access_token"]

verify=False

# 현재 포트폴리오를 조회하는 함수
def get_current_portfolio(token):
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
    }
    url = f"{URL_BASE}/uapi/domestic-stock/v1/trading/inquire-balance"
    params = {
        "CANO": CANO,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "N",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "00",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    response = requests.get(url, headers=headers, params=params)
    return response.json()

# 매수 주문 함수
def send_buy_order(token, ticker, price, qty):
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "VTTC0802U",  # 국내 주식 매수 요청 ID
    }
    data = {
        "CANO": CANO,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "PDNO": ticker,  # 종목코드
        "ORD_DVSN": "00",  # 00: 지정가, 01: 시장가
        "ORD_QTY": str(qty),  # 주문 수량
        "ORD_UNPR": str(price),  # 주문 가격
    }
    url = f"{URL_BASE}/uapi/domestic-stock/v1/trading/order-cash"
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

# 매도 주문 함수
def send_sell_order(token, ticker, price, qty):
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "VTTC0801U",  # 국내 주식 매도 요청 ID
    }
    data = {
        "CANO": CANO,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "PDNO": ticker,  # 종목코드
        "ORD_DVSN": "00",  # 00: 지정가, 01: 시장가
        "ORD_QTY": str(qty),  # 주문 수량
        "ORD_UNPR": str(price),  # 주문 가격
    }
    url = f"{URL_BASE}/uapi/domestic-stock/v1/trading/order-cash"
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

# 포트폴리오 리밸런싱 함수
def rebalance_portfolio(cum_port, optimal_df, token):
    current_portfolio = get_current_portfolio(token)
    
    for idx, row in optimal_df.iterrows():
        ticker = row['종목명']
        target_weight = row['투자비중']
        
        # 현재 종목이 포트폴리오에 있으면 비중을 조정, 없으면 매수
        if ticker in current_portfolio:
            current_qty = current_portfolio[ticker]['qty']
            current_price = current_portfolio[ticker]['price']
            current_value = current_qty * current_price
            total_value = sum([p['qty'] * p['price'] for p in current_portfolio.values()])
            current_weight = current_value / total_value
            
            if current_weight < target_weight:
                qty_to_buy = int(((target_weight - current_weight) * total_value) / current_price)
                send_buy_order(token, ticker, current_price, qty_to_buy)
            elif current_weight > target_weight:
                qty_to_sell = int(((current_weight - target_weight) * total_value) / current_price)
                send_sell_order(token, ticker, current_price, qty_to_sell)
        else:
            # 종목이 포트폴리오에 없으면 매수
            current_price = fdr.DataReader(ticker, start=datetime.today().strftime('%Y-%m-%d'), end=datetime.today().strftime('%Y-%m-%d'))['Close'].iloc[0]
            qty_to_buy = int((target_weight * sum([p['qty'] * p['price'] for p in current_portfolio.values()])) / current_price)
            send_buy_order(token, ticker, current_price, qty_to_buy)
    
    # 포트폴리오에 있지만 optimal_df에 없는 종목은 매도
    for ticker in current_portfolio:
        if ticker not in optimal_df['종목명'].values:
            send_sell_order(token, ticker, current_portfolio[ticker]['price'], current_portfolio[ticker]['qty'])

# 데이터를 가져오는 함수
def fetch_data(code, start, end):
    try:
        if isinstance(code, int) or code.isdigit() or code.endswith(".KS"):
            if isinstance(code, int):
                code = str(code)
            if code.endswith(".KS"):
                code = code.replace(".KS", "")
            ETF_price = pykrx.get_market_ohlcv_by_date(start, end, code)
            if '종가' in ETF_price.columns:
                ETF_price = ETF_price['종가'].rename(code)
            else:
                raise ValueError(f"{code}: '종가' column not found in pykrx data.")
        else:
            ETF_price = fdr.DataReader(code, start, end)['Close'].rename(code)
        return ETF_price
    except Exception as e:
        print(f"Error fetching data for {code}: {e}")
        return None

def FDR(code, start, end):
    data_frames = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_data, c, start, end): c for c in code}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                data_frames.append(result)
    return pd.concat(data_frames, axis=1) if data_frames else pd.DataFrame()



path_list = r'C:\Covenant\data\List_KRX_ETF.xlsx'

#==========================================================

list_df = pd.read_excel(path_list)
code_dict = dict(zip(list_df['종목코드'], list_df['종목명']))
code = list(code_dict.keys())


# 시작 날짜를 1년 전으로 설정
start = (datetime.today() - relativedelta(years=1)).strftime('%Y-%m-%d')
end = (datetime.today() - timedelta(days=0)).strftime('%Y-%m-%d')

#=========================================================
# ETF_price = FDR(code, start, end)

#  파일로 저장
# Path_price = r'C:\Covenant\data\ETF_KR_price.pkl'
# ETF_price.to_pickle(Path_price)
#=========================================================



# pkl 파일을 읽어서 ETF_price 데이터프레임으로 지정
Path_price = r'C:\Covenant\data\ETF_KR_price.pkl'
ETF_price = pd.read_pickle(Path_price)

#======================================================


# code_dict를 데이터프레임으로 변환
code_df = pd.DataFrame(list(code_dict.items()), columns=['종목코드', '종목명'])
code_df['종목코드'] = code_df['종목코드'].astype(str)

# 병렬로 열 이름 수정 작업 수행
def rename_column(col):
    if col in code_df['종목코드'].values:
        종목명 = code_df.loc[code_df['종목코드'] == col, '종목명'].values[0]
        return f"{col} + {종목명}"
    else:
        return col

with concurrent.futures.ThreadPoolExecutor() as executor:
    ETF_price.columns = list(executor.map(rename_column, ETF_price.columns))

# 필터링된 데이터프레임 생성
ETF_price = ETF_price[
    ETF_price.columns[
        ~ETF_price.columns.str.contains('레버리지|2X|3X|인버스|crypto|bitcoin', case=False)
    ]
]

ETF_price = ETF_price.bfill().fillna(0)
ETF_R = ETF_price.pct_change(periods=1)
df_cum = (1 + ETF_R).cumprod() - 1
df_cum.replace([float('inf'), float('-inf')], 0, inplace=True)
df_cum.iloc[0] = 0

RR_3M = ETF_price.pct_change(periods=60)
RR_3M.replace([float('inf'), float('-inf')], 0, inplace=True)
rank = RR_3M.rank(axis=1, pct=True)

count_win = (rank <= 0.4).sum(axis=0)
count_all = rank.count(axis=0)
win_prob = count_win / count_all
win_prob = win_prob.to_frame(name='top rank prob')

win_prob = win_prob.sort_values(by='top rank prob', ascending=True)
prob_top = win_prob.iloc[0:50]
prob_top = prob_top.index.tolist()

cum_50 = df_cum[prob_top]

# 변동성 높은 종목 필터링
vol_50 = cum_50.pct_change(periods=60).std() * np.sqrt(4)
vol_low = vol_50.sort_values(ascending=True)
vol_low = vol_low.head(int(len(vol_low) * 0.5))
vol_low = vol_low.index.tolist()

cum_50 = cum_50[vol_low]

# 마지막 숫자 0인 열 드롭
cols_drop = cum_50.columns[cum_50.iloc[-1] == 0]
cum_50 = cum_50.drop(columns=cols_drop)

def calculate_MDD(price):
    roll_max = price.cummax()
    drawdown = (price - roll_max) / roll_max
    max_drawdown = drawdown.min()
    return max_drawdown

MDD = calculate_MDD(cum_50)
cut_MDD = MDD.quantile(0.25)
cols_X = MDD[MDD <= cut_MDD].index

cum_50 = cum_50.loc[:, ~cum_50.columns.isin(cols_X)]

mean_cum_50 = cum_50.iloc[-1].mean()

def portfolio_mdd(weights, price_data):
    R_port = np.dot(price_data.pct_change().dropna(), weights)
    cumulative_returns = np.cumprod(1 + R_port) - 1
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    drawdown = np.nan_to_num(drawdown, nan=0.0)
    max_drawdown = drawdown.min()
    return abs(max_drawdown)

n_assets = cum_50.shape[1]
initial_weights = np.ones(n_assets) / n_assets
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((0, 1) for _ in range(n_assets))

result = minimize(portfolio_mdd, initial_weights, args=(cum_50,), method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x

optimal_df = pd.DataFrame({
    '종목명': cum_50.columns,
    '투자비중': optimal_weights
})

R_50 = ETF_R[cum_50.columns]
R_50.replace([float('inf'), float('-inf')], 0, inplace=True)
R_50.loc[R_50.index[0], :] = 0

weights = optimal_df.set_index('종목명').loc[cum_50.columns]['투자비중'].values

port_R = R_50.mul(weights, axis=1)
port_R = port_R.sum(axis=1)

cum_port = (1 + port_R).cumprod() - 1
cum_port.iloc[0] = 0



# 대시 앱 생성
app = dash.Dash(__name__)
app.title = 'Selection_KR_ETF'

app.layout = html.Div(
    style={'width': '60%', 'margin': 'auto'},
    children=[
        dcc.Graph(
            id='win prob',
            figure={
                'data': [
                    go.Scatter(
                        x=win_prob.index,
                        y=win_prob[column],
                        mode='lines',
                        name=column,
                        text=win_prob.index,
                        hoverinfo='text+y'
                    ) for column in win_prob.columns
                ],
                'layout': {
                    'title': 'win prob',
                    'xaxis': {'title': 'Ticker'},
                    'yaxis': {'title': 'Return YTD', 'tickformat': '.0%'},
                }
            },
        ),
        dcc.Graph(
            id='ETF Return',
            figure={
                'data': [
                    go.Scatter(
                        x=cum_50.index,
                        y=cum_50[column],
                        mode='lines',
                        name=column,
                        text=column,
                        hoverinfo='text'
                    ) for column in cum_50.columns
                ],
                'layout': {
                    'title': 'Selected Cum_ETF',
                    'xaxis': {'title': 'Ticker'},
                    'yaxis': {'title': 'Return YTD', 'tickformat': '.0%'},
                }
            },
        ),
        dcc.Graph(
            id='Portfolio Return',
            figure={
                'data': [
                    go.Scatter(
                        x=cum_port.index,
                        y=cum_port,
                        mode='lines',
                        hoverinfo='x+y',
                        name='Portfolio Return'
                    )
                ] + [
                    go.Scatter(
                        x=df_cum.index,
                        y=df_cum[col],
                        mode='lines',
                        hoverinfo='x+y',
                        name=f'{col} Return',
                        line=dict(dash='dash')
                    ) for col in df_cum.columns if '069500' in col
                ],
                'layout': {
                    'title': 'Selected Portfolio vs SPY Returns',
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': 'Return YTD', 'tickformat': '.0%'},
                }
            },
        ),
    ]
)

if __name__ == '__main__':
    token = get_access_token()
    rebalance_portfolio(cum_port, optimal_df, token)
    app.run_server(debug=False, host='0.0.0.0')
