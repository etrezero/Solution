import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash import dash_table
import concurrent.futures
import os
import pickle
from pykrx import stock as pykrx
import yfinance as yf
import ssl
import requests

import numpy as np
from scipy.optimize import minimize
import time
from urllib.error import HTTPError  # HTTPError 예외 처리용 모듈 임포트
from urllib3.exceptions import InsecureRequestWarning
import urllib3



# SSL 인증 경고 무시
urllib3.disable_warnings(InsecureRequestWarning)

import chardet

#캐시 - 코드 - 날짜 순으로 작업


# 캐싱 path
cache_path = r'C:\Covenant\data\ETF_US_price.pkl'

cache_US_ETF = r'C:\Covenant\data\ETF_US_price.pkl'
cache_KR_ETF = r'C:\Covenant\data\ETF_KR_price.pkl'
cache_KR_종목 = r'C:\Covenant\data\종목_KR_price.pkl'
cache_expiry = timedelta(days=1)


# 리스트 path
path_list_US_ETF = r'C:\Covenant\data\ETF_US_List.csv'
path_list_KR_ETF = r'C:\Covenant\data\List_KRX_ETF.xlsx'
path_list_KR_종목 = r'C:\Covenant\data\List_KRX_종목.xlsx'


# CSV 파일의 인코딩을 감지하는 함수==========================
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

encoding_US_ETF = detect_encoding(path_list_US_ETF)
encoding_KR_ETF = detect_encoding(path_list_KR_ETF)
encoding_KR_종목 = detect_encoding(path_list_KR_종목)

print("encoding_US_ETF", encoding_US_ETF)
print("encoding_KR_ETF", encoding_KR_ETF)
print("encoding_KR_종목", encoding_KR_종목)
# =======================================================


# ETF 코드와 이름을 매핑하는 딕셔너리 생성
df_list_US_ETF = pd.read_csv(path_list_US_ETF, encoding=encoding_US_ETF)
code_dict_US_ETF = dict(zip(df_list_US_ETF['종목코드'], df_list_US_ETF['종목명']))

df_list_KR_ETF = pd.read_excel(path_list_KR_ETF)
code_dict_KR_ETF = dict(zip(df_list_KR_ETF['종목코드'], df_list_KR_ETF['종목명']))

df_list_KR_종목 = pd.read_excel(path_list_KR_종목)
code_dict_KR_종목 = dict(zip(df_list_KR_종목['종목코드'], df_list_KR_종목['종목명']))


code_dict_추가 = {
    "ACWI": "MSCI ACWI",
    "SPY": "S&P 500 Index",
    "IEUR": "MSCI EUROPE Index",
    "EWJ": "MSCI JAPAN",
    "EEM": "MSCI EM",
    "MCHI": "MSCI China",
    "EWY": "MSCI KOREA",
    "GSG": "S&P GSCI Commodity Index TR",
    "GLD": "Gold",
    "IGF": "S&P Global Infrastructure Index",
    "VNQ": "Dow Jones US REIT Index",
    "PFF": "S&P U.S. Preferred Stock Index",
    "XLB": "Materials Select Sector Index",
    "XLV": "Health Care Select Sector Index",
    "XLP": "Consumer Staples Select Sector Index",
    "XLY": "Consumer Discretionary Select Sector Index",
    "XLE": "Energy Select Sector Index",
    "XLF": "Financial Select Sector Index",
    "XLI": "Industrial Select Sector Index",
    "XLK": "Technology Select Sector Index",
    "XLU": "Utilities Select Sector Index",
    "XLRE": "Real Estate Select Sector Index",
    "XLC": "Communication Services Select Sector Index",
    "IWF": "Russell 1000 Growth Index",
    "IWD": "Russell 1000 Value Index",
    "USMV": "MSCI USA Minimum Volatility Index",
    "VYM": "FTSE High Dividend Yield Index",
    "MTUM": "MSCI USA Momentum Index",
    "QUAL": "MSCI USA Quality Index",
    "VLUE": "MSCI USA Value Index",
    "TIP": "Bloomberg Barclays U.S. Treasury Inflation Protected Securities (TIPS) Index",
    "TLT": "ICE U.S. Treasury 20+ Year Bond Index",
    "AGG": "Bloomberg Barclays U.S. Aggregate Bond Index",
    "SPAB": "Bloomberg Barclays U.S. Aggregate Bond Index",
    "LQD": "Markit iBoxx USD Liquid Investment Grade Index",
    "HYG": "Markit iBoxx USD Liquid High Yield Index",
    "EMB": "J.P. Morgan EMBI Global Core Index",
    "BND": "Bloomberg Barclays U.S. Aggregate Bond Index",

    "QQQ": "Nasdaq-100 Index",
    "SOXX": "PHLX Semiconductor Sector Index",
    "VUG": "CRSP US Large Cap Growth Index",
    "SPYG": "S&P 500 Growth Index",
    "VTV": "CRSP US Large Cap Value Index",
    "SPYV": "S&P 500 Value Index",
    "VEA": "FTSE Developed All Cap ex US Index",
    "VWO": "FTSE Emerging Markets All Cap China A Inclusion Index",
    "SCHE": "FTSE Emerging Markets All Cap China A Inclusion Index",
    "IAUM": "LBMA Gold Price Index",
    "272910": "ACE 중장기국공채",
    "273130": "KODEX 종합채권(AA-이상)",
    "356540": "ACE 종합채권(AA-이상)",
    "365780": "ACE 국고채 10년",
    "385540": "RISE 종합채권(A-이상)",
    "USD/KRW": "원/달러 환율",
}


# code 딕셔너리 병합
code_dict = {}
code_dict.update(code_dict_US_ETF)
code_dict.update(code_dict_KR_ETF)
code_dict.update(code_dict_KR_종목)
code_dict.update(code_dict_추가)

code = list(set(code_dict.keys()))


# code_dict_A와 code_dict_B 병합
# code_dict = {**code_dict_A, **code_dict_B}


# 데이터 가져오기 함수
def fetch_data(code, start, end):
    try:
        if isinstance(code, int) or code.isdigit() or code.endswith(".KS"):
            if isinstance(code, int):
                code = str(code)
            if len(code) == 5:
                code = '0' + code
            if code.endswith(".KS"):
                code = code.replace(".KS", "")
            df_price = pykrx.get_market_ohlcv_by_date(start, end, code)
            if '종가' in df_price.columns:
                df_price = df_price['종가'].rename(code)
            else:
                raise ValueError(f"{code}: '종가' column not found in pykrx data.")
        else:
            df_price = fdr.DataReader(code, start, end)['Close'].rename(code)
        return df_price
    except Exception as e:
        print(f"Error fetching data for {code}: {e}")
        return None

# FDR 함수 수정 - 캐시 파일을 하나로 통합
def FDR(code, start, end, batch_size=10):
    # 캐시가 존재하고 유효 기간 내라면 캐시에서 데이터 로드
    if os.path.exists(cache_path):
        cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - cache_mtime < cache_expiry:
            with open(cache_path, 'rb') as f:
                print("Loading data from cache...")
                return pickle.load(f)

    # 캐시가 없거나 만료되었다면 데이터를 다시 불러오고 캐시 저장
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

    # 데이터를 캐시에 저장
    with open(cache_path, 'wb') as f:
        pickle.dump(price_data, f)
        print("Data cached.")

    return price_data

# 시작 날짜 설정
start = (datetime.today() - relativedelta(years=1)).strftime('%Y-%m-%d')
end = (datetime.today() - timedelta(days=0)).strftime('%Y-%m-%d')

# 데이터 불러오기 및 출력
df_price = FDR(code, start, end, 30)
df_price = df_price.ffill()



#======================================================

# code_dict를 데이터프레임으로 변환
code_df = pd.DataFrame(list(code_dict.items()), columns=['종목코드', '종목명'])
code_df['종목코드'] = code_df['종목코드'].astype(str)

# 병렬로 열 이름 수정 작업 수행=======================================
def rename_column(col):
    if col in code_df['종목코드'].values:
        종목명 = code_df.loc[code_df['종목코드'] == col, '종목명'].values[0]
        return f"{col} + {종목명}"
    else:
        return col

with concurrent.futures.ThreadPoolExecutor() as executor:
    df_price.columns = list(executor.map(rename_column, df_price.columns))
#==============================================================================

print("df_price===============", df_price)






# 필터링된 데이터프레임 생성
df_price = df_price[
    df_price.columns[
        ~df_price.columns.str.contains('레버리지|2X|3X|인버스|crypto|bitcoin|미국|TRF|TDF|글로벌|MSCI|방산|인도|200|100|채권|회사채|국공채', case=False)
    ]
]



df_price = df_price.bfill().fillna(0)
ETF_R = df_price.pct_change(periods=1)

print(ETF_R.tail())
df_cum = (1 + ETF_R).cumprod() - 1
df_cum.replace([float('inf'), float('-inf')], 0, inplace=True)
df_cum.iloc[0] = 0


RR_3M = df_price.pct_change(periods=20)
RR_3M.replace([float('inf'), float('-inf')], 0, inplace=True)
rank = RR_3M.rank(axis=1, pct=True)


count_win = (rank <= 0.4).sum(axis=0)    #숫자 낮을수록 모멘텀 방지
count_all = rank.count(axis=0)
win_prob = count_win / count_all
win_prob = win_prob.to_frame(name='top rank prob')

win_prob = win_prob.sort_values(by='top rank prob', ascending=True)
prob_top = win_prob.iloc[0:50]
prob_top = prob_top.index.tolist()

#===================================
cum_prob = df_cum[prob_top]
#===================================



top_50_rank = rank.iloc[-1].sort_values(ascending=True).head(50)
rank_top = top_50_rank.index.tolist()
#===========================================================
cum_rank = df_cum[rank_top]
#===========================================================



# 대시 앱 생성
app = dash.Dash(__name__)
app.title = 'Selection_US_ETF'

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
                    'title': 'Win Probability',
                    'xaxis': {'title': 'Ticker'},
                    'yaxis': {'title': 'Top Rank Probability', 'tickformat': '.0%'},
                }
            },
        ),
       

        dcc.Graph(
            id='ETF Return',
            figure={
                'data': [
                    go.Scatter(
                        x=df_cum.index,
                        y=df_cum[column],
                        mode='lines',
                        name=column,
                        text=column,
                        hoverinfo='text'
                    ) for column in df_cum.columns if 'QQQ' in column 
                ],
                'layout': {
                    'title': 'Selected Cum_ETF',
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': 'Return YTD', 'tickformat': '.0%'},
                }
            },
        ),
        

])

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')