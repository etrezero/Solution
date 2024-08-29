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



path_List = r'C:\Covenant\data\ETF_US_List.csv'



#==========================================================
list_df = pd.read_csv(path_List, encoding='euc-kr')
code_dict = dict(zip(list_df['종목코드'], list_df['종목명']))

code = list_df['종목코드'].tolist()

# # '종목코드'을 키로 하고, 'Name'과 'Asset Class'를 값으로 가지는 딕셔너리 생성
# code_dict = dict(zip(list_df['종목코드'], zip(list_df['Name'], list_df['Asset Class'])))
# print(code_dict)


# # code = {**DM, **EM}



def fetch_data(code, start, end):
    try:
        # 숫자형 코드인지 확인
        if isinstance(code, int) or code.isdigit() or code.endswith(".KS"):
            if isinstance(code, int):
                code = str(code)
            # 5자리 숫자이면 앞에 0을 붙여서 6자리 문자열로 변환
            if len(code) == 5:
                code = '0' + code
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


def FDR(code, start, end, batch_size=10):
    data_frames = []
    
    # 'code' 리스트를 batch_size 크기로 나누어 처리
    for i in range(0, len(code), batch_size):
        code_batch = code[i:i + batch_size]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(fetch_data, c, start, end): c for c in code_batch}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    data_frames.append(result)
    
    # 모든 배치 처리 후 데이터프레임으로 결합
    return pd.concat(data_frames, axis=1) if data_frames else pd.DataFrame()

# 시작 날짜를 1년 전으로 설정
start = (datetime.today() - relativedelta(years=1)).strftime('%Y-%m-%d')
end = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

# code 리스트에 들어있는 항목을 10개씩 나누어 데이터를 요청하고 합침


#======================================================
batch_size = 20
# ETF_price = FDR(code, start, end, batch_size=batch_size)
#======================================================

# # # # 엑셀 저장=================================================
Path_price_xl = r'C:\Covenant\data\Selection_US_ETF.xlsx'
# with pd.ExcelWriter(Path_price_xl, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
#     ETF_price.to_excel(writer, sheet_name='ETF_price', index=True, merge_cells=False)
# # ==================================================================




# pickle 파일로 저장
Path_price_pkl = r'C:\Covenant\data\ETF_US_price.pkl'
# ETF_price.to_pickle(Path_price_pkl)

# # pkl 파일을 읽어서 ETF_price 데이터프레임으로 지정
ETF_price = pd.read_pickle(Path_price_pkl)
# # ===============================================================







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
    ETF_price.columns = list(executor.map(rename_column, ETF_price.columns))
#==============================================================================




# 필터링된 데이터프레임 생성
ETF_price = ETF_price[
    ETF_price.columns[
        ~ETF_price.columns.str.contains('leverage|2X|3X|ultra|crypto|bitcoin', case=False)
    ]
]


ETF_price = ETF_price.bfill().fillna(0)
ETF_R = ETF_price.pct_change(periods=1)

print(ETF_R.tail())
df_cum = (1 + ETF_R).cumprod() - 1
df_cum.replace([float('inf'), float('-inf')], 0, inplace=True)
df_cum.iloc[0] = 0

RR_3M = ETF_price.pct_change(periods=60)
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
cum_50 = df_cum[prob_top]
#===================================

# 변동성 높은 종목 필터링  
vol_50 = cum_50.pct_change(periods=60).std()* np.sqrt(4)
vol_low = vol_50.sort_values(ascending=True)
vol_low = vol_low.head(int(len(vol_low) * 0.5))   #숫자 낮을수록 엄격
vol_low = vol_low.index.tolist()

#===================================
cum_50 = cum_50[vol_low]
#===================================



# 마지막 숫자 0인 열 드롭
cols_drop = cum_50.columns[cum_50.iloc[-1] == 0]
cum_50 = cum_50.drop(columns=cols_drop)



#==============================================================
def calculate_MDD(price):
    """
    시계열 데이터에서 Max Drawdown을 계산하는 함수.
    """
    roll_max = price.cummax()
    drawdown = (price - roll_max) / roll_max
    max_drawdown = drawdown.min()
    return max_drawdown


# # 'FNGO' 문자열을 포함하는 열 필터링
# fngo_columns = [col for col in cum_50.columns if 'FNGO' in col]

# # 각 'FNGO' 열에 대해 MDD 계산
# mdd_results = {col: calculate_MDD(cum_50[col]) for col in fngo_columns}

# # 결과 출력
# for col, mdd in mdd_results.items():
#     print(f"Max Drawdown for {col}: {mdd:.2%}")



# 각 열에 대해 Max Drawdown 계산
MDD = calculate_MDD(cum_50)
print("MDD=======", MDD)

# Max Drawdown이 하위 25%에 해당하는 열들을 찾기
cut_MDD = MDD.quantile(0.25)
print("cut_MDD==============", cut_MDD)
cols_X = MDD[MDD <= cut_MDD].index

# =================================================
cum_50 = cum_50.loc[:, ~cum_50.columns.isin(cols_X)]
print("cum_50=========", cum_50)
#==================================================



mean_cum_50 = cum_50.iloc[-1].mean()
print("mean_cum_50", mean_cum_50)






# 옵티마이제이션에 반영할 포트폴리오의 MDD를 계산하는 함수
def portfolio_mdd(weights, price_data):
    # 포트폴리오의 일일 수익률 계산
    R_port = np.dot(price_data.pct_change().dropna(), weights)
    # 포트폴리오의 누적 수익률 계산
    cumulative_returns = np.cumprod(1 + R_port) - 1
    # 포트폴리오의 Max Drawdown 계산
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    drawdown = np.nan_to_num(drawdown, nan=0.0)
    max_drawdown = drawdown.min()
    return abs(max_drawdown)  # 최소화 문제이므로 절대값으로 반환

# 초기 가중치 설정 (균등 가중치로 시작)
n_assets = cum_50.shape[1]
initial_weights = np.ones(n_assets) / n_assets

# 제약 조건: 투자 비중의 합이 1이어야 함
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# 가중치가 0과 1 사이의 값을 가지도록 설정
bounds = tuple((0, 1) for _ in range(n_assets))

# 최적화 수행
result = minimize(portfolio_mdd, initial_weights, args=(cum_50,), 
                  method='SLSQP', bounds=bounds, constraints=constraints)

# 최적의 투자 비중
optimal_weights = result.x

# cum_50 열 이름과 최적 투자 비중을 데이터프레임으로 생성
optimal_df = pd.DataFrame({
    '종목명': cum_50.columns,
    '투자비중': optimal_weights
})


# cum_50의 일간 수익률 계산
R_50 = ETF_R[cum_50.columns]
R_50.replace([float('inf'), float('-inf')], 0, inplace=True)
R_50.loc[R_50.index[0], :] = 0

# optimal_df의 투자비중을 가져와서 cum_50의 열 순서에 맞게 정렬
weights = optimal_df.set_index('종목명').loc[cum_50.columns]['투자비중'].values
print("Weight===============", weights)


# 각 자산의 일간 수익률에 해당 자산의 최적 투자 비중을 곱한 데이터프레임 생성
port_R = R_50.mul(weights, axis=1)
R_50.loc[R_50.index[0], :] = 0
port_R = port_R.sum(axis=1)
print("port_R===============", port_R)


cum_port = (1+port_R).cumprod() - 1
cum_port.iloc[0] = 0
print("cum_port=============", cum_port)





# # # # 엑셀 저장=================================================
path = r'C:\Covenant\data\Selection_US_ETF.xlsx'

# with pd.ExcelWriter(path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
#     R_50.to_excel(writer, sheet_name='R_50', index=True, merge_cells=False)
#     port_R.to_excel(writer, sheet_name='port_R', index=True, merge_cells=False)
#     cum_port.to_excel(writer, sheet_name='cum_port', index=True, merge_cells=False)
# ==================================================================



# # 포트폴리오 누적 수익률 계산
# cum_port = (1 + R_port).cumprod() - 1
# cum_port.iloc[0] = 0

# print("cum_port=============", cum_port)

# # 포트폴리오 수익률을 데이터프레임으로 변환
# portfolio_df = pd.DataFrame({
#     '날짜': cum_50.index[1:],  # 첫 번째 수익률은 NaN이므로 제외
#     '포트폴리오 누적 수익률': cum_port
# }).set_index('날짜')











# #=============================================================
# def calculate_mean_cum_50(cut_MDD, ETF_price):
#     # 전처리: 결측값 처리 및 누적 수익률 계산
#     ETF_price = ETF_price.ffill().fillna(0)
#     ETF_R = ETF_price.pct_change(periods=1)
#     df_cum = (1 + ETF_R).cumprod() - 1
#     df_cum.replace([float('inf'), float('-inf')], 0, inplace=True)

#     # 3개월 롤링 수익률 및 랭크 계산
#     RR_3M = ETF_price.pct_change(periods=60)
#     rank = RR_3M.rank(axis=1, pct=True)

#     # 입력된 cut_MDD를 기준으로 상위 n% 계산
#     count_win = (rank <= cut_MDD).sum(axis=0)
#     count_all = rank.count(axis=0)
#     win_prob = count_win / count_all
#     win_prob = win_prob.to_frame(name='top rank prob')

#     # 상위 50개 종목을 선택하여 누적 수익률 계산
#     win_prob = win_prob.sort_values(by='top rank prob', ascending=True)
#     prob_top = win_prob.head(50)
#     prob_top = prob_top.index.tolist()
#     cum_50 = df_cum[prob_top]

#     # 마지막 날의 평균 누적 수익률 계산
#     mean_cum_50 = cum_50.iloc[-1].mean()
    
#     return mean_cum_50


# # 함수 사용 예시
# cut_MDD_value = 0.25  # 임의의 cut_MDD 값 입력
# mean_cum_50_result = calculate_mean_cum_50(cut_MDD_value, ETF_price)
# print("mean_cum_50 for cut_MDD", cut_MDD_value, ":", mean_cum_50_result)
# #=====================================================================













# 대시 앱 생성
app = dash.Dash(__name__)
app.title = 'ETF_data'

# 대시 앱 레이아웃 설정
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
                        text=win_prob.index,  # 호버 텍스트로 전체 열 이름 설정
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
                        text=column,  # 호버 텍스트로 전체 열 이름 설정
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
                        name='Portfolio Return'  # 포트폴리오 수익률 라인의 이름
                    )
                ] + [
                    go.Scatter(
                        x=df_cum.index,
                        y=df_cum[col],
                        mode='lines',
                        hoverinfo='x+y',
                        name=f'{col} Return',  # SPY 관련 수익률 라인의 이름
                        line=dict(dash='dash')  # SPY 라인을 점선으로 표시
                    ) for col in df_cum.columns if 'SPY' in col
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
    app.run_server(debug=False, host='0.0.0.0')
