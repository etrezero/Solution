import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.graph_objs as go
from dash import html, dcc, Input, Output, State
import concurrent.futures
import requests
import dash
import requests
import pickle
import os
from pykrx import stock as pykrx
from openpyxl import Workbook

import numpy as np
import pandas as pd

from flask import Flask
import socket

import dash_table



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
        df_price.index = pd.to_datetime(df_price.index)  # 인덱스를 문자열 형식으로 변환
        df_price = df_price.sort_index(ascending=True)
        
        return df_price
    
    except Exception as e:
        print(f"Error fetching data for {code}: {e}")
        return None


# 캐시 데이터 로딩 및 데이터 병합 처리 함수
def Func(code, start, end, batch_size=10):

    cache_price = r'C:\Covenant\cache\셀렉션_EMP.pkl'
    cache_expiry = timedelta(days=1)

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




#엑셀 저장=======================================================
def save_excel(df, sheetname, index_option=None):
    
    # 파일 경로
    path = rf'C:\Covenant\data\셀렉션_EMP.xlsx'

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





# 주식 코드 목록 설정
code_dict = {
    'code_Asset':
        ['ACWI', 'ACWX', 'BND', 'DIA', 'VUG', 'VTV', 'VEA', 'IDEV', 'VWO', 'MCHI', 'AVEM', 'EEM', 'IEMG', 'HYG', 'GLD', 'KRW=X', '356540.KS'],
   
    'code_US': 
        ['QQQ', 'SOXX', 'SPY', 'DIA', 'VUG', 'SPYG', 'IWF', 'VTV', 'MGV', 'SPYV', 'IWD', 'MAGS', 'IWM'],
  
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
        ['QQQ', 'PPA', 'SOXX', 'IWF', 'INDA', 'IEMG', 'SPY', 'VEA', 'VUG', 'VTI', 'IYF'],

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






# 오늘 날짜와 기간 설정
today = datetime.now()
start = today - timedelta(days=365*5)
end = today


# 데이터 가져오기==============================================
df_price = Func(code, start, end, batch_size=10)
df_price = df_price.ffill()


df_price.index = pd.to_datetime(df_price.index)  # 인덱스를 문자열 형식으로 변환
df_price = df_price.sort_index(ascending=True)
print("df_price===============", df_price)

# ==========================================================

# ✅ Dash는 DataFrame을 직접 전달할 수 없기 때문에, df_price를 JSON으로 변환하여 dcc.Store에 저장
df_price_json = df_price.to_json(date_format='iso', orient='split')  # JSON 변환
# print("df_price_json===============", df_price_json)








# RSI 계산 함수 추가===========================================
def calculate_rsi(price, window=14):
    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi




# Flask 서버 생성
server = Flask(__name__)
app = dash.Dash(__name__, server=server)
app.title = '모니터링_EMP'



# 연도 선택을 위한 Dropdown 옵션 생성
current_year = datetime.today().year
year_options = [{'label': str(year), 'value': year} for year in range(current_year - 10, current_year + 1)]

# 기간에 따른 데이터 범위 설정 함수
def select_period(period):
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
def update_stock_options(code_group):
    if code_group and code_group in code_dict:
        options = [{'label': code, 'value': code} for code in code_dict[code_group]]
        return options, options[0]['value'] if options else None  # 🔹 첫 번째 항목을 기본값으로 설정
    return [], None  # 선택된 그룹이 없을 경우 빈 리스트 반환



@app.callback(
    [Output('cumulative-return-graph', 'figure'),
     Output('fig_trend-graph', 'figure'),
     Output('Cum-MP-graph', 'figure'),
     Output('MP_RSI-graph', 'figure'),
     Output('MP-graph', 'figure'),

     Output('weight-table', 'columns'),  # ✅ 테이블 컬럼 추가
     Output('weight-table', 'data')],  # ✅ 테이블 데이터 추가    ],

    [Input('stock-dropdown', 'value'),
     Input('year-dropdown', 'value')],

    [State('df_price_store', 'data'),  # ✅ JSON 데이터
     State('code-group-dropdown', 'value')]  # ✅ 선택된 그룹 정보
)
def update_graph(selected_code, selected_year, df_price_json, selected_group):  
    # ✅ JSON을 DataFrame으로 변환
    df_price = pd.read_json(df_price_json, orient='split')
    df_price.index = pd.to_datetime(df_price.index, errors='coerce')


    # ✅ 날짜 필터링 (선택한 연도의 데이터만 유지)
    start_date = datetime(selected_year, 1, 1)
    end_date = datetime.today()
    
    #==========================================================================
    df_selected = df_price.loc[start_date:end_date, [selected_code]].dropna()
    #==========================================================================



    if df_selected.empty:
        print("❌ 오류: 필터링 후 데이터가 비어 있음")
        return {}, {}

    # ✅ 일간 수익률 계산
    df_selected['Daily_Return'] = df_selected[selected_code].pct_change().fillna(0)

    # ✅ 누적 수익률 계산
    df_selected['cum'] = (1 + df_selected['Daily_Return']).cumprod() - 1
    df_selected['cum'] -= df_selected['cum'].iloc[0]  # 첫날 수익률을 0으로 설정

    df_selected['RSI_14'] = calculate_rsi(df_selected[selected_code], window=14)

    # ✅ RSI 기반 가중치 설정
    df_selected['Weight'] = df_selected['RSI_14'].apply(lambda x: 0.5 if x <= 30 else (0.75 if x < 70 else 1))



    # ✅ 가중치를 반영한 MP 누적 수익률 계산
    df_selected['MP'] = (1 + df_selected['Daily_Return'] * df_selected['Weight']).cumprod() - 1
    df_selected['MP'] -= df_selected['MP'].iloc[0]

    # ✅ 변곡점 (최저점, 최고점) 찾기
    min_idx = df_selected['cum'].idxmin()
    max_idx = df_selected['cum'].idxmax()
    min_value = df_selected['cum'].loc[min_idx]
    max_value = df_selected['cum'].loc[max_idx]
    start_value = df_selected['cum'].iloc[0]
    end_value = df_selected['cum'].iloc[-1]

    # ✅ 변곡점 순서에 따른 구간 설정
    if min_idx < max_idx:
        segments = [
            (df_selected.index[0], min_idx, start_value, min_value),
            (min_idx, max_idx, min_value, max_value),
            (max_idx, df_selected.index[-1], max_value, end_value),
        ]
    else:
        segments = [
            (df_selected.index[0], max_idx, start_value, max_value),
            (max_idx, min_idx, max_value, min_value),
            (min_idx, df_selected.index[-1], min_value, end_value),
        ]

    # ✅ 베이스라인 생성
    baseline = np.full_like(df_selected['cum'].values, np.nan, dtype=np.float64)
    for start_seg, end_seg, start_val, end_val in segments:
        start_idx = df_selected.index.get_loc(start_seg)
        end_idx = df_selected.index.get_loc(end_seg) + 1
        slope = (end_val - start_val) / (end_idx - start_idx - 1)
        baseline[start_idx:end_idx] = start_val + slope * np.arange(end_idx - start_idx)

    # ✅ 차이 계산
    diff = df_selected['cum'].values - baseline
    positive_area = np.where(diff > 0, diff, 0)
    negative_area = np.where(diff < 0, diff, 0)

    # ✅ 누적 수익률 그래프 생성
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(x=df_selected.index, y=df_selected['cum'], mode='lines', name=selected_code))
    fig_cumulative.add_trace(go.Scatter(x=df_selected.index, y=df_selected['MP'], mode='lines', name=f'{selected_code} MP RSI(14day)'))
    fig_cumulative.add_trace(go.Scatter(x=df_selected.index, y=df_selected['RSI_14'], mode='lines', name=f'{selected_code} RSI(14-day)', yaxis='y2'))

    fig_cumulative.update_layout(
        title=f'{selected_code} Return & RSI ({selected_year})',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2=dict(title='RSI', overlaying='y', side='right', range=[0, 300]),
        yaxis={'tickformat': ',.1%'},
        template='plotly_white',
    )

    # ✅ 트렌드 그래프 생성
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=df_selected.index, y=df_selected['cum'], mode='lines', name='Cumulative Return'))
    fig_trend.add_trace(go.Scatter(x=df_selected.index, y=baseline, mode='lines', name='Baseline', line=dict(dash='dash')))
    fig_trend.add_trace(go.Scatter(x=df_selected.index, y=positive_area, mode='lines', fill='tozeroy', name='Positive Area', line=dict(color='green', width=0)))
    fig_trend.add_trace(go.Scatter(x=df_selected.index, y=negative_area, mode='lines', fill='tozeroy', name='Negative Area', line=dict(color='red', width=0)))

    fig_trend.update_layout(
        title=f'{selected_code} Return with Baselines',
        xaxis_title='Date',
        yaxis_title='Return',
        yaxis=dict(tickformat=',.0%'),
        template='plotly_white'
    )





    # ✅ Group=====================================================================
    
    selected_group_list = code_dict.get(selected_group, [])
    print("selected_group_list==============", selected_group_list)
    df_price_group = pd.DataFrame(index=df_price.index, columns=selected_group_list)

    #=========================================================================
    df_price_group = df_price.loc[start_date:end_date, selected_group_list].dropna()
    #==========================================================================
    

    df_R_group = df_price_group.pct_change().fillna(0)
    cum_group = (1 + df_R_group.fillna(0)).cumprod() - 1
    cum_group -= cum_group.iloc[0]


    # ✅ RSI 및 가중치 계산
    df_rsi = calculate_rsi(df_price_group, window=14)
    df_Weight = df_rsi.map(lambda x: 1 if x <= 30 else (1 if x < 70 else 0.5))

    cum_RSI = (1 + df_R_group * df_Weight.shift(1).fillna(0)).cumprod() - 1
    cum_RSI -= cum_RSI.iloc[0]
    print("cum_RSI==============", cum_RSI)



    # ✅ 그래프 생성
    fig_cum_RSI = go.Figure()

    for col in df_price_group.columns:
        fig_cum_RSI.add_trace(
            go.Scatter(
                x=cum_RSI.index, 
                y=cum_RSI[col], 
                name=f'{col}'
            )
        )


    # ✅ 레이아웃 설정
    fig_cum_RSI.update_layout(
        title='cum_RSI_Return',
        xaxis_title='Date',
        yaxis_title='cum_RSI_Return',
        yaxis=dict(tickformat=',.1%'),
        xaxis=dict(range=[cum_RSI.index.min(), cum_RSI.index.max()]),
        template='plotly_white',
    )







    # ✅ 수익률 데이터 생성
    group_avg = pd.DataFrame(
        (1 + df_R_group.fillna(0).mean(axis=1)).cumprod() - 1, columns=['group_avg'])
    group_avg -= group_avg.iloc[0]

    port_RSI = pd.DataFrame(
        (1 + (df_R_group * df_Weight.shift(1).fillna(0)).mean(axis=1)).cumprod() - 1, columns=['port_RSI'])
    port_RSI -= port_RSI.iloc[0]

    # ✅ 인덱스를 기준으로 합치기 (inner join: 공통된 인덱스만 유지)
    merged_df = pd.concat([group_avg, port_RSI], axis=1, join='inner')
    merged_df = merged_df.rolling(window=60).mean().dropna()

    # ✅ 인덱스 타입이 올바른지 확인 후 변환
    if not isinstance(merged_df.index, pd.DatetimeIndex):
        merged_df.index = pd.to_datetime(merged_df.index, errors='coerce')

    print("✅ merged_df 생성 성공:\n", merged_df.head())



    # ✅ 그래프 생성
    fig_merged_df = go.Figure()
    fig_merged_df.add_trace(
        go.Scatter(x=merged_df.index, y=merged_df['group_avg'], mode='lines', name='group_avg'))
    fig_merged_df.add_trace(
        go.Scatter(x=merged_df.index, y=merged_df['port_RSI'], mode='lines', name='Portfolio RSI', line=dict(dash='dash')))

    fig_merged_df.update_layout(
        title=f'{selected_group} Portfolio & RSI',
        xaxis_title='Date',
        yaxis_title='Return',
        yaxis=dict(tickformat=',.0%'),
        template='plotly_white'
    )





# ✅ 투자비중 테이블 생성  : 14일 이동 평균 수익률 계산 ======================================================
    Avg_R_2W = df_price_group.pct_change().rolling(window=14).mean().dropna()
    Avg_R_2W_normalize = Avg_R_2W.abs().div(Avg_R_2W.abs().sum(axis=1), axis=0).fillna(0)
    
    # ✅ 개별 자산별 14일 이동 평균 투자 비중 계산
    avg_weight = Avg_R_2W_normalize.rolling(window=14).mean()
    avg_weight = avg_weight.dropna(how ='all', axis = 0)
    

    print("avg_weight==============", avg_weight)


    RSI_weight = df_Weight*avg_weight
    # save_excel(RSI_weight, 'RSI_weight', index_option=True)
    
    MP = (1+(df_R_group*RSI_weight.shift(1).fillna(0)).sum(axis=1)).cumprod()-1
    MP -= MP.iloc[0]
    MP.columns = ['MP']

    BM = (1+df_R_group.mean(axis=1).fillna(0)).cumprod()-1
    BM -= BM.iloc[0]
    BM.columns = ['BM']

    
    print("MP*************", MP)
    print("BM*************", BM)

    save_excel(MP, 'MP', index_option=True)


    # ✅ 그래프 생성
    fig_MP = go.Figure()
    fig_MP.add_trace(
        go.Scatter(x=MP.index, y=MP, mode='lines', name='MP'))
    fig_MP.add_trace(
        go.Scatter(x=BM.index, y=BM, mode='lines', name='BM', line=dict(dash='dash')))

    fig_MP.update_layout(
        title=f'{selected_group} average vs MP',
        xaxis_title='Date',
        yaxis_title='Return',
        yaxis=dict(tickformat=',.0%'),
        template='plotly_white'
    )






    RSI_weight_last = (df_Weight*avg_weight).iloc[-1]
    RSI_weight_last = pd.DataFrame({'Asset': RSI_weight_last.index, 'Weight': RSI_weight_last.values})

    # ✅ 테이블의 컬럼 설정
    table_columns = [{"name": col, "id": col} for col in RSI_weight_last.columns]


    # ✅ 숫자인 경우에만 % 포맷 적용, 문자열이면 그대로 유지
    table_data = RSI_weight.map(lambda x: f"{float(x):.1%}" if isinstance(x, (int, float)) and not np.isnan(x) else x).to_dict('records')




    # *********************************************************************************************
    return fig_cumulative, fig_trend, fig_cum_RSI, fig_merged_df, fig_MP, table_columns, table_data
    # *********************************************************************************************





@app.callback(
    [Output('stock-dropdown-1', 'options'),
     Output('stock-dropdown-2', 'options'),
     Output('stock-dropdown-1', 'value'),
     Output('stock-dropdown-2', 'value')],
    [Input('code-group-dropdown', 'value')]
)
def update_comparison_dropdowns(code_group):
    if code_group and code_group in code_dict:
        options = [{'label': code, 'value': code} for code in code_dict[code_group]]
        # 첫 번째 값을 기본값으로 설정
        return options, options, options[0]['value'], options[1]['value']
    return [], [], None, None  # 옵션을 비워 반환하고 기본값을 None으로 설정


# 콜백 함수
@app.callback(
    [Output('cumulative-return-graph-comparison', 'figure'),
     Output('excess-return-bar', 'figure'),],
    [Input('stock-dropdown-1', 'value'),
     Input('stock-dropdown-2', 'value'),
     Input('period-dropdown', 'value')]
)
def update_graph(stock_1, stock_2, selected_period):
    start, end = select_period(selected_period)

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

    fig_2cumulative.add_trace(
            go.Bar(
            x=df_1.index, y=df_1['Weight'], 
            name=f'{stock_1} RSI Weight', 
            yaxis='y2',)
    )



    fig_2cumulative.update_layout(
        title=f'Cumulative Returns ({selected_period})',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        yaxis={'tickformat': ',.1%'},  # y축 포맷 설정
        xaxis=dict(range=[df_1.index.min(), df_1.index.max()]),
        template='plotly_white',

        # ✅ 두 번째 y축 추가 (Weight 막대 그래프용)
        yaxis2=dict(
        title='Weight',
        overlaying='y',
        side='right',
        range=[0, 5]  # ✅ Weight 값 범위 0~1
    ),
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

    return fig_2cumulative, fig_excess










# 레이아웃 정의
app.layout = html.Div(
    style={
        'margin': 'auto', 
        'width': '75%', 
        'height': '100vh', 
        'display': 'flex', 
        'flexDirection': 'column'},

    children=[
        html.H1(f"Covenant EMP Selection {datetime.today().strftime('%Y-%m-%d')}", style={'textAlign': 'center'}),


        dcc.Store(id='df_price_store', data=df_price_json),  # ✅ df_price 저장


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
        
        dcc.Graph(id='fig_trend-graph', style={'width': '70%', 'margin' : 'auto'}),

        dcc.Graph(id='Cum-MP-graph', style={'width': '70%', 'margin' : 'auto'}),

        dcc.Graph(id='MP_RSI-graph', style={'width': '70%', 'margin' : 'auto'}),

        dcc.Graph(id='MP-graph', style={'width': '70%', 'margin' : 'auto'}),



        # 두 종목 비교 섹션====================================================
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
                dcc.Graph(id='cumulative-return-graph-comparison', style={'flex': '1'}),
                dcc.Graph(id='excess-return-bar', style={'flex': '1'})
            ], style={'display': 'flex', 'flexDirection': 'row'})
        ),

        html.H3(f"MP Excess Return : ", style={'textAlign': 'right'}),
        html.H5(f"RSI_14 30 이하 : Weight 50%", style={'textAlign': 'right'}),
        html.H5(f"RSI_14 30~70 : Weight 75%", style={'textAlign': 'right'}),
        html.H5(f"RSI_14 70 이상 :Weight 100%", style={'textAlign': 'right'}),
       # ==========================================================================




        html.Div([
            html.H4("Last RSI Weights"),
            dash_table.DataTable(
                id='weight-table',
                columns=[],  # ✅ 초기에는 빈 리스트
                data=[],  # ✅ 초기 데이터 없음
                style_table={'overflowX': 'auto'},  # 가로 스크롤 허용
                style_cell={'textAlign': 'center', 'padding': '10px'},  # 텍스트 중앙 정렬 및 패딩 추가
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},  # 헤더 스타일
            )
        ], style={'width': '50%', 'margin': 'auto'}),



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
    app.run_server(debug=True, host='0.0.0.0', port=port)

# ==================================================================


