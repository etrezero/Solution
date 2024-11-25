import pymysql
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, dash_table
from datetime import datetime, timedelta
from dash.dash_table.Format import Format, Scheme
from dateutil.relativedelta import relativedelta
import json
import os
from openpyxl import Workbook
import plotly.graph_objs as go
import plotly.express as px
import re
import numpy as np

import yfinance as yf
from pykrx import stock as pykrx
import concurrent.futures
import requests
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter



import sys
sys.path.append(r'C:\Covenant')
from get_hostname_from_ip import get_hostname_from_ip


IP_name = get_hostname_from_ip("192.168.194.161")
print(f"Hostname: {IP_name}")




# JSON 파일 경로=============================================================
Path_DB_json = r'C:\Covenant\data\0.DB_Table.json'


# JSON 파일을 읽어 DataFrame으로 변환
with open(Path_DB_json, 'r', encoding='utf-8') as f:
    data = json.load(f)
    df_db = pd.DataFrame(data)

df_db = df_db[['테이블한글명', '테이블영문명', '칼럼명(한글)', '칼럼명(영문)']]



# 드롭다운 목록 생성
table_options = [{'label': row['테이블한글명'], 'value': row['테이블영문명']} for index, row in df_db.iterrows()]
table_options = list({v['value']: v for v in table_options}.values())
#=================================================================================






# 캐시 경로 및 만료 시간 설정
cache_price = r'C:\Covenant\TDF\data\모니터링_TRP.pkl'
cache_expiry = timedelta(days=1)


BM_dict = {
    'ACWI': 'Equity Global',
    'BND': 'Bond Global',
    'SPY': 'US',
    'VUG' : 'Growth',
    'VTV' : 'Value',
    'VEA' : 'DM ex US',
    'VWO' : 'EM',
    'AGG' : 'Bond AGG',
    'LQD': 'Bond IG',
    'HYG': 'Bond HY',
}





# 데이터 가져오기 함수 (캐시 사용)
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

        df_price = df_price.ffill()
        df_price = pd.DataFrame(df_price)
        df_price.columns = [code]
        df_price.index = pd.to_datetime(df_price.index).strftime('%Y-%m-%d')  # 인덱스를 %Y%m%d 형식으로 변환 후 문자열로 저장
        df_price.index = pd.to_datetime(df_price.index).tz_localize(None)  # 시간대 정보 제거

        return df_price

    except Exception as e:
        print(f"Error fetching data for {code}: {e}")
        return None

def Func(code, start, end, batch_size=10):
    # 디버깅용 출력
    print(f"Start: {start}, End: {end}")
    
    # 날짜 형식 강제 변환
    start = pd.to_datetime(start).strftime('%Y-%m-%d')
    end = pd.to_datetime(end).strftime('%Y-%m-%d')

    if isinstance(code, str):  # 단일 코드 처리
        return fetch_data(code, start, end)

    if os.path.exists(cache_price):
        cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_price))
        if datetime.now() - cache_mtime < cache_expiry:
            with open(cache_price, 'rb') as f:
                print("Loading data from cache...")
                df_price = pickle.load(f)
                if df_price.empty:  # 캐싱된 데이터가 비어 있을 경우
                    print("Cached data is empty. Reloading data.")
                    return fetch_data(code, start, end)
                return df_price

    df = []
    for i in range(0, len(code), batch_size):
        code_batch = code[i:i + batch_size]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(fetch_data, c, start, end): c for c in code_batch}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    df.append(result)

    df_price = pd.concat(df, axis=1) if df else pd.DataFrame()
    df_price = df_price.bfill().ffill()

    with open(cache_price, 'wb') as f:
        pickle.dump(df_price, f)
        print("Data cached.")

    return df_price


start = (datetime.today() - relativedelta(years=1)).strftime('%Y-%m-%d')
end = (datetime.today() - timedelta(1)).strftime('%Y-%m-%d')


df_price = Func(list(BM_dict.keys()), start, end)

print(df_price)



















app = dash.Dash(__name__)
app.title = 'TDF_모니터링_TRP'
server = app.server



app.layout = html.Div(
    style={'width': '80%', 'margin': 'auto'},
    children=[

        dcc.Dropdown(
            id='table-dropdown',
            options=table_options,
            value=next((option['value'] for option in table_options if option['label'] == '펀드보유내역'), None),
            style={'width': '50%', 'margin': '10px'}
        ),
        dcc.Dropdown(
            id='column-dropdown',
            multi=True,
            value=[
                    'STD_DT',
                    'FUND_CD',
                    'FUND_NM',
                    'ITEM_NM',
                    'APLD_UPR',
                    'NAST_TAMT_AGNST_WGH',
            ],
            placeholder='Table 컬럼 선택(중복가능)',
            style={'width': '50%', 'margin': '10px'}
        ),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=(datetime.today() - relativedelta(years=1)).strftime('%Y-%m-%d'),
            end_date=(datetime.today() - timedelta(1)).strftime('%Y-%m-%d'),
            display_format='YYYYMMDD',
            style={'width': '50%', 'margin-left': '19px'}
        ),
        dcc.Dropdown(
            id='string-filter-column-dropdown',
            placeholder='문자열 조건 적용할 컬럼',
            value = 'FUND_NM',
            style={'width': '50%', 'margin': '10px'}
        ),
        dcc.Input(
            id='filter-input',
            type='text',
            placeholder='포함할 문자열(콤마로구분)',
            value = '모투자,(모)',
            style={'width': '30%', 'margin': '10px auto'}
        ),
        dcc.Input(
            id='exclude-input',
            type='text',
            placeholder='제외할 문자열',
            value ='포커스,골드',
            style={'width': '30%', 'margin': '10px auto'}
        ),
        html.Button('쿼리 실행', id='execute-query', n_clicks=0, style={'width': '30%', 'margin': '10px auto'}),
        html.Button('엑셀로 다운로드', id='download-excel', n_clicks=0, style={'width': '30%', 'margin': '10px auto'}),
        dcc.Download(id="download"),

        # 기존 db-table 유지
        dash_table.DataTable(
            id='db-table',
            columns=[{"name": col, "id": col} for col in df_db.columns],
            data=df_db.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px', 'minWidth': '100px', 'maxWidth': '200px', 'whiteSpace': 'normal'},
            style_header={'backgroundColor': 'white', 'fontWeight': 'bold'},
            style_as_list_view=True,
            page_size=10  # 페이지 사이즈 설정
        ),

        html.Div(style={'height': '40px'}),




        dash_table.DataTable(
            id='query-result-table',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px', 'minWidth': '100px', 'maxWidth': '200px', 'whiteSpace': 'normal'},
            style_header={'backgroundColor': 'white', 'fontWeight': 'bold'},
            style_as_list_view=True,
            page_size=10  # 페이지 사이즈 설정
        ),


        html.Div([
            dcc.Input(
                id='fund-cd-input',
                type='text',
                placeholder='펀드코드 입력 후 엔터',
                # value= '05N23',
                style={'width': '40%', 'margin': '10px auto'}
            ),
        ]),

        



        # 상태 관리를 위한 Store 컴포넌트
        dcc.Store(id='df-store'),
        dcc.Store(id='df-price-store'),
        html.Div(id='pivot-table-output'),  # 피벗 테이블 결과를 보여줄 영역 추가


    ]
)

# 컬럼과 테이블 데이터를 업데이트하는 콜백 함수 (db-table 유지)
@app.callback(
    [Output('column-dropdown', 'options'),
     Output('db-table', 'data')],
    [Input('table-dropdown', 'value')]
)
def update_columns_and_table(selected_table):
    if selected_table is None:
        return [], []

    filtered_df = df_db[df_db['테이블영문명'] == selected_table]
    column_options = [{'label': row['칼럼명(한글)'], 'value': row['칼럼명(영문)']} for index, row in filtered_df.iterrows()]

    return column_options, filtered_df.to_dict('records')



# 데이터를 저장하는 Store에 데이터프레임 저장
@app.callback(
    Output('df-store', 'data'),
    Input('execute-query', 'n_clicks'),
    [State('table-dropdown', 'value'),
     State('column-dropdown', 'value'),
     State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date'),
     State('string-filter-column-dropdown', 'value'),
     State('filter-input', 'value'),
     State('exclude-input', 'value')]
)
def update_table(n_clicks, selected_table, selected_columns, start_date, end_date, filter_column, include_str, exclude_str):
    if n_clicks == 0 or not selected_table or not selected_columns:
        return None

    connection = pymysql.connect(
        host='192.168.195.55',
        user='solution',
        password='Solution123!',
        database='dt',
        port=3306,
        cursorclass=pymysql.cursors.DictCursor
    )

    def execute_query(query):
        try:
            with connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
                return pd.DataFrame(result)
        except Exception as e:
            print(f"Error executing query: {e}")
            return None

    try:
        # 기본 쿼리 생성
        query = f"""
            SELECT {', '.join(selected_columns)} 
            FROM {selected_table} 
            WHERE 
            # (
            #     {selected_columns[0]} = (SELECT MAX({selected_columns[0]}) FROM {selected_table})
            #     OR {selected_columns[0]} = '{start_date.replace('-', '')}'
            # )
            {selected_columns[0]} BETWEEN '{start_date.replace('-', '')}' 
                                    AND (SELECT MAX({selected_columns[0]}) FROM {selected_table})

            AND (FUND_NM LIKE '%TDF%' 
                OR FUND_NM LIKE '%TIF%' 
                OR FUND_NM LIKE '%글로벌자산배분증권모%' 
                OR FUND_NM LIKE '%글로벌채권증권모%')
        """

        # Include 조건 추가
        if include_str:
            include_conditions = [f"{filter_column} LIKE '%{inc.strip()}%'" for inc in include_str.split(',') if inc.strip()]
            if include_conditions:
                query += " AND (" + " OR ".join(include_conditions) + ")"

        # Exclude 조건 추가
        if exclude_str:
            exclude_conditions = [f"{filter_column} NOT LIKE '%{exc.strip()}%'" for exc in exclude_str.split(',') if exc.strip()]
            if exclude_conditions:
                query += " AND (" + " AND ".join(exclude_conditions) + ")"

        print("Generated Query:", query)  # 쿼리 디버깅용 출력

        # 쿼리 실행
        df_select = execute_query(query)
        connection.close()

        if df_select is not None:
            return df_select.to_dict('records')  # JSON 형태로 저장
        else:
            return None

    except Exception as e:
        print(f"Error executing query: {e}")
        connection.close()
        return None





#필터컬럼을 컬럼 드롭다운 목록과 일치시키는 함수
@app.callback(
    Output('string-filter-column-dropdown', 'options'),
    Input('column-dropdown', 'value')
)
def sync_filter_column_options(selected_columns):
    if not selected_columns:
        return []
    # 선택된 컬럼을 기반으로 옵션 목록 생성
    return [{'label': col, 'value': col} for col in selected_columns]




# df_select를 데이터를 테이블에 표시
@app.callback(
    [Output('query-result-table', 'columns'),
     Output('query-result-table', 'data')],
    Input('df-store', 'data')
)
def update_table_display(data):
    if data is None:
        return [], []

    df = pd.DataFrame(data)
    columns = [{"name": col, "id": col} for col in df.columns]
    return columns, df.to_dict('records')




# 피벗 테이블 생성 및 출력
@app.callback(
    Output('pivot-table-output', 'children'),
    Input('fund-cd-input', 'n_submit'),  # 엔터 키 입력 시 트리거
    State('fund-cd-input', 'value'),  # 입력된 펀드 코드
    State('df-store', 'data'),  # 저장된 데이터
    State('df-price-store', 'data'),  # df_price 데이터를 가져오기 위한 State

)
def generate_pivot_table(n_submit, selected_fund_cd, data, df_price):
    if not n_submit or not selected_fund_cd or data is None:
        return "쿼리실행 후, 펀드코드 넣고 Enter 치세요=================="


    df = pd.DataFrame(data)

    
    # FUND_CD와 FUND_NM을 키와 값으로 하는 딕셔너리 생성
    fund_dict = dict(zip(df['FUND_CD'], df['FUND_NM']))
    print("fund_dict================", fund_dict)



    주식 = [
        '.*Equity.*', 
        '.*Estate.*',
        '.*MSCI.*',
        '.*Resource.*',
        '.*Large.*',
        '.*주식.*',
        '.*자산배분.*',
    ]

    채권 = [
        '.*종합.*', 
        '.*채권.*',
        '.*Bond.*',
        '.*AGG.*',
        '.*Agg.*',
        '.*국고채.*',
        '.*Income.*',
        '.*SICAV - Glo.*',
        '.*SICAV - Div.*',
        '.*RETURN.*',
        '.*SHORT.*',

    ]


    기타 = [
        '.*콜론.*', 
        '.*증거금.*',
        '.*예금.*',
        '.*DEPOSIT.*',
        '.*KRW/USD.*', 
        '.*미수금.*',
        '.*미지급금.*',
        '.*분배금.*',
        '.*CALL.*',

        ]
    
    

    환헷지 = [
        '.*미국달러.*', 
        '.*KRW/USD.*', 
        
        ]
    

    # 투자종목 리스트 만들어서 BM 맵핑하는 과정=============================================
    # 주식과 채권 패턴을 하나의 정규식으로 결합
    주식_pattern = '|'.join(주식)
    채권_pattern = '|'.join(채권)

    # ITEM_NM 열에서 주식 또는 채권 패턴을 포함하는 항목 필터링
    item_list = list(df.loc[
        df['ITEM_NM'].str.contains(주식_pattern + '|' + 채권_pattern, regex=True, na=False), 
        'ITEM_NM'
    ].unique())

    print("item_list================", item_list)




    item_dict = {
        'Emerging Markets Equity Fund': 'Equity Global',
        'Global Natural Resources Equity Fund': 'Equity Global',
        'Global Focused Growth Equity Fund': 'Equity Global',
        'Global Real Estate Securities Fund': 'Equity Global',
        
        'T Rowe Price Funds SICAV - US Large-Cap': 'US',
        'Emerging Markets Discovery Equity Fund': 'EM',
        'US Large Cap Equity Fund': 'Growth',
        'Global Value Equity Fund': 'Value',
        'European Equity Fund': 'DM ex US',
        'Japanese Equity Fund': 'DM ex US',
        'ISHARES MSCI JAPAN USD ACC': 'DM ex US',
        'ISHARES CORE MSCI PACIF X-JP': 'DM ex US',

        'ACE 종합채권(AA-이상)KIS액티브': 'Bond Global',
        'ACE 국고채3년': 'Bond Global',
        'ACE 국고채10년': 'Bond Global',

        'T Rowe Price Funds SICAV - Div': 'Bond HY',
        'Global High Income Bond Fund': 'Bond HY',
        'Emerging Markets Bond Fund': 'BOND HY',
        'Emerging Local Markets Bond Fund': 'Bond HY',

        'T ROWE PRICE TOTAL RETURN': 'Bond AGG',
        'ISHARES US AGG BND USD ACC': 'Bond AGG',
        'ISHARES GLB AGGREGATE BOND USD H A': 'Bond AGG',
        'T. Rowe Price Funds SICAV - Global Aggre': 'Bond AGG',
        'T. ROWE PRICE-RES US AGGRE-I': 'Bond AGG',
        'iShares Core U.S. Aggregate Bo': 'Bond AGG',
        
        'T Rowe Price Funds SICAV - Glo': 'Bond IG',
        'T ROWE PRICE ULTRA SHORT-TER': 'Bond IG',

        '한국투자글로벌자산배분증권모투자신탁(채': 'Equity Global',
        '한국투자글로벌채권증권모투자신탁(채권-재': 'Bond Global',
        '한국투자TDF알아서2050증권모투자신탁(주식': 'Equity Global',
        '한국투자TDF알아서2055증권모투자신탁(주식': 'Equity Global',
        '한국투자TDF알아서2060증권모투자신탁(주식': 'Equity Global',
        '한국투자TDF알아서2020증권모투자신탁(채권': 'Bond Global',
    }





    


    # 피벗 테이블 생성
    PV_W = df.pivot_table(
        index='STD_DT',
        columns=['FUND_CD', 'ITEM_NM'],
        values='NAST_TAMT_AGNST_WGH',
        aggfunc='sum',
        fill_value=0,
        margins=False
    )

    PV_단가 = df.pivot_table(
        index='STD_DT',
        columns=['FUND_CD', 'ITEM_NM'],
        values='APLD_UPR',
        aggfunc='sum',
        fill_value=0,
        margins=False
    )

    
    PV_W = PV_W.loc[:, selected_fund_cd]
    PV_단가 = PV_단가.loc[:, selected_fund_cd]
    

    # 기타 리스트에 포함된 열 필터
    regex_pattern = '|'.join(기타)
    matched_columns = [col for col in PV_W.columns if re.search(regex_pattern, col[1])]
    if matched_columns:
        PV_W['기타'] = PV_W[matched_columns].sum(axis=1)
        PV_W.drop(columns=matched_columns, inplace=True)
        #단가에서는 기타열 삭제했음
        PV_단가.drop(columns=matched_columns, inplace=True)


    # 날짜 형식을 'YYYY-MM-DD'로 변환
    PV_W.index = pd.to_datetime(PV_W.index).strftime('%Y-%m-%d')
    PV_단가.index = pd.to_datetime(PV_단가.index).strftime('%Y-%m-%d')
    PV_단가 = PV_단가.replace([0, np.inf, -np.inf], np.nan).bfill().ffill()
    
    # PV_단가 = PV_단가.apply(pd.to_numeric, errors='coerce')
    
    # save_excel(PV_W, "PV_W", index_option=None)
    # save_excel(PV_단가, "PV_단가", index_option=None)

    PV_R = PV_단가.pct_change().fillna(0)

    cum_단가 = (1+PV_R).cumprod() - 1
    cum_단가 = cum_단가.replace([0, np.inf, -np.inf], np.nan).ffill()
    

    print("PV_W====================", PV_W)
    print("PV_단가====================", PV_단가)
    print("cum_단가=====================", cum_단가)

    
    # save_excel(PV_단가, 'PV_단가')
    # save_excel(PV_R, 'PV_R')
    # save_excel(cum_단가, 'cum_단가')






    # 그룹화 및 가중치 계산 (주식, 채권, 기타, 환헷지)
    def calculate_weights_and_cumulative(columns_pattern, name):
        # Compile the regex pattern
        pattern = '|'.join(columns_pattern)
        
        # Find matched columns
        matched_columns = PV_W.columns[PV_W.columns.to_series().str.contains(pattern, regex=True)]
        
        # If no columns match, return zeroed DataFrames/Series
        if matched_columns.empty:
            print(f"No columns matched for {name}")
            weight = pd.DataFrame(0, index=PV_W.index, columns=[f"{name}_Weight"])
            weight_sum = pd.Series(0, index=PV_W.index, name=f"Total_{name}_Weight")
            cumulative = pd.DataFrame(0, index=PV_R.index, columns=[f"{name}_Cumulative"])
            return weight, weight_sum, cumulative
        
        # If columns match, calculate weight, weight_sum, and cumulative return
        weight = PV_W.loc[:, matched_columns]
        weight_sum = weight.sum(axis=1).rename(f"Total_{name}_Weight")
        cumulative = (1 + PV_R.loc[:, matched_columns]).cumprod() - 1
        cumulative = cumulative.replace([np.inf, -np.inf], np.nan).ffill()
        cumulative.columns = [f"{col}" for col in matched_columns]
        
        return weight, weight_sum.to_frame(name=f'{name}합계'), cumulative



    # 주식, 채권, 기타, 환헷지
    W_주식, sum_W_주식, cum_주식 = calculate_weights_and_cumulative(주식, '주식')
    W_채권, sum_W_채권, cum_채권 = calculate_weights_and_cumulative(채권, '채권')
    W_기타, sum_W_기타, cum_기타 = calculate_weights_and_cumulative(기타, '기타')
    W_환헷지, sum_W_환헷지, cum_환헷지 = calculate_weights_and_cumulative(환헷지, '환헷지')

    
    print("sum_W_주식====================", sum_W_주식)
    print("sum_W_채권====================", sum_W_채권)
    print("cum_주식====================", cum_주식)
    print("cum_채권====================", cum_채권)

    
    


    df_price = pd.read_pickle(cache_price)
    df_R = df_price.pct_change().fillna(0)

    cum_BM = (1 + df_R).cumprod() - 1
    print("cum_BM====================", cum_BM)


    # 열 이름 변경을 위해 BM_dict의 키와 값을 매핑
    cum_BM = cum_BM.rename(columns=BM_dict)













#    # 공통 열만 필터링
#     common_col = list(set(df_R.columns).intersection(df_weight.columns))
#     df_R_filtered = df_R[common_col]
#     df_weight_filtered = df_weight[common_col]

#     # 인덱스 맞추기 (필요 시)
#     df_R_filtered = df_R_filtered.reindex(df_weight.index)
#     df_R_filtered = df_R_filtered.fillna(0)

#     # 원소별 곱셈 수행
#     df_ctr = df_R_filtered * df_weight_filtered
#     cum_port = df_ctr.cumsum()
    
#     cum_port.index = pd.to_datetime(cum_port.index).strftime('%Y-%m-%d')
#     print(cum_port.index)

    

















    for bm_key, bm_name in BM_dict.items():
        if bm_name in cum_BM.columns:
            print(f"BM Key: {bm_key}, BM Name: {bm_name}")
            matching_cols = [col for col in cum_단가.columns if item_dict.get(col) == bm_name]
            print(f"Matching columns for {bm_name}: {matching_cols}")



    
    # BM별 라인 그래프 생성
    bm_graphs = []
    for bm_key, bm_name in BM_dict.items():
        if bm_name in cum_BM.columns:
            # Filter cum_단가 columns that match the current bm_name
            matching_cols = [col for col in cum_단가.columns if item_dict.get(col) == bm_name]

            # If there are no matching columns for cum_단가, skip this BM
            if not matching_cols:
                continue

            # Create Scatter plots for BM and matching cum_단가
            bm_graph = dcc.Graph(
                figure=go.Figure(
                    data=[
                        # Add the BM line
                        go.Scatter(
                            x=cum_BM.index,
                            y=cum_BM[bm_name],
                            mode='lines',
                            name=f"{bm_key} (BM)",  # BM Key for legend
                            hoverlabel=dict(namelength=-1),
                        )
                    ] 
                    
                    +
                    [
                        # Add the lines for cum_단가 columns matching bm_name
                        go.Scatter(
                            x=cum_단가.index,
                            y=cum_단가[col],
                            mode='lines',
                            name=f"{col}",  # Column name for legend
                            hoverlabel=dict(namelength=-1),
                            line=dict(dash='dot'),  # Dashed line for distinction
                        )
                        for col in matching_cols
                    ],
                    
                    layout=go.Layout(
                        title=f"{bm_key} 서브펀드 수익률",  # Title using BM Key
                        xaxis=dict(title="Date", tickformat="%Y-%m-%d"),
                        yaxis=dict(title="Cumulative Return", tickformat=".0%"),
                        showlegend=True,
                        template="plotly_white",
                        annotations=[
                            dict(
                                x=cum_BM.index[-1],
                                y=cum_BM[bm_name].iloc[-1],
                                text=f"{cum_BM[bm_name].iloc[-1]:.1%}",
                                xanchor="left",
                                yanchor="bottom",
                                showarrow=True,
                                arrowhead=2,
                                ax=20,
                                ay=0,
                            )
                        ]
                        +
                        [
                            # Add annotations for each matching cum_단가 column
                            dict(
                                x=cum_단가.index[-1],
                                y=cum_단가[col].iloc[-1],
                                text=f"{cum_단가[col].iloc[-1]:.1%}",
                                xanchor="left",
                                yanchor="bottom",
                                showarrow=True,
                                arrowhead=2,
                                ax=20,
                                ay=0,
                            )
                            for col in matching_cols],
                    )

                )
            )
            bm_graphs.append(bm_graph)








    return html.Div(
        
        style={'width': '80%', 'margin': 'auto'},
        children= [

        
        html.H4(f"{selected_fund_cd} - 투자비중"),

        dash_table.DataTable(
            columns=[
                    {
                        "name": col,
                        "id": col,
                        "type": "numeric",  # 숫자 형식 지정
                        "format": Format(precision=2, scheme=Scheme.fixed)  # .0% 형식 지정 Scheme.percent
                    } 
                    for col in PV_W.reset_index().columns
                ],            
            data=PV_W.reset_index().to_dict('records'),  # 인덱스를 열로 변환 후 데이터 변환            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'center', 
                'padding': '5px', 
                'minWidth': '100px', 
                'maxWidth': '200px', 
                'whiteSpace': 'normal'
            },
            style_header={
                'backgroundColor': '#3762AF', 
                'color': 'white', 
                'fontWeight': 'bold'
            },
            page_size=10  # 페이지 사이즈 설정
        ),

        
        
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Scatter(
                        x=cum_주식.index,  # x축: 인덱스
                        y=cum_주식[col],  # y축: 각 열의 데이터
                        mode='lines',  # 라인
                        name=col,  # 범례 이름
                        hoverlabel=dict(namelength=-1),
                    ) for col in cum_주식.columns  # 각 열에 대해 라인 추가
                ],
                layout=go.Layout(
                    title="Equity 서브펀드 수익률",
                    xaxis=dict(
                        title="Date",
                        tickformat="%Y-%m-%d"  # x축 틱 포맷 설정
                    ),
                    yaxis=dict(
                        title="Cumulative Return",
                        tickformat=".0%"  # y축 틱 포맷 설정
                    ),
                    showlegend=True,
                    template="plotly_white",  # 그래프 스타일 설정
                    annotations=[
                        dict(
                            x=cum_주식.index[-1],  # 마지막 데이터의 x 위치
                            y=cum_주식[col].iloc[-1],  # 마지막 데이터의 y 값
                            text=f"{cum_주식[col].iloc[-1]:.1%}",  # 백분율 형식으로 텍스트 표시
                            xanchor="left",  # 텍스트 위치
                            yanchor="bottom",
                            showarrow=True,  # 화살표 표시 여부
                            arrowhead=2,  # 화살표 스타일
                            ax=20,  # 화살표 x축 오프셋
                            ay=0,  # 화살표 y축 오프셋
                        )
                        for col in cum_주식.columns  # 각 열에 대해 어노테이션 추가
                    ]
                )
            )
        ),



        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Scatter(
                        x=cum_채권.index,  # x축: 인덱스
                        y=cum_채권[col],  # y축: 각 열의 데이터
                        mode='lines',  # 라인
                        name=col,  # 범례 이름
                        hoverlabel=dict(namelength=-1),
                    ) for col in cum_채권.columns  # 각 열에 대해 라인 추가
                ],
                layout=go.Layout(
                    title="FI 서브펀드 수익률",
                    xaxis=dict(
                        title="Date",
                        tickformat="%Y-%m-%d"  # x축 틱 포맷 설정
                    ),
                    yaxis=dict(
                        title="Cumulative Return",
                        tickformat=".0%"  # y축 틱 포맷 설정
                    ),
                    showlegend=True,
                    template="plotly_white",  # 그래프 스타일 설정
                    annotations=[
                        dict(
                            x=cum_채권.index[-1],  # 마지막 데이터의 x 위치
                            y=cum_채권[col].iloc[-1],  # 마지막 데이터의 y 값
                            text=f"{cum_채권[col].iloc[-1]:.1%}",  # 백분율 형식으로 텍스트 표시
                            xanchor="left",  # 텍스트 위치
                            yanchor="bottom",
                            showarrow=True,  # 화살표 표시 여부
                            arrowhead=2,  # 화살표 스타일
                            ax=20,  # 화살표 x축 오프셋
                            ay=0,  # 화살표 y축 오프셋
                        )
                        for col in cum_채권.columns  # 각 열에 대해 어노테이션 추가
                    ]
                )
            )
        ),








        # 투자비중 주식 자산구분 파이 그래프
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Pie(
                        labels=W_주식.columns.map(lambda col: item_dict.get(col, col)),  # item_dict로 매핑
                        values=W_주식.iloc[-1],
                        hole=0.3,  # 도넛 모양
                        textinfo="label+percent",  # 항목 이름과 퍼센트를 그래프에 직접 표시
                        textposition="outside"  # 텍스트를 파이 외부에 표시
                    )
                ],
                layout=go.Layout(
                    title=f"{selected_fund_cd} 주식 자산구분 투자비중",
                    showlegend=False  # 레전드를 숨김
                )
            )
        ),




        # 투자비중 개별주식 펀드 파이 그래프
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Pie(
                        labels=W_주식.columns,
                        values=W_주식.iloc[-1],
                        # labels=PV_W.drop(columns=['hedge', '기타']).columns,
                        # values=PV_W.drop(columns=['hedge', '기타']).iloc[-1],
                        hole=0.3,  # 도넛 모양
                        textinfo="label+percent",  # 항목 이름과 퍼센트를 그래프에 직접 표시
                        textposition="outside"  # 텍스트를 파이 외부에 표시
                    )
                ],
                layout=go.Layout(
                    title=f"{selected_fund_cd} 주식 개별펀드 투자비중",
                    showlegend=False  # 레전드를 숨김
                )
            )
        ),



        # 주식 - 투자비중 라인그래프
        dcc.Graph(
            figure={
                'data': [
                    go.Scatter(
                        x=W_주식.index,
                        y=W_주식[col],
                        mode='lines',
                        name=f'{col}',
                        hoverlabel=dict(namelength=-1),
                        # line=dict(color='red')
                    ) for col in W_주식.columns
                ],

                'layout': {
                    'title': "투자비중_주식",
                    'xaxis': {'title': "Date", 'tickformat': "%Y-%m-%d"},
                    'yaxis': {'title': "Weight", 'tickformat': ".1f"},
                    'template': "plotly_white",
                    'showlegend': True,
                    'annotations': [
                        dict(
                            x=W_주식.index[-1],  # 마지막 데이터의 x 위치
                            y=W_주식[col].iloc[-1],  # 마지막 데이터의 y 값
                            text=f"{W_주식[col].iloc[-1]:.1f}",  # 백분율 형식으로 텍스트 표시
                            xanchor="left",  # 텍스트 위치
                            yanchor="bottom",
                            showarrow=True,  # 화살표 표시 여부
                            arrowhead=2,  # 화살표 스타일
                            ax=20,  # 화살표 x축 오프셋
                            ay=0,  # 화살표 y축 오프셋
                        )
                        for col in W_주식.columns  # 각 열에 대해 어노테이션 추가
                    ]
                }
            }
        ),







        # 투자비중 채권 자산구분 파이 그래프
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Pie(
                        labels=W_채권.columns.map(lambda col: item_dict.get(col, col)),  # item_dict로 매핑
                        values=W_채권.iloc[-1],
                        hole=0.3,  # 도넛 모양
                        textinfo="label+percent",  # 항목 이름과 퍼센트를 그래프에 직접 표시
                        textposition="outside"  # 텍스트를 파이 외부에 표시
                    )
                ],
                layout=go.Layout(
                    title=f"{selected_fund_cd} 채권 자산구분 투자비중",
                    showlegend=False  # 레전드를 숨김
                )
            )
        ),





        # 투자비중 개별 채권 파이 그래프
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Pie(
                        labels=W_채권.columns,
                        values=W_채권.iloc[-1],
                        # labels=PV_W.drop(columns=['hedge', '기타']).columns,
                        # values=PV_W.drop(columns=['hedge', '기타']).iloc[-1],
                        hole=0.3,  # 도넛 모양
                        textinfo="label+percent",  # 항목 이름과 퍼센트를 그래프에 직접 표시
                        textposition="outside"  # 텍스트를 파이 외부에 표시
                    )
                ],
                layout=go.Layout(
                    title=f"{selected_fund_cd} 개별 채권펀드 투자비중",
                    showlegend=False  # 레전드를 숨김
                )
            )
        ),




        # 채권 - 투자비중 라인그래프
        dcc.Graph(
            figure={
                'data': [
                    go.Scatter(
                        x=W_채권.index,
                        y=W_채권[col],
                        mode='lines',
                        name=f'{col}',
                        hoverlabel=dict(namelength=-1),
                        # line=dict(color='red')
                    ) for col in W_채권.columns
                ],

                'layout': {
                    'title': "투자비중_채권",
                    'xaxis': {'title': "Date", 'tickformat': "%Y-%m-%d"},
                    'yaxis': {'title': "Weight", 'tickformat': ".1f"},
                    'template': "plotly_white",
                    'showlegend': True,
                    'annotations': [
                        dict(
                            x=W_채권.index[-1],  # 마지막 데이터의 x 위치
                            y=W_채권[col].iloc[-1],  # 마지막 데이터의 y 값
                            text=f"{W_채권[col].iloc[-1]:.1f}",  # 백분율 형식으로 텍스트 표시
                            xanchor="left",  # 텍스트 위치
                            yanchor="bottom",
                            showarrow=True,  # 화살표 표시 여부
                            arrowhead=2,  # 화살표 스타일
                            ax=20,  # 화살표 x축 오프셋
                            ay=0,  # 화살표 y축 오프셋
                        )
                        for col in W_채권.columns  # 각 열에 대해 어노테이션 추가
                    ]
                }
            }
        ),




        # 환헷지 - 투자비중 라인그래프
        dcc.Graph(
            figure={
                'data': [
                    go.Scatter(
                        x=sum_W_환헷지.index,
                        y=sum_W_환헷지[col],
                        mode='lines',
                        name=f'{col}',
                        hoverlabel=dict(namelength=-1),
                        # line=dict(color='red')
                    ) for col in sum_W_환헷지.columns
                ],

                'layout': {
                    'title': "USD Sell Exposure",
                    'xaxis': {'title': "Date", 'tickformat': "%Y-%m-%d"},
                    'yaxis': {'title': "Weight", 'tickformat': ".1f"},
                    'template': "plotly_white",
                    'showlegend': True,
                    'annotations': [
                        dict(
                            x=sum_W_환헷지.index[-1],  # 마지막 데이터의 x 위치
                            y=sum_W_환헷지[col].iloc[-1],  # 마지막 데이터의 y 값
                            text=f"{sum_W_환헷지[col].iloc[-1]:.1f}",  # 백분율 형식으로 텍스트 표시
                            xanchor="left",  # 텍스트 위치
                            yanchor="bottom",
                            showarrow=True,  # 화살표 표시 여부
                            arrowhead=2,  # 화살표 스타일
                            ax=20,  # 화살표 x축 오프셋
                            ay=0,  # 화살표 y축 오프셋
                        )
                        for col in sum_W_환헷지.columns  # 각 열에 대해 어노테이션 추가
                    ]
                }
            }
        ),





        # Render all graphs
        *bm_graphs,


])










#엑셀 저장=======================================================
def save_excel(df, sheetname, index_option=None):
    
    # 파일 경로
    path = rf'C:\Covenant\TDF\data\모니터링_TRP.xlsx'

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
        print(f"'{path}  {sheetname}' 저장 완료.")






@app.callback(
    Output("download", "data"),
    Input("download-excel", "n_clicks"),
    State('query-result-table', 'data'),
    prevent_initial_call=True,
)
def download_excel(n_clicks, table_data):
    if not table_data:
        return None
    df = pd.DataFrame(table_data)
    return dcc.send_data_frame(df.to_excel, "query_result.xlsx", sheet_name="df_select")





# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
