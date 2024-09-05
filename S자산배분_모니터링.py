import FinanceDataReader as fdr
from pykrx import stock as pykrx
import pandas as pd
from datetime import datetime, timedelta
import json
import pymysql
import dash
from dash import dcc, html, Input, Output, State, dash_table
from dateutil.relativedelta import relativedelta
import concurrent.futures
from tqdm import tqdm
import plotly.graph_objects as go
import numpy as np
import re

# Excel 파일 경로 : json 테이블 생성할 때만 사용
# Path_DB = r'C:\Covenant\data\0.DB_Table.xlsx'
# df_db = pd.read_excel(Path_DB, sheet_name='테이블명세')
# df_db.to_json(Path_DB_json, orient='records')

Path_DB_json = r'C:\Covenant\data\0.DB_Table.json'

# JSON 파일을 읽어 DataFrame으로 변환
with open(Path_DB_json, 'r', encoding='utf-8') as f:
    data = json.load(f)
    df_db = pd.DataFrame(data)

df_db = df_db[['테이블한글명', '테이블영문명', '칼럼명(한글)', '칼럼명(영문)']]

# 드롭다운 목록 생성
table_options = [{'label': row['테이블한글명'], 'value': row['테이블영문명']} for index, row in df_db.iterrows()]
table_options = list({v['value']: v for v in table_options}.values())
column_options = [{'label': row['칼럼명(한글)'], 'value': row['칼럼명(영문)']} for index, row in df_db.iterrows()]


EQ_dict = {
    'Global X U.S. Infrasturucture Devolopmen': 'PAVE',
    'ISHARES CORE MSCI EMERGING': 'IEMG',
    'ISHARES MSCI ACWI ETF': 'ACWI',
    'ISHARES PHLX SOX SEMICONDUCT': 'SOXX',
    'Invesco QQQ Trust Series 1': 'QQQ',
    'ROUNDHILL MAGNIFICENT SEVEN ETF': 'MAGS',
    'SPDR DJIA TRUST': 'DIA',
    'SPDR TRUST SERIES 1': 'SPY',
    'VANGUARD FTSE DEVELOPED ETF': 'VEA',
    'VANGUARD GROWTH ETF': 'VUG',
    'VANGUARD LONG-TERM CORP BOND': 'VCLT',
    'Vanguard Total Stock Market ET': 'VTI',
    'KODEX 200': '069500',
}

FI_dict = {
    "ACE 국고채3년": "114460",
    "ACE 국고채10년": "365780",
    'KOSEF 국고채10년': '148070',
    "ACE 종합채권(AA-이상)KIS액티브": "356540",
    "KBSTAR 종합채권(A-이상)액티브": "385540",
    "RISE 종합채권(A-이상)액티브": "385540",
    "KODEX 종합채권(AA-이상)액티브": "273130",
    'KODEX 국고채30년액티브': '276990',
}

code1 = {
    "069500": ["KOSPI200", 2.4],     
    "SPY": ["S&P500", 4.9],        
    "URTH": ["MSCI World", 15.0],   
    "VWO": ["MSCI EM", 6.6],        
    "IWF": ["Russell 1000 Growth", 12.2],  
    "QQQ": ["NASDAQ 100", 3.9],     
    "SOXX": ["Semiconductor", 2.0], 
    "PAVE": ["US Infrastructure", 2.0]   
}

code2 = [
    'ACWI', 'BND', 
    '069500', 'SPY', 'URTH', 'VTI', 
    'QQQ', 'SOXX',
    'VUG', 'SPYG', 'IWF',
    'VTV', 'SPYV', 'IWD',
    'VEA', 'VWO', 'SCHE',
    'IAUM',
    'USD/KRW',
] 

# 딕셔너리를 데이터프레임으로 변환
df_code1 = pd.DataFrame.from_dict(code1, orient='index', columns=['Name', 'Weight'])
df_EQ_dict = pd.DataFrame(list(EQ_dict.items()), columns=['Name', 'Code'])

# 중복 항목 제거
code = list(
    set(
        list(code1.keys()) 
        + code2 
        + list(EQ_dict.values()) 
        + list(FI_dict.values())
    )
)

print("code===================================", code)

def fetch_data(code, start, end):
    try:
        # 주식 코드가 숫자로만 되어 있으면 pykrx를 사용
        if code.isdigit() or code.endswith(".KS"):
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

# 시작 날짜 설정
# start = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
start = datetime(datetime.today().year, 1, 1).strftime('%Y-%m-%d')
end = (datetime.today() - timedelta(days=0)).strftime('%Y-%m-%d')

ETF_price = FDR(code, start, end).ffill().fillna(0)
ETF_R = ETF_price.pct_change(periods=1).replace([-1, np.nan, np.inf, -np.inf], 0)
print(ETF_R.head())

cum_ETF = (1+ETF_R).cumprod() -1 
print(cum_ETF.tail())

# 누적 수익률 그래프 생성
EQ_figure = go.Figure()

# EQ_dict의 값에 해당하는 항목만 필터링
eq_etf_codes = list(EQ_dict.values())
cum_ETF_EQ = cum_ETF[eq_etf_codes]

for col in cum_ETF_EQ.columns:
    EQ_figure.add_trace(
        go.Scatter(
            x=cum_ETF_EQ.index,
            y=cum_ETF_EQ[col],
            mode='lines',
            name=col,
            hovertext=[f"{col}: {y:.2%}" for y in cum_ETF_EQ[col]],
            hoverinfo='text+x+y'
        )
    )

EQ_figure.update_layout(
    title='Equity ETF Cumulative Return',
    xaxis={
        'title': 'Date',
        'tickformat': '%Y-%m-%d',
    },
    yaxis={'title': 'Cumulative Return', 'tickformat': '.1%'},
)

# FI_dict의 값에 해당하는 항목만 필터링
fi_etf_codes = list(FI_dict.values())
cum_ETF_FI = cum_ETF[fi_etf_codes]

# 누적 수익률 그래프 생성 for FI
FI_figure = go.Figure()
for col in cum_ETF_FI.columns:
    # FI_dict에서 해당하는 키를 찾아 범례로 사용
    legend_name = next((k for k, v in FI_dict.items() if v == col), col)
    
    hovertext = []
    for y in cum_ETF_FI[col]:
        try:
            hovertext.append(f"{legend_name}: {y:.2%}")
        except (TypeError, ValueError):
            hovertext.append(f"{legend_name}: N/A")
    
    FI_figure.add_trace(
        go.Scatter(
            x=cum_ETF_FI.index,
            y=cum_ETF_FI[col],
            mode='lines',
            name=legend_name,  # 범례에 키 이름 사용
            hovertext=hovertext,
            hoverinfo='text+x+y'
        )
    )

FI_figure.update_layout(
    title='FI ETF Cumulative Return',
    xaxis={
        'title': 'Date',
        'tickformat': '%Y-%m-%d',
    },
    yaxis={'title': 'Cumulative Return', 'tickformat': '.1%'},
)



app = dash.Dash(__name__)
app.title = 'S자산배분 모니터링'
server = app.server

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

@app.callback(
    [Output('query-result-table', 'columns'),
     Output('query-result-table', 'data'),
     Output('pivot-table', 'columns'),
     Output('pivot-table', 'data'),
     Output('pivot-table-R', 'columns'),
     Output('pivot-table-R', 'data'),
     Output('graph-PV_weight-EQ', 'figure'),
     Output('graph-PV_weight-FI', 'figure')],
    Input('execute-query', 'n_clicks'),
    [State('table-dropdown', 'value'),
     State('column-dropdown', 'value'),
     State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date'),
     State('string-filter-column-dropdown', 'value'),
     State('filter-input', 'value'),
     State('exclude-input', 'value'),
     State('pivot-filter-input', 'value')]
)
def update_table(n_clicks, selected_table, selected_columns, start_date, end_date, filter_column, include_str, exclude_str, pivot_filter_str):
    if n_clicks == 0 or not selected_table or not selected_columns:
        return [], [], [], [], [], [], {}, {}, {}, {}

    df_select = get_df_select(selected_table, selected_columns, start_date, end_date, filter_column, include_str, exclude_str)

    if df_select is not None:
        columns = [{"name": col, "id": col} for col in df_select.columns]
        data = df_select.to_dict('records')

        # 피벗 테이블 생성
        PV_W = pd.pivot_table(
            df_select,
            values='NAST_TAMT_AGNST_WGH',  # 집계할 값
            index='STD_DT',                # 행 인덱스
            columns='ITEM_NM',             # 열 인덱스
            aggfunc='mean',                # 집계 함수
            fill_value=0                   # NaN 값을 0으로 채움
        )
        PV_W.index = pd.to_datetime(PV_W.index, format='%Y%m%d').strftime('%Y-%m-%d')
        print('PV_W================', PV_W)

        PV_price = pd.pivot_table(
            df_select,
            values='APLD_UPR',             # 집계할 값
            index='STD_DT',                # 행 인덱스
            columns='ITEM_NM',             # 열 인덱스
            aggfunc='mean',                # 집계 함수
            fill_value=0                   # NaN 값을 0으로 채움
        )
        PV_R = PV_price.pct_change(periods=1)

        print('PV_W.columns=================', PV_W.columns)

        X_keywords = ['.*KRW.*', '.*USD.*', '.*DEPOSIT.*', '.*예금.*',
                      '.*분배금.*', '.*원천세.*', '.*미수금.*', '.*미지급금.*', '.*REPO.*']

        # 제외할 열을 제외한 나머지 열 이름만 남기기
        PV_W_columns = [col for col in PV_W.columns if not any(re.match(keyword, col) for keyword in X_keywords)]
        PV_W_columns = sorted(list(set(PV_W_columns)))

        PV_W = PV_W[PV_W_columns]
        print('PV_W XXXXXXX.columns=================', PV_W.columns)

        PV_R_columns = [col for col in PV_R.columns if not any(re.match(keyword, col) for keyword in X_keywords)]
        PV_R_columns = sorted(list(set(PV_R_columns)))
        PV_R = PV_R[PV_R_columns]
        PV_R = PV_R.replace([-1, np.nan], 0)

        print('PV_R.columns=================', PV_R.columns)

        PV_W_columns = [{"name": col, "id": col} for col in PV_W.reset_index().columns]
        PV_W_data = PV_W.applymap(lambda x: f'{x:.2f}%')
        PV_W_data = PV_W_data.reset_index().to_dict('records')

        PV_R_columns = [{"name": col, "id": col} for col in PV_R.reset_index().columns]
        PV_R_data = PV_R.applymap(lambda x: f'{x:.2%}')
        PV_R_data = PV_R_data.reset_index().to_dict('records')

        ctr = PV_W * PV_R

        # 누적 수익률 계산
        cum_PV_R = (1 + ctr/100).cumprod() - 1
        cum_PV_R.replace(-1, np.nan, inplace=True)
        cum_PV_R.ffill(inplace=True)

        # PV_W를 EQ와 FI로 나누기
        PV_W_EQ = PV_W[[col for col in PV_W.columns if col in EQ_dict.keys()]]
        PV_W_FI = PV_W[[col for col in PV_W.columns if col in FI_dict.keys()]]

        # EQ 그래프 생성
        figure_EQ = go.Figure()
        for col in PV_W_EQ.columns:
            hover_text = [col] * len(PV_W_EQ)
            figure_EQ.add_trace(
                go.Scatter(
                    x=PV_W_EQ.index,
                    y=PV_W_EQ[col],
                    mode='lines',
                    name=col,
                    hovertext=hover_text,
                    hoverinfo='text+x+y'
                )
            )

        figure_EQ.update_layout(
            title='Equity Portfolio Weight',
            xaxis={
                'title': 'Date',
                'tickformat': '%Y%m%d',
            },
            yaxis={'title': 'Return', 'tickformat': '.1f'},
        )

        # FI 그래프 생성
        figure_FI = go.Figure()
        for col in PV_W_FI.columns:
            hover_text = [col] * len(PV_W_FI)
            figure_FI.add_trace(
                go.Scatter(
                    x=PV_W_FI.index,
                    y=PV_W_FI[col],
                    mode='lines',
                    name=col,
                    hovertext=hover_text,
                    hoverinfo='text+x+y'
                )
            )

        figure_FI.update_layout(
            title='Fixed Income Portfolio Weight',
            xaxis={
                'title': 'Date',
                'tickformat': '%Y%m%d',
            },
            yaxis={'title': 'Return', 'tickformat': '.1f'},
        )

        return columns, data, PV_W_columns, PV_W_data, PV_R_columns, PV_R_data, figure_EQ, figure_FI
    else:
        return [], [], [], [], [], [], {}, {}, {}, {}

def get_df_select(selected_table, selected_columns, start_date, end_date, filter_column, include_str, exclude_str):
    connection = pymysql.connect(
        host='192.168.195.55',
        user='solution',
        password='Solution123!',
        database='dt',
        port=3306,
        cursorclass=pymysql.cursors.DictCursor
    )

    def execute_query(connection, query):
        try:
            with connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
                return pd.DataFrame(result)
        except Exception as e:
            print(f"Error executing query: {e}")
            return None

    try:
        query = f"""
        SELECT {', '.join(selected_columns)} 
        FROM {selected_table} 
        WHERE {selected_columns[0]} >= '{start_date.replace('-', '')}' AND {selected_columns[0]} <= '{end_date.replace('-', '')}'
        """

        if include_str:
            include_conditions = [f"{filter_column} LIKE '%{inc}%'" for inc in include_str.split(',')]
            query += " AND (" + " OR ".join(include_conditions) + ")"
        if exclude_str:
            exclude_conditions = [f"{filter_column} NOT LIKE '%{exc}%'" for exc in exclude_str.split(',')]
            query += " AND (" + " OR ".join(exclude_conditions) + ")"

        df_select = execute_query(connection, query)
        connection.close()

        return df_select

    except Exception as e:
        print(f"Error executing query: {e}")
        connection.close()
        return None

@app.callback(
    Output('string-filter-column-dropdown', 'options'),
    Input('column-dropdown', 'value')
)
def update_string_filter_column_options(selected_columns):
    if not selected_columns:
        return []

    return [{'label': col, 'value': col} for col in selected_columns]

@app.callback(
    Output("download", "data"),
    Input("download-excel", "n_clicks"),
    State('query-result-table', 'data'),
    prevent_initial_call=True,
)
def download_as_excel(n_clicks, table_data):
    if not table_data:
        return None
    df = pd.DataFrame(table_data)
    return dcc.send_data_frame(df.to_excel, "query_result.xlsx", sheet_name="Sheet1", index=False)

@app.callback(
    Output("download-pivot", "data"),
    Input("download-pivot-excel", "n_clicks"),
    State('pivot-table', 'data'),
    prevent_initial_call=True,
)
def download_pivot_as_excel(n_clicks, pivot_data):
    if not pivot_data:
        return None
    df = pd.DataFrame(pivot_data)
    return dcc.send_data_frame(df.to_excel, "PV_W.xlsx", sheet_name="PivotSheet", index=False)

@app.callback(
    Output("download-pivot-R", "data"),
    Input("download-pivot-R-excel", "n_clicks"),
    State('pivot-table-R', 'data'),
    prevent_initial_call=True,
)
def download_pivot_r_as_excel(n_clicks, pivot_r_data):
    if not pivot_r_data:
        return None
    df = pd.DataFrame(pivot_r_data)
    return dcc.send_data_frame(df.to_excel, "PV_R.xlsx", sheet_name="PivotRSheet", index=False)

app.layout = html.Div(
    style={'width': '60%', 'margin': 'auto'},
    children=[
        dcc.Dropdown(
            id='table-dropdown',
            options=table_options,
            # 초기값 설정
            value=next((option['value'] for option in table_options if option['label'] == '펀드보유내역'), None),
            placeholder='Table 선택',
            style={'width': '50%', 'margin': '10px'}
        ),
        dcc.Dropdown(
            id='column-dropdown',
            multi=True,
            placeholder='Table 컬럼 선택(중복가능)',
            # 초기값 설정
            value=['STD_DT', 'FUND_CD', 'MNR_NM', 'FUND_NM', 'ITEM_NM', 'ITEM_CD', 'APLD_UPR', 'NAST_TAMT_AGNST_WGH'],
            style={'width': '50%', 'margin': '10px'}
        ),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date='2024-01-01',
            end_date=datetime.today().strftime('%Y-%m-%d'),
            display_format='YYYYMMDD',
            style={'width': '50%', 'margin-left': '19px'}
        ),
        dcc.Dropdown(
            id='string-filter-column-dropdown',
            placeholder='문자열 조건 적용할 컬럼',
            value=next((option['value'] for option in column_options if option['label'] == '펀드코드'), None),
            style={'width': '50%', 'margin': '10px'}
        ),
        dcc.Input(
            id='filter-input',
            type='text',
            placeholder='포함할 문자열(콤마로구분)',
            value='3JM13',
            style={'width': '30%', 'margin': '10px auto'}
        ),
        dcc.Input(
            id='exclude-input',
            type='text',
            placeholder='제외할 문자열',
            value='',
            style={'width': '30%', 'margin': '10px auto'}
        ),
        html.Button(
            '쿼리 실행', 
            id='execute-query', 
            n_clicks=1, 
            style={
                'width': '20%', 
                'margin': '10px auto',
                'color': '#FFFFFF', #글자 흰색 
                'backgroundColor' : '#3762AF',
        }),
        html.Div(style={'height': '10px'}),
        dcc.Input(
            id='pivot-filter-input',
            type='text',
            placeholder='피벗 테이블 열 필터링(콤마로 구분)',
            value='  ',
            style={'width': '30%', 'margin': '10px'}
        ),
        dash_table.DataTable(
            id='db-table',
            columns=[{"name": col, "id": col} for col in df_db.columns],
            data=df_db.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px', 'minWidth': '100px', 'maxWidth': '200px', 'whiteSpace': 'normal'},
            style_header={'backgroundColor': 'white', 'fontWeight': 'bold'},
            style_as_list_view=True,
            page_size=15,
        ),
        html.Div(style={'height': '40px'}),
        html.Button('데이터 다운로드', id='download-excel', n_clicks=0, style={'width': '30%', 'margin': '10px auto'}),
        dcc.Download(id="download"),
        dash_table.DataTable(
            id='query-result-table',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px', 'minWidth': '100px', 'maxWidth': '200px', 'whiteSpace': 'normal'},
            style_header={'backgroundColor': 'white', 'fontWeight': 'bold'},
            style_as_list_view=True,
            page_size=15,
        ),
        html.Div(style={'height': '40px'}),
        html.Button('Weight Excel 다운로드', id='download-pivot-excel', n_clicks=0, style={'width': '20%', 'margin': '10px auto'}),
        dcc.Download(id="download-pivot"),
        dash_table.DataTable(
            id='pivot-table',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px', 'minWidth': '100px', 'maxWidth': '200px', 'whiteSpace': 'normal'},
            style_header={'backgroundColor': 'white', 'fontWeight': 'bold'},
            style_as_list_view=True,
            page_size=10,
        ),
        html.Div(style={'height': '40px'}),
        html.Button('Return Excel 다운로드', id='download-pivot-R-excel', n_clicks=0, style={'width': '20%', 'margin': '10px auto'}),
        dcc.Download(id="download-pivot-R"),
        dash_table.DataTable(
            id='pivot-table-R',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px', 'minWidth': '100px', 'maxWidth': '200px', 'whiteSpace': 'normal'},
            style_header={'backgroundColor': 'white', 'fontWeight': 'bold'},
            style_as_list_view=True,
            page_size=10,
        ),
        dcc.Graph(id='graph-PV_weight-EQ'),
        dcc.Graph(id='graph-PV_weight-FI'),
        dcc.Graph(
            id='graph-cum-etf',
            figure=EQ_figure
        ),
        dcc.Graph(
            id='graph-cum-fi',
            figure=FI_figure
        ),
    ]
)

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
