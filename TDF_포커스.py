import FinanceDataReader as fdr
from pykrx import stock as pykrx
import os
import pandas as pd
from datetime import datetime, timedelta
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pymysql
from dateutil.relativedelta import relativedelta
import numpy as np
import scipy.stats as stats
import concurrent.futures
from tqdm import tqdm  # tqdm 임포트

# 날짜 변수 정의
today = datetime.now()
start_date = '20221005'
end_date = today.strftime('%Y%m%d')




# 데이터베이스 쿼리 실행 함수===========================
def execute_query(connection, query):
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
            return result
    except Exception as e:
        print(f"Error executing query: {e}")
        return None

# 데이터베이스에서 데이터 가져오기
def fetch_data(query):
    connection = pymysql.connect(
        host='192.168.195.55',
        user='solution',
        password='Solution123!',
        database='dt',
        port=3306,
        cursorclass=pymysql.cursors.DictCursor
    )

    result = execute_query(connection, query)
    connection.close()
    
    if result is not None:
        return pd.DataFrame(result)
    else:
        return None

# 메인 데이터 로드 함수
def main(start_date, end_date):
    queries = {
        'query1': f"""
            SELECT 
                A.GIJUN_YMD,         
                A.FUND_CD,           
                A.SUIK_JISU,         
                B.FUND_FNM           
            FROM 
                FDTFN201 as A
            LEFT JOIN 
                FDTFN001 as B
            ON 
                A.FUND_CD = B.FUND_CD
            WHERE 
                (B.FUND_FNM LIKE '%TDF%' OR B.FUND_FNM LIKE '%TIF%')
                AND GIJUN_YMD >= '{start_date}';
        """,
    }

    results = {}

    # TQDM을 사용하여 쿼리 실행 진행률 표시
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_query = {executor.submit(fetch_data, query): name for name, query in queries.items()}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_query), total=len(queries), desc="Fetching data"):
            query_name = future_to_query[future]
            try:
                result = future.result()
                if result is not None:
                    results[query_name] = result
                else:
                    print(f"{query_name} returned no results.")
            except Exception as e:
                print(f"Error fetching data for {query_name}: {e}")

    df1 = results.get('query1')

    if df1 is not None:
        df1['GIJUN_YMD'] = pd.to_datetime(df1['GIJUN_YMD'].astype(str), format='%Y%m%d')
        df1 = df1.ffill()

    return df1, print("df1================", df1)

if __name__ == "__main__":
    start_date = '20221005'
    end_date = datetime.now().strftime('%Y%m%d')

    # 전체 작업에 대한 진행률 표시
    with tqdm(total=100, desc="Total Progress", leave=False) as pbar:
        df1 = main(start_date, end_date)
        pbar.update(100)  # 모든 작업 완료 후 100%로 업데이트

    print("Processing complete.")

#========================================================



# 데이터 처리 및 준비
path_TDF = r'C:\Covenant\data'
if not os.path.exists(path_TDF):
    os.makedirs(path_TDF)
    
file_name = 'TDF수익률지수.pkl'
path_TDF_pkl = os.path.join(path_TDF, file_name)

# df1을 .pkl 파일에서 읽어오기
df1 = pd.read_pickle(path_TDF_pkl)

start = df1['GIJUN_YMD'].iloc[0]
T0 = df1['GIJUN_YMD'].iloc[-1]
print(T0)

# 상장지수펀드 제외
df1 = df1[~df1['FUND_FNM'].str.contains('상장지수')]

# '운용사명' 열을 FUND_FNM 열의 첫 두 글자로 생성
df1['운용사명'] = df1['FUND_FNM'].apply(lambda x: x[:2] if pd.notnull(x) else '')

# 빈티지열 기본값을 기타로 설정하고 빈티지인 경우 빈티지값을 빈티지 열에 추가
df1['빈티지'] = '기타'

vintage_list = ['TIF', '2030', '2035', '2040', '2045', '2050', '2055', '2060']
for vintage in vintage_list:
    df1.loc[df1['FUND_FNM'].str.contains(vintage), '빈티지'] = vintage

# 디폴트 및 특수구분 열 추가
df1.loc[df1['FUND_FNM'].str.contains('O'), '디폴트'] = 'O'
df1.loc[df1['FUND_FNM'].str.contains('(모)'), '운용펀드'] = '운용펀드'
df1.loc[df1['FUND_FNM'].str.contains('ETF포커스'), '특수구분'] = '한투 포커스'
df1.loc[df1['FUND_FNM'].str.contains('알아서') & ~df1['FUND_FNM'].str.contains('ETF포커스'), '특수구분'] = '한투 TRP'
df1.loc[df1['FUND_FNM'].str.contains('삼성한국형'), '특수구분'] = '삼성한국형'
df1.loc[df1['FUND_FNM'].str.contains('ETF를담은'), '특수구분'] = '삼성ETF담은'
df1.loc[df1['FUND_FNM'].str.contains('미래에셋자산배분'), '특수구분'] = '미래자산'
df1.loc[df1['FUND_FNM'].str.contains('미래에셋전략배분'), '특수구분'] = '미래전략'
df1.loc[df1['FUND_FNM'].str.contains('온국민'), '특수구분'] = 'KB 온국민'
df1.loc[df1['FUND_FNM'].str.contains('다이나믹'), '특수구분'] = 'KB 다이나믹'
df1.loc[~df1['FUND_FNM'].str.contains('온국민|다이나믹') & df1['FUND_FNM'].str.contains('케이비'), '특수구분'] = 'KB'
df1.loc[df1['FUND_FNM'].str.contains('신한장기'), '특수구분'] = '신한 장기성장'
df1.loc[df1['FUND_FNM'].str.contains('신한마음편한'), '특수구분'] = '신한 마음편한'
df1.loc[df1['FUND_FNM'].str.contains('KCGI'), '특수구분'] = 'KCGI'
df1.loc[df1['FUND_FNM'].str.contains('IBK'), '특수구분'] = 'IBK'
df1.loc[df1['FUND_FNM'].str.contains('BNK'), '특수구분'] = 'BNK'
df1.loc[df1['FUND_FNM'].str.contains('NH'), '특수구분'] = 'NH'
df1.loc[df1['FUND_FNM'].str.contains('케이에스에프'), '특수구분'] = 'KSF'
df1.loc[df1['FUND_FNM'].str.contains('케이디비'), '특수구분'] = 'KDB'
df1.loc[df1['FUND_FNM'].str.contains('DB자산'), '특수구분'] = 'DB'

# 요약명칭 열 생성
df1['요약명칭'] = df1.apply(
    lambda row: f"{row['특수구분']} {row['빈티지']}{' UH' if 'UH' in row['FUND_FNM'] else ''}" 
                if pd.notnull(row['특수구분']) 
                else "{} {}{}".format(
                    row['운용사명'][:2] if pd.notnull(row['운용사명']) 
                    else (row['FUND_FNM'][:2] if pd.notnull(row['FUND_FNM']) 
                          else row['FUND_FNM'][:2]), 
                    row['빈티지'], 
                    ' UH' if 'UH' in row['FUND_FNM'] else ''),
    axis=1
)

# '특수구분'이 '한투 포커스' 또는 '한투 TRP'에 해당하고, '운용펀드'가 '운용펀드'인 행만 남기기
df1_한투_운용펀드 = df1[df1['특수구분'].isin(['한투 포커스', '한투 TRP']) & df1['운용펀드'].str.contains('운용펀드', na=False)]

# '특수구분'이 '한투 포커스' 또는 '한투 TRP'가 아닌 행만 남기기
df1_한투외펀드 = df1[~df1['특수구분'].isin(['한투 포커스', '한투 TRP'])]

# 두 데이터프레임을 결합하기
df1 = pd.concat([df1_한투_운용펀드, df1_한투외펀드])

print('df1_운용펀드===============', df1)

PV1 = df1.pivot_table(index='GIJUN_YMD', columns=['요약명칭'], values='SUIK_JISU', aggfunc='mean', fill_value=0, margins=False)
PV1.index = pd.to_datetime(PV1.index)
PV1 = PV1.asfreq('D')
PV1 = PV1.ffill()
print('PV1===============', PV1.head())

# 수익률 데이터 계산
df_R = PV1.pct_change(periods=1).fillna(0)
df_R = df_R.apply(lambda col: col.map(lambda x: 0 if abs(x) > 0.05 else x))
df_R.replace([np.inf, -np.inf], np.nan, inplace=True)  # 무한대 값을 NaN으로 대체
df_R.ffill()

# 1년 전 같은 날짜를 계산하여 각 날짜에 대한 롤링 1년 수익률 계산
df_RR_1Y = PV1.copy()
df_RR_1Y['1년 전'] = df_RR_1Y.index - pd.DateOffset(years=1)

def calculate_RR_1Y_return(current_date, df):
    one_year_ago = current_date - pd.DateOffset(years=1)
    if one_year_ago in df.index:
        return df.loc[current_date] / df.loc[one_year_ago] - 1
    else:
        return pd.Series([np.nan] * df.shape[1], index=df.columns)

df_1Y = df_RR_1Y.index.map(lambda date: calculate_RR_1Y_return(date, PV1))
df_1Y = pd.DataFrame(df_1Y.tolist(), index=PV1.index, columns=PV1.columns)

# 결측값을 NaN으로 대체
df_1Y.replace([np.inf, -np.inf], np.nan, inplace=True)
df_1Y.ffill()


# 변동성 데이터 요약
요약_vintage_1Y = pd.DataFrame(df_1Y.iloc[-1]).reset_index()
요약_vintage_1Y.columns = ['요약명칭', '1Y']

print('롤링 1년 수익률===============', df_1Y)


# 주간 변동성 연간화 (과거 1년 동안의 데이터로 계산)
df_W = PV1.pct_change(periods=7).dropna(how='all')
df_W_interval = df_W[::-1].iloc[::7].dropna(how='all')[::-1]

Vol_1Y = df_W_interval.rolling(window=52).std() * np.sqrt(52)
Vol_1Y = Vol_1Y.asfreq('D').ffill().dropna(how='all')
print("Vol_1Y======================", Vol_1Y)



#그래프 표시기간 영역 별도 지정
start2 = pd.to_datetime(start_date, format='%Y%m%d') + pd.DateOffset(years=1)


# 빈티지에 따른 데이터를 추출하고, 1년 이상의 데이터만 선택하여 처리하는 함수
def process_volatility_data(Vol_1Y, start_date, vintage_year):
    vintage_cols = [col for col in Vol_1Y.columns if vintage_year in col]
    vintage_volatility = Vol_1Y[vintage_cols]

    vintage_volatility = vintage_volatility[vintage_volatility.index > start2]
    
    # 변동성 데이터 요약
    요약_vintage_volatility = pd.DataFrame(vintage_volatility.iloc[-1]).reset_index()
    요약_vintage_volatility.columns = ['요약명칭', '변동성']

    
    # 변동성 데이터 요약 열에 1Y 수익률 열 추가
    요약_vintage_volatility['수익률'] = 요약_vintage_1Y['1Y']  # 수익률은 변동성과 다른 값으로 계산해야 합니다. 이 부분은 실제 수익률 데이터로 교체해야 합니다.

    # '포커스'와 'TRP' 필터링
    요약_vintage_volatility['요약명칭'] = 요약_vintage_volatility['요약명칭'].astype(str)
    vintage_volatility_포커스 = 요약_vintage_volatility[요약_vintage_volatility['요약명칭'].str.contains('포커스')]
    vintage_volatility_TRP = 요약_vintage_volatility[요약_vintage_volatility['요약명칭'].str.contains('TRP')]
    vintage_volatility_전체 = 요약_vintage_volatility[~요약_vintage_volatility['요약명칭'].str.contains('포커스|TRP')]

    return vintage_volatility, 요약_vintage_volatility, vintage_volatility_포커스, vintage_volatility_TRP, vintage_volatility_전체


df_cum = (1 + df_R).cumprod() - 1
df_cum.replace([np.inf, -np.inf], np.nan, inplace=True)
df_cum.fillna(0, inplace=True)

# Sharpe Ratio 계산
def calculate_sharpe_ratio(R_1Y, Vol_1Y, risk_free_rate=0.03):
    Sharpe_Ratio = pd.DataFrame()
    common_columns = R_1Y.columns.intersection(Vol_1Y.columns)

    for col in common_columns:
        Sharpe_Ratio[col] = (R_1Y[col] - risk_free_rate) / Vol_1Y[col]
    Sharpe_1Y_rank = 101 - Sharpe_Ratio.rank(axis=1, ascending=True, pct=True) * 100

    return Sharpe_Ratio, Sharpe_1Y_rank

# YTD 계산
def calculate_YTD(PV1):
    Year_End = f"{pd.to_datetime(PV1.index[-1]).year - 1}-12-31"
    R_YTD = (PV1 / PV1.loc[Year_End]) - 1
    R_YTD.replace([np.inf, -np.inf], np.nan, inplace=True)
    R_YTD.fillna(0, inplace=True)
    R_YTD_rank = 101 - R_YTD.rank(axis=1, ascending=True, pct=True) * 100
    return R_YTD, R_YTD_rank



# 대시 앱 생성
app = dash.Dash(__name__)
app.title = 'TDF_포커스'
server = app.server

# 대시 앱 레이아웃 설정
app.layout = html.Div(
    style={'width': '65%', 'margin-left': '2'},
    children=[
        html.H3('TDF_포커스', style={'textAlign': 'center'}),
        
        dcc.Dropdown(
            id='vintage-dropdown',
            options=[{'label': f'{vintage}', 'value': vintage} for vintage in vintage_list],
            value='2050',  # 기본 선택값
            style={'width': '30%'},  # 드롭다운 너비 설정
            clearable=False
        ),

        # 요약명칭 입력란 추가
        dcc.Input(
            id='summary-names-input',
            type='text',
            placeholder="Enter summary names, separated by commas",
            value='한투, 미래, 삼성, KB',  # 기본값 설정
            style={'width': '30%'},
        ),
        html.Button('Update Graphs', id='update-button', n_clicks=0),

        dcc.Graph(id='line-graph'),
        dcc.Graph(id='scatter-graph'),
        dcc.Graph(id='volatility-graph'),
        dcc.Graph(id='return-graph'),
        dcc.Graph(id='sharpe-graph'),
        dcc.Graph(id='sharpe-rank-graph'),
        dcc.Graph(id='ytd-rank-graph'),
        dcc.Graph(id='drawdown-graph'),
    ]
)



# 콜백 함수
@app.callback(
    [
        Output('line-graph', 'figure'),
        Output('scatter-graph', 'figure'),
        Output('volatility-graph', 'figure'),
        Output('return-graph', 'figure'),
        Output('sharpe-graph', 'figure'),
        Output('sharpe-rank-graph', 'figure'),
        Output('ytd-rank-graph', 'figure'),
        Output('drawdown-graph', 'figure'),
    ],
    [Input('vintage-dropdown', 'value'),
     Input('summary-names-input', 'value')]
)
def update_graphs(selected_vintage, summary_names):
    summary_list = [name.strip() for name in summary_names.split(',')]
    
    # 선택된 빈티지와 요약명칭 필터 적용
    filtered_columns = [col for col in PV1.columns if selected_vintage in col and any(name in col for name in summary_list)]

    # 특정 기간 이후의 데이터만 선택하도록 start2 설정
    start2 = pd.to_datetime(start_date, format='%Y%m%d') + pd.DateOffset(years=1)

    vintage_volatility, 요약_vintage_volatility, vintage_volatility_포커스, vintage_volatility_TRP, vintage_volatility_전체 = process_volatility_data(Vol_1Y, start_date, selected_vintage)

    # 특정 기간 이후의 데이터만 필터링
    R_1Y_vintage = df_1Y[[col for col in df_1Y.columns if selected_vintage in col and any(name in col for name in summary_list)]]
    R_1Y_vintage = R_1Y_vintage[R_1Y_vintage.index > start2]

    Sharpe_Ratio_vintage, Sharpe_1Y_rank_vintage = calculate_sharpe_ratio(R_1Y_vintage, vintage_volatility)
    Sharpe_Ratio_vintage = Sharpe_Ratio_vintage[Sharpe_Ratio_vintage.index > start2]
    Sharpe_1Y_rank_vintage = Sharpe_1Y_rank_vintage[Sharpe_1Y_rank_vintage.index > start2]

    R_YTD, R_YTD_rank = calculate_YTD(PV1)

    # Line Graph
    line_graph = {
        'data': [
            go.Scatter(
                x=df_cum.index,
                y=df_cum[col],
                mode='lines',
                name=col,
                line=dict(
                    width=4 if '한투' in col else 1, 
                    color='#3762AF' if '포커스' in col else ('#630' if '한투' in col else None),
                    dash='dot' if 'UH' in col else 'solid'  # "UH" 포함시 점선
                ),
            ) for col in filtered_columns
        ],
        'layout': {
            'title': f'ITD 수익률 (TDF {selected_vintage})',
            'xaxis': {'title': 'Date', 'tickformat': '%Y%m%d'},
            'yaxis': {'title': 'Return', 'tickformat': '.0%'},
        }
    }


    # Scatter Graph
    scatter_graph = {
        'data': [
            go.Scatter(
                x=vintage_volatility_포커스['변동성'], 
                y=vintage_volatility_포커스['수익률'],
                mode='markers+text',
                name='ETF 포커스',
                marker=dict(color='#3762AF', size=18),
                hoverinfo='text',
                text=vintage_volatility_포커스['요약명칭'],
                textposition='bottom center',  # 텍스트를 마커의 오른쪽에 배치
                textfont=dict(color='rgba(55, 98, 175, 0.7)'),  # 텍스트의 불투명도를 0.5로 설정
            ),
            go.Scatter(
                x=vintage_volatility_TRP['변동성'], 
                y=vintage_volatility_TRP['수익률'],
                mode='markers+text',
                name='TRP',
                marker=dict(color='#630', size=18),
                hoverinfo='text',
                text=vintage_volatility_TRP['요약명칭'],
                textposition='bottom center',  # 텍스트를 마커의 오른쪽에 배치
                textfont=dict(color='rgba(102, 51, 0, 0.7)'),  # 텍스트의 불투명도를 0.5로 설정


            ),
            go.Scatter(
                x=vintage_volatility_전체['변동성'], 
                y=vintage_volatility_전체['수익률'],
                mode='markers+text',
                name='전체 TDF',
                marker=dict(color='#808080', size=12, opacity=0.3),
                hoverinfo='text',
                text=vintage_volatility_전체['요약명칭'],
                textposition='middle right',  # 텍스트를 마커의 오른쪽에 배치
                textfont=dict(color='rgba(128, 128, 128, 0.3)'),  # 텍스트의 불투명도를 0.5로 설정
            ),
        ],
        'layout': {
            'title': f'1Y 위험대비 수익률 (TDF {selected_vintage})',
            'xaxis': {
                'title': '변동성', 
                'tickformat': '.0%', 
                'range': [0, vintage_volatility_전체['변동성'].max()+0.01],  # x축 범위를 0부터 max까지 설정
                'autorange': True
            },
            'yaxis': {
                'title': 'Return(YTD)', 
                'tickformat': '.0%', 
                'range': [
                    min(vintage_volatility_전체['수익률'].min(), 0),  # y축 범위를 min값(0보다 작은 경우 포함)부터 max까지 설정
                    vintage_volatility_전체['수익률'].max()+0.01
                ],
                'autorange': False
            },
        }
    }


    # Volatility Graph
    volatility_graph = {
        'data': [
            go.Scatter(
                x=vintage_volatility.index,
                y=vintage_volatility[col],
                mode='lines',
                name=col,
                line=dict(
                    width=4 if '한투' in col else 1, 
                    color='#3762AF' if '포커스' in col else ('#630' if '한투' in col else None),
                    dash='dot' if 'UH' in col else 'solid'  # "UH" 포함시 점선
                ),
            ) for col in filtered_columns
        ],
        'layout': {
            'title': f'1Y Rolling Volatility (TDF {selected_vintage})',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Volatility', 'tickformat': '.0%'},
        }
    }

    # Return Graph
    return_graph = {
        'data': [
            go.Scatter(
                x=R_1Y_vintage.index,
                y=R_1Y_vintage[col],
                mode='lines',
                name=col,
                line=dict(
                    width=4 if '한투' in col else 1, 
                    color='#3762AF' if '포커스' in col else ('#630' if '한투' in col else None),
                    dash='dot' if 'UH' in col else 'solid'  # "UH" 포함시 점선
                ),
            ) for col in filtered_columns
        ],
        'layout': {
            'title': f'1Y Rolling Return (TDF {selected_vintage})',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Return', 'tickformat': '.0%'},
        }
    }

    # Sharpe Ratio Graph
    sharpe_graph = {
        'data': [
            go.Scatter(
                x=Sharpe_Ratio_vintage.index,
                y=Sharpe_Ratio_vintage[col],
                mode='lines',
                name=col,
                line=dict(
                    width=4 if '한투' in col else 1, 
                    color='#3762AF' if '포커스' in col else ('#630' if '한투' in col else None),
                    dash='dot' if 'UH' in col else 'solid'  # "UH" 포함시 점선
                ),
            ) for col in filtered_columns
        ],
        'layout': {
            'title': f'1Y Rolling Sharpe Ratio (TDF {selected_vintage})',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Sharpe Ratio', 'tickformat': '.0f'},
        }
    }

    # Sharpe Rank Graph
    sharpe_rank_graph = {
        'data': [
            go.Scatter(
                x=Sharpe_1Y_rank_vintage.index,
                y=Sharpe_1Y_rank_vintage[col],
                mode='lines',
                name=col,
                line=dict(
                    width=4 if '한투' in col else 1, 
                    color='#3762AF' if '포커스' in col else ('#630' if '한투' in col else None),
                    dash='dot' if 'UH' in col else 'solid'  # "UH" 포함시 점선
                ),
            ) for col in filtered_columns
        ],
        'layout': {
            'title': f'1Y Rolling Sharpe Ratio %Ranking (TDF {selected_vintage})',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Rank (%)', 'tickformat': '.0f', 'range': [100, 1]},
        }
    }

    # YTD Rank Graph
    ytd_rank_graph = {
        'data': [
            go.Scatter(
                x=R_YTD_rank.index,
                y=R_YTD_rank[col],
                mode='lines',
                name=col,
                line=dict(
                    width=4 if '한투' in col else 1, 
                    color='#3762AF' if '포커스' in col else ('#630' if '한투' in col else None),
                    dash='dot' if 'UH' in col else 'solid'  # "UH" 포함시 점선
                ),
            ) for col in filtered_columns
        ],
        'layout': {
            'title': f'YTD Return %Ranking (TDF {selected_vintage})',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Rank (%)', 'tickformat': '.0f', 'range': [100, 1]},
        }
    }

    # Drawdown Graph
    cumulative_returns = (1 + df_R).cumprod()
    max_cumulative_returns = cumulative_returns.rolling(window=365).max()
    DD = (cumulative_returns / max_cumulative_returns) - 1
    DD_vintage = DD[[col for col in DD.columns if selected_vintage in col and any(name in col for name in summary_list)]].dropna(how='all')

    drawdown_graph = {
        'data': [
            go.Scatter(
                x=DD_vintage.index,
                y=DD_vintage[col],
                mode='lines',
                name=col,
                line=dict(
                    width=4 if '한투' in col else 1, 
                    color='#3762AF' if '포커스' in col else ('#630' if '한투' in col else None),
                    dash='dot' if 'UH' in col else 'solid'  # "UH" 포함시 점선
                ),
            ) for col in DD_vintage.columns
        ],
        'layout': {
            'title': f'1Y Rolling Drawdown (TDF {selected_vintage})',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Drawdown', 'tickformat': '.0%'},
        }
    }

    return line_graph, scatter_graph, volatility_graph, return_graph, sharpe_graph, sharpe_rank_graph, ytd_rank_graph, drawdown_graph


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
