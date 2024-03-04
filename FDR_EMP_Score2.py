import FinanceDataReader as fdr
from datetime import datetime, timedelta
import pandas as pd
import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# 엑셀 파일 경로
save_path = r'C:/Users/서재영/Documents/Python Scripts/data/FDR_EMP.xlsx'
path_Price1_df = r'C:/Users/서재영/Documents/Python Scripts/Price1_df.json'

# 오늘 날짜를 변수에 할당
today = datetime.now()
today_str = today.strftime('%Y-%m-%d')
print(today_str)

# 시작 날짜 설정
Start = today - timedelta(days=365)
End = today - timedelta(days=0)
period = 100

# JSON 파일로 저장
# Price1_df.to_json(path_Price1_df)  # orient='index'로 설정하여 인덱스를 키로 사용하여 저장

# path_Pric1_df.json 파일 읽기
Pric1_df = pd.read_json(path_Price1_df)

Pric1_df_R1D = Pric1_df.pct_change(periods=1)

# 3개월 rolling return 계산
rolling_return = (Pric1_df_R1D + 1).rolling(window=63).apply(lambda x: x.prod()) - 1
rolling_mean = rolling_return.mean(axis=0)

# 3개월 변동성 계산
Pric1_df_W_R = Pric1_df.pct_change(periods=7)
Pric1_df_W_R.index = pd.to_datetime(Pric1_df_W_R.index, format='%Y-%m-%d')
Pric1_df_W_interval = Pric1_df_W_R.resample('W').last()
volatility = Pric1_df_W_interval.rolling(window=12).std() * (52 ** 0.5)

# 3개월 DD (Drawdown) 계산
cumulative_returns = (1 + Pric1_df.pct_change(fill_method=None)).cumprod()
max_cumulative_returns = cumulative_returns.rolling(window=63).max()
DD = (cumulative_returns / max_cumulative_returns) - 1

# 3개월 skewness 계산
cumulative_skewness = Pric1_df_R1D.rolling(window=63).apply(lambda x: x.skew()).fillna(0).cumsum()

# period_tail
period_tail = 100
rolling_return_tail = rolling_return.tail(period_tail)
volatility_tail = volatility.tail(period_tail)
DD_tail = DD.tail(period_tail)
cumulative_skewness_tail = cumulative_skewness.tail(period_tail)
df_Sum_DD = DD_tail.sum(axis=1)

cumulative_returns_tail = (1 + Pric1_df_R1D.tail(period_tail)).cumprod()
cumulative_returns_VUG_ACWI = cumulative_returns_tail[['VUG', 'ACWI']]

cov_matrix = Pric1_df_W_interval.tail().cov()

# 데이터프레임으로 만들기
rolling_return_df = pd.DataFrame(rolling_return_tail)
volatility_df = pd.DataFrame(volatility_tail)
DD_df = pd.DataFrame(DD_tail)
cumulative_skewness_df = pd.DataFrame(cumulative_skewness_tail)
df_Sum_DD_df = pd.DataFrame(df_Sum_DD)
cumulative_returns_df = pd.DataFrame(cumulative_returns_tail)
cumulative_returns_VUG_ACWI_df = pd.DataFrame(cumulative_returns_VUG_ACWI)
cov_matrix_df = pd.DataFrame(cov_matrix)
formatted_cov_matrix_df = cov_matrix_df.apply(lambda x: x.map(lambda val: f"{val * 100:.4f}%" if pd.notnull(val) else ""))

# Plotly 그래프 생성 함수
def create_plotly_graph(df, title):
    data = []
    for col in df.columns:
        trace = go.Scatter(x=df.index, y=df[col], mode='lines', name=col)
        data.append(trace)
    layout = dict(title=title, xaxis=dict(title='Date'), yaxis=dict(title='Values'))
    return {'data': data, 'layout': layout}

# 데이터를 딕셔너리로 구성
data_dict = {
    'Rolling Return': rolling_return_df,
    'Volatility': volatility_df,
    'Drawdown': DD_df,
    'Cumulative Skewness': cumulative_skewness_df,
    'Sum of Drawdown': df_Sum_DD_df,
    'Cumulative Returns (VUG, ACWI)': cumulative_returns_VUG_ACWI_df,
}


# 각 데이터프레임의 마지막 행의 값을 저장할 딕셔너리 생성
last_row_values = {}

# 각 데이터프레임의 마지막 행의 값을 저장합니다.
for name, df in data_dict.items():
    # 데이터프레임의 마지막 행의 값을 가져와서 딕셔너리에 저장합니다.
    last_row_values[name] = df.iloc[-1]

# 딕셔너리를 데이터프레임으로 변환하고 인덱스를 날짜로 설정합니다.
table_df = pd.DataFrame(last_row_values).T
table_df.index.name = '종목코드'



# 데이터프레임 생성
table_df = pd.DataFrame(last_row_values, index=data_dict.keys())


# Dash 애플리케이션 생성
app = dash.Dash(__name__)

# 드롭다운 목록에 표시할 그래프 타이틀 리스트
graph_titles = list(data_dict.keys())

# 레이아웃 설정
app.layout = html.Div([
    html.H1(f"ETF Stat({today_str})"),  # 날짜 추가
   
    html.Div([
        dcc.Dropdown(
            id='graph-dropdown',
            options=[{'label': title, 'value': title} for title in graph_titles],
            value=graph_titles[0], # 기본값으로 첫 번째 그래프 선택
            style={
                'width': '50%', 
                'margin': 'auto',
                'float': 'right',
                'margin-right' : '10%',
                'z-index' : '9999',
                }
        ),
        
        dcc.Graph(id='graph-display', 
            style={
                'width': '80%', 
                'margin': 'auto',
                'margin-top': '1',
                'float': 'center',
                'z-index' : '1',
                })  # 그래프를 표시할 공간
    ]),

    
    html.Div([
        dash_table.DataTable(
            id='cov-matrix-table',
            columns=[{'name': col, 'id': col} for col in table_df.columns],
            data=table_df.to_dict('records'),
            style_header={'textAlign': 'center'},  # 헤더 스타일 가운데 정렬
            style_cell={'textAlign': 'center'},  # 셀 스타일 가운데 정렬
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
                'lineHeight': '15px',
                'font-size': '15px'
            }
        ),
    ], style={
        'width': '80%', 
        'float': 'center',
        'margin': 'auto',
        }),
    
])



# 그래프를 표시할 위치에 대한 콜백 함수
@app.callback(
    Output('graph-display', 'figure'),  # 그래프를 표시할 위치를 지정합니다.
    [Input('graph-dropdown', 'value')]  # 선택한 그래프 타이틀을 입력으로 받습니다.
)
def update_graph(selected_title):
    # 선택한 그래프 타이틀에 해당하는 그래프를 생성합니다.
    return create_plotly_graph(data_dict[selected_title], selected_title)

# 애플리케이션 실행
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
