import FinanceDataReader as fdr
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt

import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd

# 엑셀 파일 경로
save_path = 'C:\Covenant\data\TDF_holdings.xlsx'

# 오늘 날짜를 변수에 할당
today = datetime.now()
today_str = today.strftime('%Y-%m-%d')
print(today_str)

# 시작 날짜 설정
Start = today - timedelta(days=1000)
End = today - timedelta(days=0)

def fetch_and_save_data(sheet_name, 종목코드_list):
    data_frames = []
    for 종목코드 in tqdm(종목코드_list, desc=f'Fetching data for {sheet_name}'):
        try:
            df = fdr.DataReader(종목코드, Start, End)['Close'].rename(종목코드)
            data_frames.append(df)
        except Exception as e:
            print(f"Error fetching data for 종목코드 '{종목코드}': {e}")
    combined_df = pd.concat(data_frames, axis=1)
    
    # 인덱스를 날짜 형식으로 변경
    combined_df.index = pd.to_datetime(combined_df.index)
    # 인덱스 날짜 형식을 YY-MM-DD로 포맷팅
    combined_df.index = combined_df.index.strftime('%y-%m-%d')
    
    with pd.ExcelWriter(save_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        combined_df.to_excel(writer, sheet_name=sheet_name, index=True)  # 인덱스 포함
    print(f'데이터가 {save_path}의 {sheet_name} 시트에 저장되었습니다.')

# List1 데이터 가져오기
df_list1 = pd.read_excel(save_path, sheet_name='List1', usecols=[0], names=['종목코드'])
종목코드_list1 = df_list1['종목코드'].astype(str).tolist()
fetch_and_save_data('Price1', 종목코드_list1)

# 표준화 합산 점수 계산 함수 정의
def calculate_score(df):
    # 인덱스를 날짜 형식으로 변경
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
    
    # 3개월 rolling return 계산
    rolling_return = df.pct_change(periods=63).fillna(0)

    # 3개월 변동성 계산
    df_W_R = df.pct_change(periods=7)
    df_W_interval = df_W_R.resample('W').last()
    volatility = df_W_interval.rolling(window=4 * 3).std() * (52 ** 0.5)

    # 3개월 DD (Drawdown) 계산
    cumulative_returns = (1 + df.pct_change(fill_method=None)).cumprod()
    max_cumulative_returns = cumulative_returns.rolling(window=63).max()    
    DD = (cumulative_returns / max_cumulative_returns) - 1

    # Winning mean gap 계산
    rolling_mean = rolling_return.mean(axis=0)
    winning_mean_gap = (rolling_return - rolling_mean)

    # 3개월 skewness의 일간 변화 계산
    skewness_change = df.rolling(window=63).apply(lambda x: skew(x.pct_change().fillna(0))).fillna(0).cumsum(axis=0)

    return rolling_return, volatility, DD, winning_mean_gap, skewness_change, cumulative_returns

# Price1 시트 데이터 가져오기
price1_df = pd.read_excel(save_path, sheet_name='Price1', index_col=0)

# 인덱스를 날짜 형식으로 변경하고 "YYYY-MM-DD"로 포맷팅
price1_df.index = pd.to_datetime(price1_df.index, format='%y-%m-%d').strftime('%Y-%m-%d')

# 각 열에 대해 점수 계산
rolling_return, volatility, DD, winning_mean_gap, skewness_change, cumulative_returns = calculate_score(price1_df)

# 시트 이름 설정
sheet_names = ['rolling_return', 'volatility', 'DD', 'winning_mean_gap', 'skewness_change', 'cumulative_returns']

# 결과를 각 시트에 저장
with pd.ExcelWriter(save_path, engine='openpyxl', mode='a') as writer:
    for i, data in enumerate([rolling_return, volatility, DD, winning_mean_gap, skewness_change, cumulative_returns]):
        sheet_name = sheet_names[i]
        if sheet_name in writer.book.sheetnames:
            writer.book.remove(writer.book[sheet_name])  # 이미 있는 시트 제거
        data.to_excel(writer, sheet_name=sheet_name, index=True)



# 예시 데이터 불러오기
rolling_return_tail = pd.read_excel(save_path, sheet_name='rolling_return', index_col=0).tail(500)
volatility_tail = pd.read_excel(save_path, sheet_name='volatility', index_col=0).tail(500)
DD_tail = pd.read_excel(save_path, sheet_name='DD', index_col=0).tail(500)
cumulative_returns_tail = pd.read_excel(save_path, sheet_name='cumulative_returns', index_col=0).tail(500)
skewness_change_tail = pd.read_excel(save_path, sheet_name='skewness_change', index_col=0).tail(500)
df_Sum_DD = DD_tail.sum(axis=1)
cumulative_returns_ACWI_BND = cumulative_returns_tail[['ACWI', 'BND']]






# Plotly 그래프 생성 함수
def create_plotly_graph(df, title):
    data = []
    for col in df.columns:
        trace = go.Scatter(x=df.index, y=df[col], mode='lines', name=col)
        data.append(trace)
    layout = dict(title=title, xaxis=dict(title='Date'), yaxis=dict(title='Values'))
    return {'data': data, 'layout': layout}




# Dash 앱 설정
app = dash.Dash(__name__)



# 스타일 설정 딕셔너리
graph_style = {
    'width': '70%', 
    'height': '600px', 
    'margin': 'auto',
    'display': 'flex',
    'justify-content': 'center',  # 가로 방향 가운데 정렬
    'text-align': 'center',
    'align-items': 'center'  # 세로 방향 가운데 정렬
}



# 그래프 레이아웃 설정
app.layout = html.Div([
    html.H1("3M Return_based Analysis"),
    
    # 그래프 1: Rolling Return
    dcc.Graph(id='rolling-return-graph', figure=create_plotly_graph(rolling_return_tail, 'Rolling Return'), style=graph_style),
    
    # 그래프 2: Volatility
    dcc.Graph(id='volatility-graph', figure=create_plotly_graph(volatility_tail, 'Volatility'), style=graph_style),
    
    # 그래프 3: DD
    dcc.Graph(id='dd-graph', figure=create_plotly_graph(DD_tail, 'Drawdown'), style=graph_style),
    
    # 그래프 4: Cumulative Returns
    dcc.Graph(id='cumulative-returns-graph', figure=create_plotly_graph(cumulative_returns_tail, 'Cumulative Returns'), style=graph_style),
    
    # 그래프 5: Skewness Change
    dcc.Graph(id='skewness-change-graph', figure=create_plotly_graph(skewness_change_tail, 'Skewness Change'), style=graph_style),
    
    # 그래프 6: Sum of DD
    dcc.Graph(id='sum-of-dd-graph', figure=create_plotly_graph(pd.DataFrame(df_Sum_DD), 'Sum of Drawdown'), style=graph_style),
    
    # 그래프 7: Cumulative Returns (ACWI and BND)
    dcc.Graph(id='cumulative-returns-acwi-bnd-graph', figure=create_plotly_graph(cumulative_returns_ACWI_BND, 'Cumulative Returns (ACWI and BND)'), style=graph_style),
])



# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=True,host='0.0.0.0') 


    #http://192.168.194.140:8050
    
    # 노트북 192.168.219.101:8050