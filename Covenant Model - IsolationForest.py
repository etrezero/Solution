import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import IsolationForest
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from pykrx import stock as pykrx
from scipy.optimize import minimize
import FinanceDataReader as fdr

# Dash 애플리케이션 생성
app = dash.Dash(__name__)

# 종목 코드 설정
code = {
    "252670": "KODEX 200선물인버스2X", 
    "122630": "KODEX 레버리지", 
    "069500": "KODEX 200",
    "114800": "KODEX 인버스",
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "005380": "현대차",
    "207940": "삼성바이오로직스",
    "035420": "NAVER",
    "068270": "셀트리온",
    
    "SOXX": "SOXX",
    "VUG": "VUG",
    "VTV": "VTV",
    "VEA": "VEA",
    "VWO": "VWO",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "MAGS": "MAGS",
    "DIA": "DIA",
    "AAPL": "AAPL",
    "TSLA": "TSLA",
    "NVDA": "NVDA",
    "GOOG": "GOOG",
    
    "BND": "Total Bond",
    "HYG": "High Yield Corporate Bond",
    "TMV": "20+Treasury Bear 3X",
    "TLT": "20+Treasury",
}

# 연도 선택을 위한 Dropdown 옵션 생성
current_year = datetime.today().year
year_options = [{'label': str(year), 'value': year} for year in range(current_year - 10, current_year + 1)]

# Layout 정의에 슬라이더 추가
app.layout = html.Div(
    style={
        'margin': 'auto', 
        'width': '75%', 
        'height': '100vh', 
        'display': 'flex', 
        'flexDirection': 'column'},
    children=[
        html.H1(f"Stock Anomaly Detection (Isolation Forest)", style={'textAlign': 'center'}),
        html.Label("Select Stock:"),
        dcc.Dropdown(
            id='stock-dropdown',
            options=[{'label': name, 'value': code} for code, name in code.items()],
            value='068270',
            style={'width': '40%'}
        ),
        html.Label("Select Start Year:"),
        dcc.Dropdown(
            id='year-dropdown',
            options=year_options,
            value=current_year,
            style={'width': '40%'}
        ),
        html.Label("Set Contamination Level (Anomaly Proportion):"),
        dcc.Slider(
            id='contamination-slider',
            min=0.01,  # 최소 contamination 값
            max=0.5,  # 최대 contamination 값
            step=0.01,
            value=0.1,  
            marks={i: f'{i:.2f}' for i in np.arange(0.01, 0.51, 0.05)},  # 슬라이더 표시 값
        ),
        dcc.Loading(
            id="loading-graph",
            type="default",
            children=html.Div([
                dcc.Graph(id='cumulative-return-graph', style={'flex': '1'}),
                dcc.Graph(id='anomaly-graph', style={'flex': '1'})
            ], style={'display': 'flex', 'flexDirection': 'row'})
        ),
    ]
)


# 데이터 가져오기 함수 수정
def fetch_data(code, start, end):
    try:
        if isinstance(code, int) or code.isdigit():
            if len(code) == 5:
                code = '0' + code
            # pykrx 데이터 사용
            df_price = pykrx.get_market_ohlcv_by_date(start, end, code)[['종가']]
            df_price.columns = ['Close']  # 열 이름을 'Close'로 변경
        else:
            # FinanceDataReader 데이터 사용
            df_price = fdr.DataReader(code, start, end)[['Adj Close']]
            df_price.columns = ['Close']  # 열 이름을 'Close'로 변경
        return df_price
    except Exception as e:
        print(f"Error fetching data for {code}: {e}")
        return None


# Isolation Forest로 이상치 감지 함수 수정
def detect_anomalies_isolation_forest(df, contamination=0.01):
    if 'Close' in df.columns:
        target_col = 'Close'
    elif 'Adj Close' in df.columns:
        target_col = 'Adj Close'
    elif '종가' in df.columns:
        target_col = '종가'
    else:
        raise KeyError("Close, Adj Close, or 종가 column not found in the DataFrame.")
    
    model = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly'] = model.fit_predict(df[[target_col]])
    anomalies = df[df['anomaly'] == -1]
    return anomalies

# 누적 수익률 계산 함수
def calculate_cumulative_return(df):
    if 'Close' in df.columns:
        target_col = 'Close'
    elif 'Adj Close' in df.columns:
        target_col = 'Adj Close'
    elif '종가' in df.columns:
        target_col = '종가'
    df['Daily_Return'] = df[target_col].pct_change().fillna(0)
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
    return df



# 콜백 함수 수정
@app.callback(
    [Output('cumulative-return-graph', 'figure'),
     Output('anomaly-graph', 'figure')],
    [Input('stock-dropdown', 'value'),
     Input('year-dropdown', 'value'),
     Input('contamination-slider', 'value')]  # contamination 값 추가
)
def update_graph(code, selected_year, contamination):
    start = f"{selected_year}-01-01"
    end = f"{selected_year}-12-31"

    # 주가 데이터 다운로드
    df = fetch_data(code, start, end)
    if df is None or df.empty:
        return {}, {}

    # 이상치 감지 (contamination 값 적용)
    anomalies = detect_anomalies_isolation_forest(df, contamination)

    # 누적 수익률 계산
    df = calculate_cumulative_return(df)

    # 누적 수익률 그래프
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Return'], mode='lines', name='Cumulative Return'))
    fig_cumulative.update_layout(
        title=f'Cumulative Return for {code} ({selected_year})',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        yaxis={'tickformat': '.0%'}
    )

    # 이상치 감지 그래프
    fig_anomalies = go.Figure()
    fig_anomalies.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price', line=dict(color='blue')))
    fig_anomalies.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Close'], mode='markers', name='Anomalies', marker=dict(color='red', size=10)))
    fig_anomalies.update_layout(
        title=f'Anomalies Detected in {code} ({selected_year}) - Contamination: {contamination:.2%}',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis={'tickformat': ',.0f'}
    )

    return fig_cumulative, fig_anomalies


# 애플리케이션 실행
if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0")
