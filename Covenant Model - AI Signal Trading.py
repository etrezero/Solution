import FinanceDataReader as fdr
import yfinance as yf
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from pykrx import stock as pykrx
from scipy.optimize import minimize
import requests



# Dash 애플리케이션 생성
app = dash.Dash(__name__)


# 종목 코드 설정
stock_code = {
    "252670": "KODEX 200선물인버스2X", 
    "122630": "KODEX 레버리지", 
    "069500": "KODEX 200",
    "114800": "KODEX 인버스",
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "005380": "현대차",
    "207940": "삼성바이오로직스",
    "035420": "NAVER",
    "101360": "에코앤드림",
    "044450": "KSS해운",
    
    "SOXX": "SOXX",
    "VUG": "VUG",
    "SPY": "SPY",
    "MAGS": "MAGS",
}

# 연도 선택을 위한 Dropdown 옵션 생성
current_year = datetime.today().year
year_options = [{'label': str(year), 'value': year} for year in range(current_year - 10, current_year + 1)]

# SMA 윈도우별 변동성 계산 함수
def calculate_volatility(df_price, stock_code_str, sma_windows):
    if df_price is None or df_price.empty:
        return None
    volatilities = {}
    for window in sma_windows:
        df_price['SMA'] = df_price[stock_code_str].rolling(window=window).mean()
        df_price['Strategy_Return'] = df_price[stock_code_str].pct_change().fillna(0)
        volatility = df_price['Strategy_Return'].std()
        volatilities[window] = volatility
    return min(volatilities, key=volatilities.get) if volatilities else None

# 기본 SMA 윈도우 계산 함수
def get_optimal_sma_window(stock_code_str, start, end):
    df_price = fetch_data(stock_code_str, start, end)
    if df_price is None:
        return None
    sma_windows = range(5, 51)  # SMA 윈도우 범위 설정
    optimal_sma_window = calculate_volatility(df_price, stock_code_str, sma_windows)
    return optimal_sma_window

# Layout 정의 (Dropdown 및 Loading 컴포넌트 추가)
app.layout = html.Div(
    style={'margin': 'auto', 'width': '65%', 'height': '100vh', 'display': 'flex', 'flexDirection': 'column'},
    children=[
        html.H1(f"Covenant AI Signal {datetime.today().strftime('%Y-%m-%d')}", style={'textAlign': 'center'}),        
       
        html.Div(
            id='buy-sell-signal', 
            style={'fontSize': 24, 'textAlign': 'center', 'color': 'green'}),
       
        html.Label("Select Stock:"),
        dcc.Dropdown(
            id='stock-dropdown',
            options=[{'label': name, 'value': code} for code, name in stock_code.items()],
            value='069500',
            style={'width': '50%'},
        ),
        html.Label("Select Start Year:"),
        dcc.Dropdown(
            id='year-dropdown',
            options=year_options,
            value=current_year,
            style={'width': '40%'},
        ),
        html.Label("Time Window for Prediction:"),
        
        dcc.Input(
            id='sma-window', 
            type='number', 
            style={'width': '10%'},
            value=5, min=0, step=5),
        
        dcc.Loading(
            id="loading-graph",
            type="default",
            children=html.Div([
                dcc.Graph(id='cumulative-return-graph', style={'flex': '1'}),
                dcc.Graph(id='signal-graph', style={'flex': '1'})
            ], style={'display': 'flex', 'flexDirection': 'row'})
        ),
        dcc.Loading(
            id="loading-text",
            type="default",
            children=html.Div(
                id='optimal-weights-output',
                style={'height': '100px'}
            )
        )
    ]
)


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
            df_price = yf_data.history(start=start, end=end)['Close'].rename(code)

        df_price = pd.DataFrame(df_price)
        df_price.columns = [code]
        return df_price
    
    except Exception as e:
        print(f"Error fetching data for {code}: {e}")
        return None


# LSTM 데이터를 생성하는 함수
def create_lstm_data(df_price, target_col, time_steps=50):
    scaler = MinMaxScaler()
    df_price_scaled = scaler.fit_transform(df_price[[target_col]])

    X, y = [], []
    for i in range(len(df_price_scaled) - time_steps):
        X.append(df_price_scaled[i:i+time_steps, 0])
        y.append(df_price_scaled[i+time_steps, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# LSTM 모델 생성 함수
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(input_shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 최적화 함수 정의
def optimize_weights(df_price):
    def calc_strategy(weights):
        df_price['Weighted_Signal'] = (weights[0] * df_price['BB_Signal'] +
                                       weights[1] * df_price['LSTM_Signal'] +
                                       weights[2] * df_price['SMA_Signal'])
        df_price['Weighted_Signal_Final'] = df_price['Weighted_Signal'].apply(lambda x: 1 if x > 0 else 0)
        df_price['Strategy_Return'] = df_price['Daily_Return'] * df_price['Weighted_Signal_Final'].shift(1).fillna(0)
        df_price['Cum_Return'] = (1 + df_price['Strategy_Return']).cumprod() - 1
        return -df_price['Cum_Return'].iloc[-1]

    initial_weights = [1/3, 1/3, 1/3]
    constraints = {'type': 'eq', 'fun': lambda weights: 1 - sum(weights)}
    bounds = [(0, 1), (0, 1), (0, 1)]

    result = minimize(calc_strategy, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x





@app.callback(
    [Output('cumulative-return-graph', 'figure'),
     Output('signal-graph', 'figure'),
     Output('optimal-weights-output', 'children'),
     Output('buy-sell-signal', 'children'),
     Output('sma-window', 'value')],
    [Input('stock-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_graph(stock_code, selected_year):
    # 시작 및 끝 날짜 설정
    start = datetime(selected_year, 1, 1).strftime('%Y-%m-%d')
    end = datetime(selected_year, 12, 31).strftime('%Y-%m-%d')

    optimal_sma_window = get_optimal_sma_window(stock_code, start, end)
    if optimal_sma_window is None:
        return {}, {}, f"Error: No data for {stock_code}", "데이터 없음", None

    df_price = fetch_data(stock_code, start, end)
    if df_price is None or df_price.empty:
        return {}, {}, f"Error: No data for {stock_code}", "데이터 없음", None

    # 일간 수익률 계산
    df_price['Daily_Return'] = df_price[stock_code].pct_change().fillna(0)

    # Bollinger Bands 시그널
    df_price['MA20'] = df_price[stock_code].rolling(window=20).mean()
    df_price['Upper'] = df_price['MA20'] + (df_price[stock_code].rolling(window=20).std() * 2)
    df_price['Lower'] = df_price['MA20'] - (df_price[stock_code].rolling(window=20).std() * 2)
    df_price['BB_Signal'] = 0
    df_price.loc[(df_price[stock_code] > df_price['Upper']), 'BB_Signal'] = 1
    df_price.loc[(df_price[stock_code] < df_price['Lower']), 'BB_Signal'] = -1

    # SMA 시그널
    df_price['SMA'] = df_price[stock_code].rolling(window=optimal_sma_window).mean()
    df_price['SMA_Signal'] = 0
    df_price.loc[(df_price['SMA'].diff() > 0), 'SMA_Signal'] = 1
    df_price.loc[(df_price['SMA'].diff() < 0), 'SMA_Signal'] = -1

    # LSTM 시그널 생성
    X, y, scaler = create_lstm_data(df_price, stock_code)
    lstm_model = create_lstm_model(X.shape)
    lstm_model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    predicted = lstm_model.predict(X)
    predicted = scaler.inverse_transform(predicted)
    padding_length = len(df_price) - len(predicted)
    lstm_signals_resized = np.concatenate([np.zeros(padding_length), np.sign(predicted.flatten())])

    df_price['LSTM_Signal'] = lstm_signals_resized

    # 최적화된 가중치로 최종 시그널 생성 (시그널 값은 1 또는 0)
    optimal_weights = optimize_weights(df_price)
    df_price['Weighted_Signal'] = (optimal_weights[0] * df_price['BB_Signal'] +
                                   optimal_weights[1] * df_price['LSTM_Signal'] +
                                   optimal_weights[2] * df_price['SMA_Signal'])

    # 시그널 값이 1 또는 0으로만 설정되도록 처리
    df_price['Weighted_Signal_Final'] = df_price['Weighted_Signal'].apply(lambda x: 1 if x > 0 else 0)

    # 포트폴리오 수익률 계산: 시그널이 1일 때만 일간 수익률 반영, 시그널이 0일 때는 수익률 0
    # 포트폴리오 수익률이 선택종목의 수익률을 넘을 수 없도록 설정
    df_price['Portfolio_Return'] = np.where(df_price['Weighted_Signal_Final'] == 1, df_price['Daily_Return'], 0)

    # 선택된 기간 내에서 전략 누적 수익률을 0으로 시작하도록 조정
    df_price['Portfolio_Cum_Return'] = (1 + df_price['Portfolio_Return']).cumprod() - 1
    df_price['Portfolio_Cum_Return'] -= df_price['Portfolio_Cum_Return'].iloc[0]  # 첫날 수익률을 0으로 설정

    # 선택된 종목의 누적 수익률 계산
    df_price['Stock_Cum_Return'] = (1 + df_price['Daily_Return']).cumprod() - 1
    df_price['Stock_Cum_Return'] -= df_price['Stock_Cum_Return'].iloc[0]  # 첫날 수익률을 0으로 설정

    last_signal = df_price['Weighted_Signal_Final'].iloc[-1]
    signal_text = "매수" if last_signal == 1 else "매도"




    # 전략과 선택한 종목의 누적 수익률=========================
    fig_cumulative = go.Figure()

    # 포트폴리오 누적 수익률
    fig_cumulative.add_trace(go.Scatter(x=df_price.index, y=df_price['Portfolio_Cum_Return'], mode='lines', name='Covenant Portfolio'))

    # 선택한 종목의 누적 수익률
    fig_cumulative.add_trace(go.Scatter(x=df_price.index, y=df_price['Stock_Cum_Return'], mode='lines', name=f'{stock_code}'))

    fig_cumulative.update_layout(
        title=f'Cumulative Return for {stock_code} ({selected_year})',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        yaxis={'tickformat': '.0%'},
        xaxis=dict(range=[df_price.index.min(), df_price.index.max()])  # 시작과 끝 날짜로 x-axis 범위 제한
    )

    # 시그널 그래프=======================================
    fig_signal = go.Figure()

    fig_signal.add_trace(
        go.Scatter(
            x=df_price.index, 
            y=df_price['Weighted_Signal_Final'], 
            mode='lines', name='Weighted Signal Final')
    )
    
    fig_signal.update_layout(
        title=f'Signal for {stock_code} ({selected_year})',
        xaxis_title='Date',
        yaxis_title='Signal (0 or 1)',
        xaxis=dict(range=[df_price.index.min(), df_price.index.max()])  # 시작과 끝 날짜로 x-axis 범위 제한
    )

    weights_text = f"Optimal Weights: Bollinger: {optimal_weights[0]:.2f}, LSTM: {optimal_weights[1]:.2f}, SMA: {optimal_weights[2]:.2f}"

    return fig_cumulative, fig_signal, weights_text, signal_text, optimal_sma_window



# 애플리케이션 실행
if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0")