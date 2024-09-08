import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from pykrx import stock as pykrx
from scipy.optimize import minimize

# Dash 애플리케이션 생성
app = dash.Dash(__name__)

# 종목 코드 설정
stock_code = {
    "252670": "KODEX 200선물인버스2X", 
    "122630": "KODEX 레버리지", 
    "069500": "KODEX 200",
    "SOXX": "SOXX",
    "VUG": "VUG",
    "SPY": "SPY",
    "MAGS": "MAGS",
}

# 연도 선택을 위한 Dropdown 옵션 생성
current_year = datetime.today().year
year_options = [{'label': str(year), 'value': year} for year in range(current_year - 10, current_year + 1)]

# SMA 윈도우별 변동성 계산 함수
def calculate_volatility(df_price, stock_code, start, end, sma_windows):
    volatilities = {}
    for window in sma_windows:
        df_price['SMA'] = df_price[stock_code].rolling(window=window).mean()
        df_price['Strategy_Return'] = df_price[stock_code].pct_change().fillna(0)
        volatility = df_price['Strategy_Return'].std()
        volatilities[window] = volatility
    return min(volatilities, key=volatilities.get)  # 최소 변동성 SMA 윈도우 반환

# 기본 SMA 윈도우 계산
def get_optimal_sma_window(stock_code, start, end):
    df_price = fetch_data(stock_code, start, end)
    sma_windows = range(5, 51)  # SMA 윈도우 범위 설정 (5일 ~ 50일)
    optimal_sma_window = calculate_volatility(df_price, stock_code, start, end, sma_windows)
    return optimal_sma_window

# Layout 정의 (Dropdown 및 Loading 컴포넌트 추가)
app.layout = html.Div(
    style={
        'margin': 'auto',
        'width': '65%', 
        'height': '100vh', 
        'display': 'flex', 
        'flexDirection': 'column'  # 전체 레이아웃 크기 설정
    },
   
    children=[
        html.H1("Covenant Model - AI Signal Trading", style={'textAlign': 'center'}),  # 중앙 정렬
       
       # 매수/매도 텍스트 출력 영역 추가
        html.Div(id='buy-sell-signal', style={'fontSize': 24, 'textAlign': 'center', 'color': 'green'}),
       
        html.Label("Select Stock:"),
        dcc.Dropdown(
            id='stock-dropdown',
            options=[{'label': name, 'value': code} for code, name in stock_code.items()],
            value='069500',  # 기본 종목
            style={'width': '50%',},
        ),
        
        html.Label("Select Start Year:"),
        dcc.Dropdown(
            id='year-dropdown',
            options=year_options,
            value=current_year,
            style={'width': '40%',},
        ),

        html.Label("Time Window for Prediction:"),
        dcc.Input(
            id='sma-window', 
            type='number', 
            style={'width': '10%',},
            value=5, min=0, step=5),
        
        # Loading 컴포넌트로 그래프 로딩 표시 (전체 크기 조정)
        dcc.Loading(
            id="loading-graph",
            type="default",  # 로딩 스타일 설정 (스피너)
            children=html.Div([
                dcc.Graph(id='cumulative-return-graph', style={'flex': '1'}),  # 누적 수익률 그래프
                dcc.Graph(id='signal-graph', style={'flex': '1'})  # Weighted Signal Final 그래프
            ], style={'display': 'flex', 'flexDirection': 'row'})  # 그래프 두 개를 나란히 표시
        ),

        # Loading 컴포넌트로 최적화된 가중치 로딩 표시 (전체 크기 조정)
        dcc.Loading(
            id="loading-text",
            type="default",  # 로딩 스타일 설정 (스피너)
            children=html.Div(
                id='optimal-weights-output',
                style={'height': '100px'}  # 텍스트 가로 100%, 세로 100px
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
            df_price = fdr.DataReader(code, start, end)['Close']
        
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
        return -df_price['Cum_Return'].iloc[-1]  # 누적 수익률의 음수 값을 반환

    initial_weights = [1/3, 1/3, 1/3]
    constraints = {'type': 'eq', 'fun': lambda weights: 1 - sum(weights)}
    bounds = [(0, 1), (0, 1), (0, 1)]

    result = minimize(calc_strategy, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

@app.callback(
    [Output('cumulative-return-graph', 'figure'),
     Output('signal-graph', 'figure'),
     Output('optimal-weights-output', 'children'),
     Output('buy-sell-signal', 'children'),  # 매수/매도 텍스트 추가
     Output('sma-window', 'value')],
    [Input('stock-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_graph(stock_code, selected_year):
    # 선택된 연도에 따른 시작 날짜 설정
    start = datetime(selected_year, 1, 1).strftime('%Y-%m-%d')
    end = datetime.today().strftime('%Y-%m-%d')

    # 최적 SMA 윈도우 계산
    optimal_sma_window = get_optimal_sma_window(stock_code, start, end)

    # 데이터 가져오기 및 전처리
    df_price = fetch_data(stock_code, start, end)
    
    if df_price is None:
        return {}, {}, "Error fetching data.", "데이터 없음", optimal_sma_window
    
    # 'Daily_Return' 계산
    df_price['Daily_Return'] = df_price[stock_code].pct_change().fillna(0)

    # Bollinger Bands 시그널 개선
    df_price['MA20'] = df_price[stock_code].rolling(window=20).mean()
    df_price['Upper'] = df_price['MA20'] + (df_price[stock_code].rolling(window=20).std() * 2)
    df_price['Lower'] = df_price['MA20'] - (df_price[stock_code].rolling(window=20).std() * 2)
    df_price['BB_Upper_Slope'] = np.gradient(df_price['Upper'])
    df_price['BB_Lower_Slope'] = np.gradient(df_price['Lower'])
    
    df_price['BB_Signal'] = 0
    df_price.loc[(df_price[stock_code] > df_price['Upper']) & (df_price['BB_Upper_Slope'] > 0), 'BB_Signal'] = 1  # 상승 추세에서 상단 밴드 돌파 시 매수
    df_price.loc[(df_price[stock_code] < df_price['Lower']) & (df_price['BB_Lower_Slope'] < 0), 'BB_Signal'] = -1  # 하락 추세에서 하단 밴드 돌파 시 매도

    # SMA 기울기 및 가속도 계산
    df_price['SMA'] = df_price[stock_code].rolling(window=optimal_sma_window).mean()  # 최적 SMA 윈도우 사용
    df_price['Slope'] = np.degrees(np.arctan(np.gradient(df_price['SMA'])))
    df_price['Slope_Acceleration'] = np.gradient(df_price['Slope'])
    
    df_price['SMA_Signal'] = 0
    df_price.loc[(df_price['Slope'] < -70) & (df_price['Slope_Acceleration'] < 0), 'SMA_Signal'] = -1  # 급격한 하락 시 매도
    df_price.loc[(df_price['Slope'] > -30) & (df_price['Slope_Acceleration'] > 0), 'SMA_Signal'] = 1  # 급격한 상승 시 매수

    # LSTM 시그널 생성 (실제 학습)
    X, y, scaler = create_lstm_data(df_price, stock_code)
    lstm_model = create_lstm_model(X.shape)
    lstm_model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # 예측 값 생성 및 LSTM 시그널
    predicted = lstm_model.predict(X)
    predicted = scaler.inverse_transform(predicted)
    
    # LSTM 예측 값을 패딩하여 데이터프레임 길이와 맞춤
    padding_length = len(df_price) - len(predicted)
    lstm_signals_resized = np.concatenate([np.zeros(padding_length), np.sign(predicted.flatten())])
    
    # LSTM 시그널을 데이터프레임에 추가
    df_price['LSTM_Signal'] = lstm_signals_resized

    # 최적 가중치 계산 (최적화 과정에서 누적 수익률 최대화)
    optimal_weights = optimize_weights(df_price)

    # 최적화된 가중치로 전략 계산
    df_price['Weighted_Signal'] = (optimal_weights[0] * df_price['BB_Signal'] +
                                   optimal_weights[1] * df_price['LSTM_Signal'] +
                                   optimal_weights[2] * df_price['SMA_Signal'])
    
    df_price['Weighted_Signal_Final'] = df_price['Weighted_Signal'].apply(lambda x: 1 if x > 0 else 0)
    df_price['Strategy_Return'] = df_price['Daily_Return'] * df_price['Weighted_Signal_Final']
    df_price['Cum_Return'] = (1 + df_price['Strategy_Return']).cumprod() - 1
    
    # 마지막 시그널을 기준으로 매수/매도 텍스트 설정
    last_signal = df_price['Weighted_Signal_Final'].iloc[-1]
    signal_text = "매수" if last_signal == 1 else "매도"

    # Plotly 그래프 생성 - Cumulative Return
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(x=df_price.index, y=df_price['Cum_Return'], mode='lines', name='Cumulative Return'))
    fig_cumulative.update_layout(
        title=f'Cumulative Return for {stock_code}',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        yaxis={'tickformat': '.0%'}  # Y축을 퍼센트로 표시
    )

    # Plotly 그래프 생성 - Weighted Signal Final
    fig_signal = go.Figure()
    fig_signal.add_trace(
        go.Scatter(
            x=df_price.index, 
            y=df_price['Weighted_Signal_Final'], 
            mode='lines', name='Weighted Signal Final'))
    fig_signal.update_layout(
        title=f'Weighted Signal Final for {stock_code}',
        xaxis_title='Date',
        yaxis_title='Signal (0 or 1)'
    )

    # 최적화된 가중치 출력
    weights_text = f"Optimal Weights: Bollinger: {optimal_weights[0]:.2f}, LSTM: {optimal_weights[1]:.2f}, SMA: {optimal_weights[2]:.2f}"

    return fig_cumulative, fig_signal, weights_text, signal_text, optimal_sma_window

# 애플리케이션 실행
if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0")

