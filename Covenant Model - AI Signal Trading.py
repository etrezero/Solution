import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.dash_table.Format import Format, Scheme
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
import dash_table

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


# Layout 정의 (Dropdown 및 Loading 컴포넌트 추가)
app.layout = html.Div(
    style={'margin': 'auto', 'width': '75%', 'height': '100vh', 'display': 'flex', 'flexDirection': 'column'},
    children=[
        html.H1(f"Covenant AI Signal {datetime.today().strftime('%Y-%m-%d')}", style={'textAlign': 'center'}),        html.Div(id='buy-sell-signal', style={'fontSize': 24, 'textAlign': 'center', 'color': 'green'}),
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
        ),
        
        
        dash_table.DataTable(
            id='data-table',
            style_cell={
                'textAlign': 'center',  # 텍스트 가운데 정렬
                'padding': '10px',
                'fontFamily': 'Arial',
                'fontSize': '15px',
                'minWidth': '80px',  # 모든 셀의 최소 너비 설정
                'maxWidth': '80px',  # 모든 셀의 최대 너비 설정
                'width': '80px',     # 모든 셀의 고정 너비 설정
            },
            style_table={'width': '100%', 'overflowX': 'auto'},  # 테이블 크기와 스크롤 설정
            page_size=30  # 페이지당 표시할 행 수를 30으로 설정
        ),


        
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
            df_price = fdr.DataReader(code, start, end)['Adj Close']
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
     
     Output('data-table', 'data'),  # 테이블 출력을 위한 Output 추가
     Output('data-table', 'columns')],  # 테이블 출력
    [Input('stock-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_graph(stock_code, selected_year):
    # 시작 및 끝 날짜 설정
    start = datetime(selected_year, 1, 1).strftime('%Y-%m-%d')
    end = datetime(selected_year, 12, 31).strftime('%Y-%m-%d')

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
    df_price['SMA'] = df_price[stock_code].rolling(window=5).mean()
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

    # LSTM 예측 결과를 1 또는 -1로 변환
    lstm_signals_resized = np.concatenate([np.zeros(padding_length), np.sign(predicted.flatten())])
    df_price['LSTM_Signal'] = lstm_signals_resized


    # 최적화된 가중치로 최종 시그널 생성 (시그널 값은 1 또는 0)
    optimal_weights = optimize_weights(df_price)
    df_price['Weighted_Signal'] = (optimal_weights[0] * df_price['BB_Signal'] +
                                   optimal_weights[1] * df_price['LSTM_Signal'] +
                                   optimal_weights[2] * df_price['SMA_Signal'])

    # 시그널 값이 1 또는 0으로만 설정되도록 처리
    df_price['Weighted_Signal_Final'] = df_price['Weighted_Signal'].apply(lambda x: 1 if x > 0 else 0)

    # 포트폴리오 수익률 계산: 전날 시그널이 1일 때만 다음 날 수익률 반영, 시그널이 0일 때는 수익률 0
    df_price['Portfolio_Return'] = np.where(df_price['Weighted_Signal_Final'].shift(1) == 1, df_price['Daily_Return'], 0)

    # 선택된 기간 내에서 전략 누적 수익률을 0으로 시작하도록 조정
    df_price['Portfolio_Cum_Return'] = (1 + df_price['Portfolio_Return']).cumprod() - 1
    df_price['Portfolio_Cum_Return'] -= df_price['Portfolio_Cum_Return'].iloc[0]  # 첫날 수익률을 0으로 설정

    # 선택된 종목의 누적 수익률 계산
    df_price['Stock_Cum_Return'] = (1 + df_price['Daily_Return']).cumprod() - 1
    df_price['Stock_Cum_Return'] -= df_price['Stock_Cum_Return'].iloc[0]  # 첫날 수익률을 0으로 설정

    last_signal = df_price['Weighted_Signal_Final'].iloc[-1]
    signal_text = "매수" if last_signal == 1 else "매도"

    # 누적 수익률 그래프 (전략과 선택한 종목의 누적 수익률을 함께 표시)
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


    # 시그널 그래프
    fig_signal = go.Figure()
    fig_signal.add_trace(
        go.Scatter(
            x=df_price.index, 
            y=df_price['Weighted_Signal_Final'], 
            mode='lines', name='Weighted Signal Final'))
    fig_signal.update_layout(
        title=f'Weighted Signal Final for {stock_code} ({selected_year})',
        xaxis_title='Date',
        yaxis_title='Signal (0 or 1)',
        xaxis=dict(range=[df_price.index.min(), df_price.index.max()])  # 시작과 끝 날짜로 x-axis 범위 제한
    )

    weights_text = f"Optimal Weights: Bollinger: {optimal_weights[0]:.2f}, LSTM: {optimal_weights[1]:.2f}, SMA: {optimal_weights[2]:.2f}"



    # 필요한 열만 선택하여 테이블로 표시
    # 인덱스를 'Date'라는 이름의 열로 변환하여 추가
    df_selected = df_price[['Daily_Return', 'BB_Signal', 'SMA_Signal', 'LSTM_Signal', 'Weighted_Signal_Final', 'Strategy_Return', 'Cum_Return']].reset_index()
    df_selected['날짜'] = pd.to_datetime(df_selected['날짜']).dt.strftime('%Y%m%d')  # 인덱스를 %Y%m%d 형식으로 변환

    # '매수/매도' 열 추가: Weighted_Signal_Final이 1이면 '매수', 0이면 '매도'
    df_selected['매수/매도'] = df_selected['Weighted_Signal_Final'].apply(lambda x: "매수" if x == 1 else "매도")
    

    # 열 이름에 'return'이 포함된 열을 .2%로 표시하는 설정
    # 매수/매도 열을 인덱스 열 옆에 추가
    table_columns = [
        {"name": "날짜", "id": "날짜", "type": "text"},  # 날짜 열을 제일 앞에 추가
        {"name": "매수/매도", "id": "매수/매도", "type": "text"},  # 매수/매도 열 추가
        {"name": "Daily_Return", "id": "Daily_Return", "type": "numeric", "format": Format(precision=2, scheme=Scheme.percentage)},  # 퍼센트 포맷 추가
        {"name": "Strategy_Return", "id": "Strategy_Return", "type": "numeric", "format": Format(precision=2, scheme=Scheme.percentage)},  # 퍼센트 포맷 추가
        *[
            {"name": col, "id": col, "type": "numeric", "format": Format(precision=2, scheme=Scheme.percentage)} 
            if 'return' in col.lower() else {"name": col, "id": col, "type": "numeric"}  # return이 포함된 열에 퍼센트 포맷 적용
            for col in df_selected.columns if col not in ['날짜', '매수/매도', 'Daily_Return', 'Strategy_Return', 'Weighted_Signal_Final']  # 이미 추가된 열 제외
        ]
    ]

    table_data = df_selected.to_dict('records')




    return fig_cumulative, fig_signal, weights_text, signal_text, table_data, table_columns

# 애플리케이션 실행
if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0")
