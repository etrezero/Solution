import FinanceDataReader as fdr
import yfinance as yf
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.dash_table.Format import Format, Scheme
import dash_table
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
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import RandomOverSampler
from keras.callbacks import EarlyStopping



# Dash 애플리케이션 생성
app = dash.Dash(__name__)


# 종목 코드 설정
stock_code = {
    "069500": "KODEX 200",
    "114800": "KODEX 인버스",
    
    "SPY": "S&P500",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLF": "Financial",
    "XLV": "Health Care",
    "XLI": "Industrial",
    "XLB": "Materials",
    "XLK": "Technology",
    "XLRE": "Real Estate",
    "XLC": "Communication Services"
}



# 연도 선택을 위한 Dropdown 옵션 생성
current_year = datetime.today().year
year_options = [{'label': str(year), 'value': year} for year in range(current_year - 10, current_year + 1)]

# Layout 정의 (Dropdown 및 Loading 컴포넌트 추가)
app.layout = html.Div(
    style={'margin': 'auto', 'width': '85%', 'height': '100vh', 'display': 'flex', 'flexDirection': 'column'},
    children=[
        html.H1(f"Covenant AI Signal {datetime.today().strftime('%Y-%m-%d')}", style={'textAlign': 'center'}),        

        html.Div(
            id='buy-sell-signal', 
            style={'fontSize': 24, 'textAlign': 'center', 'color': 'green'}
        ),

        html.Label("Select Stock:"),
        dcc.Dropdown(
            id='stock-dropdown',
            options=[{'label': name, 'value': code} for code, name in stock_code.items()],
            value='069500',
            style={'width': '40%'}
        ),
        html.Label("Select Start Year:"),
        dcc.Dropdown(
            id='year-dropdown',
            options=year_options,
            value=current_year-10,
            style={'width': '40%'}
        ),

        # window 슬라이더 추가
        html.Label("Window Size:"),
        html.Div(
            dcc.Slider(
                id='window-weight',
                min=5, max=50, step=1, value=20,
                marks={i: f'{i}' for i in range(5, 51, 5)},
            ), style={'width': '50%'}
        ),

        # 가중치 슬라이더들
        html.Label("Bollinger Bands Weight:"),
        html.Div(
            dcc.Slider(
                id='bollinger-weight',
                min=0, max=1, step=0.05, value=0,
                marks={i / 10: f'{i * 10}%' for i in range(11)},
            ), style={'width': '50%'}
        ),

        html.Label("Rolling Return Weight:"),
        html.Div(
            dcc.Slider(
                id='rolling-return-weight',
                min=0, max=1, step=0.05, value=0,
                marks={i / 10: f'{i * 10}%' for i in range(11)},
            ), style={'width': '50%'}
        ),

        html.Label("SMA Weight:"),
        html.Div(
            dcc.Slider(
                id='sma-weight',
                min=0, max=1, step=0.05, value=1,
                marks={i / 10: f'{i * 10}%' for i in range(11)},
            ), style={'width': '50%'}
        ),

        dcc.Loading(
            id="loading-graph",
            type="default",
            children=html.Div([
                html.Div([
                    dcc.Graph(id='cumulative-return-graph', style={'width': '100%'}),
                ], style={'width': '100%', 'display': 'inline-block'}),

                html.Div([
                    dcc.Graph(id='signal-graph', style={'width': '80%'})
                ], style={'width': '100%', 'display': 'inline-block'}),

            ], style={'display': 'flex', 'width': '100%'})
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
                'textAlign': 'center',
                'padding': '10px',
                'fontFamily': 'Arial',
                'fontSize': '15px',
                'minWidth': '80px',
                'maxWidth': '80px',
                'width': '80px',
            },
            style_table={'width': '100%', 'overflowX': 'auto'},
            page_size=15
        ),
    ]
)




# 데이터 가져오기 함수에서 선택된 종목 코드를 사용
def fetch_data(selected_stock_code, start, end):
    try:
        if isinstance(selected_stock_code, int) or selected_stock_code.isdigit():
            if len(selected_stock_code) == 5:
                selected_stock_code = '0' + selected_stock_code
            df_price = pykrx.get_market_ohlcv_by_date(start, end, selected_stock_code)['종가']
        else:
            session = requests.Session()
            session.verify = False  # SSL 인증서 검증 비활성화
            yf_data = yf.Ticker(selected_stock_code, session=session)
            df_price = yf_data.history(start=start, end=end)['Close']

        df_price = pd.DataFrame(df_price)
        df_price.columns = [selected_stock_code]  # 올바른 열 이름 설정
        df_price.index = pd.to_datetime(df_price.index).strftime('%Y-%m-%d')  # 인덱스를 %Y-%m-%d 형식으로 변환
    
        return df_price
    
    except Exception as e:
        print(f"Error fetching data for {selected_stock_code}: {e}")
        return None



# 누적 수익률 그래프가 제대로 나오도록 수정한 코드

@app.callback(
    [Output('cumulative-return-graph', 'figure'),
     Output('signal-graph', 'figure'),
     Output('optimal-weights-output', 'children'),
     Output('buy-sell-signal', 'children'),
     Output('data-table', 'data'),
     Output('data-table', 'columns')],
    [Input('stock-dropdown', 'value'),
     Input('year-dropdown', 'value'),
     Input('bollinger-weight', 'value'),
     Input('rolling-return-weight', 'value'),
     Input('sma-weight', 'value'),
     Input('window-weight', 'value')]  # window-weight 추가
)
def update_graph(selected_stock_code, selected_year, bollinger_weight, rolling_return_weight, sma_weight, window):
    # 시작 및 끝 날짜 설정
    start = datetime(selected_year, 1, 1).strftime('%Y-%m-%d')
    end = datetime.today().strftime('%Y-%m-%d')

    # 데이터 가져오기
    df_price = fetch_data(selected_stock_code, start, end)
    if df_price is None or df_price.empty:
        return {}, {}, f"Error: No data for {selected_stock_code}", "데이터 없음", None, None

    # 일간 수익률 계산
    df_price['Daily_Return'] = df_price[selected_stock_code].pct_change().fillna(0)

    # Bollinger Bands 시그널
    df_price['MA20'] = df_price[selected_stock_code].rolling(window=window).mean()
    df_price['Upper'] = df_price['MA20'] + (df_price[selected_stock_code].rolling(window=window).std() * 0.5)
    df_price['Lower'] = df_price['MA20'] - (df_price[selected_stock_code].rolling(window=window).std() * 0.5)
    df_price['BB_Signal'] = 0
    df_price.loc[df_price[selected_stock_code] > df_price['Upper'], 'BB_Signal'] = 1
    df_price.loc[df_price[selected_stock_code] < df_price['Lower'], 'BB_Signal'] = -1

    # Rolling Return 시그널
    df_price['RR'] = df_price[selected_stock_code].pct_change(window).fillna(0).rolling(window=window).mean()
    df_price['RR_Signal'] = 0
    df_price['RR_Diff'] = df_price['RR'].diff()
    df_price.loc[df_price['RR_Diff'] > 0, 'RR_Signal'] = 1
    df_price.loc[df_price['RR_Diff'] < 0, 'RR_Signal'] = -0.5

    # SMA 시그널
    df_price['SMA'] = df_price[selected_stock_code].rolling(window=window).mean()
    df_price['SMA_Signal'] = 0
    df_price.loc[df_price['SMA'].diff() > 0, 'SMA_Signal'] = 1
    df_price.loc[df_price['SMA'].diff() < 0, 'SMA_Signal'] = -0.5

    # 가중치로 최종 시그널 생성
    df_price['Weighted_Signal'] = (bollinger_weight * df_price['BB_Signal'] +
                                   rolling_return_weight * df_price['RR_Signal'] +
                                   sma_weight * df_price['SMA_Signal'])
    df_price['Weighted_Signal_Final'] = df_price['Weighted_Signal'].apply(lambda x: 1 if x > 0 else 0)

    # 포트폴리오 수익률 계산
    df_price['Portfolio_Return'] = np.where(df_price['Weighted_Signal_Final'].shift(1) == 1, df_price['Daily_Return'], 0)
    df_price['Portfolio_Cum_Return'] = (1 + df_price['Portfolio_Return']).cumprod() - 1
    df_price['Stock_Cum_Return'] = (1 + df_price['Daily_Return']).cumprod() - 1

    # 초과 수익률 계산
    df_price['Excess_Return'] = df_price['Portfolio_Cum_Return'] - df_price['Stock_Cum_Return']

    # 매수/매도 시그널
    last_signal = df_price['Weighted_Signal_Final'].iloc[-1]
    signal_text = "매수" if last_signal == 1 else "매도"

    # 전략과 선택한 종목의 누적 수익률 그래프
    fig_cumulative = go.Figure()

    # 전략포트폴리오 누적 수익률
    fig_cumulative.add_trace(go.Scatter(x=df_price.index, y=df_price['Portfolio_Cum_Return'], mode='lines', name='Covenant Portfolio'))

    # 선택종목의 누적 수익률
    fig_cumulative.add_trace(go.Scatter(x=df_price.index, y=df_price['Stock_Cum_Return'], mode='lines', name=f'{selected_stock_code}'))

    # 누적 수익률 Gap : bar 차트로 표시 (Excess Return)
    fig_cumulative.add_trace(go.Bar(
        x=df_price.index,
        y=df_price['Excess_Return'],
        name='Excess Return',
        marker_color='orange',
        opacity=1  # 투명도 설정
    ))

    fig_cumulative.update_layout(
        title=f'Cumulative Return and Excess Return for {selected_stock_code} (From {selected_year})',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        yaxis={'tickformat': '.0%'},
        xaxis=dict(range=[df_price.index.min(), df_price.index.max()]),
    )

    # 시그널 그래프
    fig_signal = go.Figure()
    fig_signal.add_trace(
        go.Scatter(
            x=df_price.index, 
            y=df_price['Weighted_Signal_Final'], 
            mode='markers', 
            name='Weighted Signal Final'
        )
    )
    fig_signal.update_layout(
        title=f'Signal for {selected_stock_code} (From {selected_year})',
        xaxis_title='Date',
        yaxis_title='Signal (0 or 1)',
        xaxis=dict(range=[df_price.index.min(), df_price.index.max()])
    )

    # 최적 가중치 출력
    weights_text = f"Optimal Weights: Bollinger: {bollinger_weight*100:.1f}%, Rolling Return: {rolling_return_weight*100:.1f}%, SMA: {sma_weight*100:.1f}%"

    # 테이블 생성
    df_selected = df_price[['Daily_Return', 'Portfolio_Cum_Return', 'Stock_Cum_Return', 'BB_Signal', 'SMA_Signal', 'RR_Signal', 'Weighted_Signal_Final']].reset_index()    
    df_selected['날짜'] = df_price.index
    df_selected['매수/매도'] = df_selected['Weighted_Signal_Final'].apply(lambda x: "매수" if x == 1 else "매도")

    table_columns = [
        {"name": "날짜", "id": "날짜", "type": "text"},
        {"name": "매수/매도", "id": "매수/매도", "type": "text"},
        {"name": "Daily_Return", "id": "Daily_Return", "type": "numeric", "format": Format(precision=2, scheme=Scheme.percentage)},
        {"name": "Portfolio_Cum_Return", "id": "Portfolio_Cum_Return", "type": "numeric", "format": Format(precision=2, scheme=Scheme.percentage)},  # 전략 포트폴리오 누적 수익률        
        {"name": "BB_Signal", "id": "BB_Signal", "type": "numeric"},
        {"name": "SMA_Signal", "id": "SMA_Signal", "type": "numeric"},
        {"name": "RR_Signal", "id": "RR_Signal", "type": "numeric"}
    ]
    table_data = df_selected.to_dict('records')

    return fig_cumulative, fig_signal, weights_text, signal_text, table_data, table_columns



# 애플리케이션 실행
if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0")