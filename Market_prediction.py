import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime
from dateutil.relativedelta import relativedelta
import concurrent.futures
from tqdm import tqdm
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

# 시작 및 종료 날짜 설정
start0 = (datetime.today() - relativedelta(years=1)).strftime('%Y-%m-%d')
end0 = datetime.today().strftime('%Y-%m-%d')

# 병렬로 데이터 가져오기 함수
def fetch_data(symbol):
    try:
        df = fdr.DataReader(symbol, start0, end0)['Close']
        df.index = pd.to_datetime(df.index, errors='coerce')  # 잘못된 날짜를 NaT로 변환
        df = df.dropna()  # NaT를 가진 행을 제거
        return symbol, df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return symbol, None

# 데이터프레임 병합 함수
def get_data(symbols):
    data = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_data, symbol): symbol for symbol in symbols}
        with tqdm(total=len(futures), desc="Fetching data") as pbar:
            for future in concurrent.futures.as_completed(futures):
                symbol, df = future.result()
                if df is not None:
                    data[symbol] = df
                pbar.update(1)
    return pd.DataFrame(data)

# 예시 심볼 (여러 심볼 사용)
symbols = ['SPY', 'NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']  # 여기에 원하는 심볼 추가 가능
df_price = get_data(symbols)

# 결측치 처리
df_price = df_price.ffill()

# 데이터셋 생성 함수
def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# LSTM 모델 생성 및 학습 함수
def build_and_train_model(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    X, Y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, Y, batch_size=1, epochs=1, verbose=0)

    return model, scaler, scaled_data

# 병렬로 주가 예측 함수
def predict_stock_price(symbol, data, time_step=60):
    model, scaler, scaled_data = build_and_train_model(data, time_step)

    # 현재가 가져오기
    current_price = data.iloc[-1]

    # 예측
    recent_data = scaled_data[-time_step:].reshape(1, time_step, 1)
    predictions = model.predict(recent_data)
    predicted_price = scaler.inverse_transform(predictions)

    return symbol, predicted_price[0, 0], current_price

# 병렬로 주가 예측 수행
predictions = {}
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(predict_stock_price, symbol, df_price[symbol], 60): symbol for symbol in symbols}
    with tqdm(total=len(futures), desc="Predicting prices") as pbar:
        for future in concurrent.futures.as_completed(futures):
            symbol, predicted_price, current_price = future.result()
            predictions[symbol] = (predicted_price, current_price)
            pbar.update(1)



# Dash 애플리케이션 구성
app = dash.Dash(__name__)


app.layout = html.Div(children=[
    html.H1(children='Stock Price Prediction'),

    dcc.Graph(
        id='price-graph',
        figure={
            'data': [
                go.Scatter(
                    x=df_price.index,
                    y=df_price[symbol],
                    mode='lines',
                    name=f'{symbol} Actual Price'
                ) for symbol in symbols  # 실제 주가를 선으로 표시
            ] + [
                go.Scatter(
                    x=[df_price.index[-1]],  # 마지막 날짜
                    y=[predicted_price],  # 예측된 가격
                    mode='markers+text',  # 마커와 텍스트를 동시에 표시
                    marker=dict(color='red', size=10),
                    name=f'{symbol} Predicted Price',
                    text=[f'{symbol}: {((predicted_price - current_price) / current_price * 100):.2f}%'],  # 심볼 이름과 % 변동률 함께 표시
                    textposition='top right'  # 텍스트 위치를 데이터 포인트의 우측 상단으로 설정
                ) for symbol, (predicted_price, current_price) in predictions.items()
            ],
            'layout': go.Layout(
                title='Actual vs Predicted Prices',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Price (USD)'},
                showlegend=True,
                margin=dict(r=150)  # 오른쪽 여유 공간 확보
            )
        },
        style={'width': '70vw', 'margin': '0 auto'}  # 그래프의 너비를 화면의 70%로 설정하고 가운데 정렬
    )
])




if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0")