import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash import dash_table
import concurrent.futures
import os
import pickle
from pykrx import stock as pykrx
import yfinance as yf
import ssl
import requests

import numpy as np
from scipy.optimize import minimize
import time
from urllib.error import HTTPError  # HTTPError 예외 처리용 모듈 임포트
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import chardet




# 캐싱 path
cache_path = r'C:\Covenant\data\Trading_Timing.pkl'
cache_expiry = timedelta(days=1)


code_A = {"252670": "KODEX 200선물인버스2X",
          "122630": "KODEX 레버리지"}

# code 딕셔너리 병합
code_dict = {}
code_dict.update(code_A)


code = list(set(code_dict.keys()))


# code_dict_A와 code_dict_B 병합
# code_dict = {**code_dict_A, **code_dict_B}

# 시작 날짜 설정
start = (datetime.today() - relativedelta(years=1)).strftime('%Y-%m-%d')
end = (datetime.today() - timedelta(days=0)).strftime('%Y-%m-%d')



# 데이터 가져오기 함수
def fetch_data(code, start, end):
    try:
        if isinstance(code, int) or code.isdigit() or code.endswith(".KS"):
            if isinstance(code, int):
                code = str(code)
            if len(code) == 5:
                code = '0' + code
            if code.endswith(".KS"):
                code = code.replace(".KS", "")
            # pykrx로부터 '종가' 열을 불러와 'Close'로 변경
            df_price = pykrx.get_market_ohlcv_by_date(start, end, code)
            if '종가' in df_price.columns:
                df_price = df_price['종가'].rename(code)
            else:
                raise ValueError(f"{code}: '종가' column not found in pykrx data.")
        else:
            # FinanceDataReader로 데이터를 불러옴
            df_price = fdr.DataReader(code, start, end)['Close'].rename(code)
        return df_price
    except Exception as e:
        print(f"Error fetching data for {code}: {e}")
        return None

# 병렬로 데이터를 가져오는 함수
def FDR(code, start, end, batch_size=10):
    
    data_frames = []
    for i in range(0, len(code), batch_size):
        code_batch = code[i:i + batch_size]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(fetch_data, c, start, end): c for c in code_batch}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    data_frames.append(result)

    price_data = pd.concat(data_frames, axis=1) if data_frames else pd.DataFrame()

    return price_data


# 데이터 불러오기 및 출력
try:
    df_price = FDR(code, start, end)  # start, end 인자를 전달
    df_price = df_price.ffill()  # 결측값 보정
    print(df_price)
except ValueError as ve:
    print(f"Error: {ve}")




# 볼린저 밴드 시그널 생성 함수
def bollinger_bands(df_price, target_col, n=20, k=2):
    df_price['MA20'] = df_price[target_col].rolling(window=n).mean()
    df_price['Upper'] = df_price['MA20'] + (df_price[target_col].rolling(window=n).std() * k)
    df_price['Lower'] = df_price['MA20'] - (df_price[target_col].rolling(window=n).std() * k)
    df_price['Signal'] = 0
    df_price['Signal'][df_price[target_col] > df_price['Upper']] = -1  # 매도 신호
    df_price['Signal'][df_price[target_col] < df_price['Lower']] = 1   # 매수 신호
    return df_price

# 칼만 필터 적용
def kalman_filter(df_price, target_col):
    kf = np.zeros(len(df_price))
    kf[0] = df_price[target_col][0]
    for i in range(1, len(df_price)):
        kf[i] = kf[i-1] + 0.1 * (df_price[target_col][i] - kf[i-1])  # 간단한 칼만 필터 구현
    df_price['Kalman'] = kf
    return df_price

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

# LSTM 모델 생성
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(input_shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model









# 주요 종목 코드 설정
target_col = "122630"  # KODEX 200선물인버스2X

# 데이터 불러오기 및 출력
df_price = FDR([target_col], start, end)
df_price = df_price.ffill()  # 결측값 보정

# 볼린저 밴드 계산
df_price = bollinger_bands(df_price, target_col)

# 칼만 필터 적용
df_price = kalman_filter(df_price, target_col)

# LSTM 데이터 준비
X, y, scaler = create_lstm_data(df_price, target_col)

# LSTM 모델 학습
lstm_model = create_lstm_model(X.shape)
lstm_model.fit(X, y, epochs=10, batch_size=32)

# LSTM 예측값 추가
predicted = lstm_model.predict(X)
predicted = scaler.inverse_transform(predicted)

# LSTM 기반 시그널 계산
df_price['LSTM_Signal'] = 0

# 슬라이스 범위를 조정하여 LSTM 시그널 계산
signal_length = len(predicted) - 1  # predicted[1:]와 predicted[:-1]의 길이를 맞춤
lstm_signals = np.sign(predicted[1:] - predicted[:-1])


# 투자 비중 조절 및 누적 수익률 계산
df_price['Signal_Final'] = df_price['Signal'] + df_price['LSTM_Signal']
df_price['Signal_Final'] = df_price['Signal_Final'].apply(lambda x: 100 if x > 0 else (0 if x < 0 else 50))

df_price['Daily_Return'] = df_price[target_col].pct_change()
df_price['Strategy_Return'] = df_price['Daily_Return'] * (df_price['Signal_Final'] / 100)
df_price['Cum_Return'] = (1 + df_price['Strategy_Return']).cumprod() - 1

# 누적 수익률 그래프 시각화
plt.figure(figsize=(14, 7))
plt.plot(df_price.index, df_price['Cum_Return'], label='Strategy Cumulative Return')
plt.title(f'{target_col} 종목 매매 시그널 기반 누적 수익률')
plt.legend()
plt.show()

