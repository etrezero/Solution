import yfinance as yf
from pykrx import stock as pykrx
import concurrent.futures
import requests
import pickle
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.metrics import dtw
from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA




# 캐시 경로 및 만료 시간 설정
cache_price = r'C:\Covenant\data\DTW_price.pkl'
cache_expiry = timedelta(days=1)


# 데이터 가져오기 함수 (캐시 사용)
def fetch_data(code, start, end):
    try:
        if isinstance(code, int) or code.isdigit():
            if len(code) == 5:
                code = '0' + code
            df_price = pykrx.get_market_ohlcv_by_date(start, end, code, freq='w')['종가']
        else:
            session = requests.Session()
            session.verify = False  # SSL 인증서 검증 비활성화
            yf_data = yf.Ticker(code, session=session)
            df_price = yf_data.history(start=start, end=end).resample('W').last()['Close']

        df_price = df_price.ffill().bfill()
        df_price = pd.DataFrame(df_price)
        df_price.columns = [code]
        df_price.index = pd.to_datetime(df_price.index).strftime('%Y-%m-%d')  # 인덱스를 %Y%m%d 형식으로 변환 후 문자열로 저장
        df_price.index = pd.to_datetime(df_price.index).tz_localize(None)  # 시간대 정보 제거

        return df_price

    except Exception as e:
        print(f"Error fetching data for {code}: {e}")
        return None

def Func(code, start, end, batch_size=10):
    # 디버깅용 출력
    print(f"Start: {start}, End: {end}")
    
    # 날짜 형식 강제 변환
    start = pd.to_datetime(start).strftime('%Y-%m-%d')
    end = pd.to_datetime(end).strftime('%Y-%m-%d')

    if isinstance(code, str):  # 단일 코드 처리
        return fetch_data(code, start, end)

    if os.path.exists(cache_price):

        cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_price))
        if datetime.now() - cache_mtime < cache_expiry:
            with open(cache_price, 'rb') as f:
                print("Loading data from cache...")
                df_price = pickle.load(f)
                if df_price.empty:  # 캐싱된 데이터가 비어 있을 경우
                    print("Cached data is empty. Reloading data.")
                    return fetch_data(code, start, end)
                return df_price

    df = []
    for i in range(0, len(code), batch_size):
        code_batch = code[i:i + batch_size]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(fetch_data, c, start, end): c for c in code_batch}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    df.append(result)

    df_price = pd.concat(df, axis=1) if df else pd.DataFrame()
    df_price = df_price.bfill().ffill()

    with open(cache_price, 'wb') as f:
        pickle.dump(df_price, f)
        print("Data cached.")

    return df_price


start = (datetime.today() - relativedelta(years=10)).strftime('%Y-%m-%d')
end = (datetime.today() - timedelta(1)).strftime('%Y-%m-%d')


df_price = Func('069500.KS', start, end)

print(df_price)




    

# DTW 기반 국면 분석 함수
def dynamic_time_warping_analysis(stock_data, n_clusters=3):
    # 주가 데이터를 정규화하여 DTW에 적용
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(stock_data)

    # DTW 기반 클러스터링
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", verbose=True)
    labels = model.fit_predict(scaled_data.reshape(-1, 1, 1))

    return labels



# 시계열 데이터 시각화
def plot_stock_data(stock_data, labels):
    plt.figure(figsize=(10, 6))

    for cluster_id in np.unique(labels):
        cluster_data = stock_data[labels == cluster_id]
        plt.plot(cluster_data.index, cluster_data.values, label=f'Cluster {cluster_id}')

    plt.title('Dynamic Time Warping Clustering')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()




# # DTW 기반 국면 분석
labels = dynamic_time_warping_analysis(df_price, n_clusters=4)

# 분석 결과 시각화
plot_stock_data(df_price, labels)

# 국면별 주요 통계 출력
for cluster_id in np.unique(labels):
    cluster_data = df_price[labels == cluster_id]
    print(f"Cluster {cluster_id} Summary:")
    print(cluster_data.describe())






