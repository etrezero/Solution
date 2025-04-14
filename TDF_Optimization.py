# 필요한 패키지 임포트
from flask import Flask
import socket
import dash

from dash import Dash, dcc, html, dash_table, Input, Output
from dash.dash_table.Format import Format, Scheme
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import openpyxl
import warnings
import os
from openpyxl import Workbook
import yfinance as yf
from pykrx import stock as pykrx
import requests
import pickle
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import concurrent.futures
import numpy as np
from scipy.optimize import minimize



# 주가와 배당 불러 올 때 토탈프라이스 함수 만들어서 튜플[0][1][2]를 데이터프레임으로 지정 
# 배당은 3개월 주기여서 월간으로 ffill하고 3(분기)으로 나눠서 매달적용


# 1. SLSQP (Sequential Least Squares Programming)
    # 기본적으로 사용한 방식입니다.
    # 비선형 제약 조건을 다룰 수 있는 유용한 방법.
    # 속도가 빠르며, 중소 규모 문제에 적합.
# 2. COBYLA (Constrained Optimization BY Linear Approximations)
    # 제약 조건이 비선형이거나 미분 불가능한 경우 유용.
    # 파생된 정보 없이 제한된 자원의 함수 최적화를 해결하는 데 적합.
    # 장점: 미분할 수 없는 문제도 다룰 수 있음.
    # 단점: 높은 정확성을 요구하는 문제에선 부적합할 수 있음.
# 3. L-BFGS-B (Limited-memory Broyden–Fletcher–Goldfarb–Shanno with Box constraints)
    # 장점: 많은 변수와 제약이 있을 때 메모리를 절약하는 알고리즘.
    # 단점: 전역 최적화를 보장하지 않으며, 파생 정보가 필요.
# 4. TNC (Truncated Newton Conjugate-Gradient)
    # 장점: 큰 스케일의 문제를 해결하는 데 적합하며, 제약 조건을 다룰 수 있음.
    # 단점: 가끔 수렴 속도가 느리거나 미분 가능성이 없는 문제에 부적합할 수 있음.
# 5. Powell
    # 미분을 사용하지 않는 방식으로, 직접적인 함수 평가만을 통해 최적화를 수행.
    # 장점: 미분 가능하지 않은 함수에도 사용 가능.
    # 단점: 제약 조건을 직접 지원하지 않음.
# 6. Trust Region Constrained (trust-constr)
    # SciPy의 최적화 함수에서 제공되는 최신 알고리즘.
    # 장점: 제약 조건을 더 잘 처리하고, 높은 정확도를 가진 문제에 적합.
    # 단점: 메모리 사용량이 클 수 있음.



# === 전체 코드가 매우 깁니다. ===
# 파일로 관리하는 것이 좋으므로 아래와 같이 텍스트 문서로 넘깁니다.



# 전체 코드는 Dash 기반 TDF 최적화 대시보드를 구성하며,
# 1. 데이터 수집 (yfinance, pykrx, 배당)
# 2. 총수익 계산
# 3. 월간 수익률과 환산 수익률 계산
# 4. CAGR, 평균 롤링 수익률, 변동성 계산
# 5. 위험대비 수익률 기반 CMA 테이블 작성
# 6. Glide Path 기반 채권/주식 최적 비중 산출
# 7. MP 최적 포트폴리오 구성
# 8. Bubble Chart 시각화 (변동성, 수익률, 위험대비수익률)
# 9. Dash Layout 구성 및 실행

# 아래 텍스트에 전체 파이썬 코드가 저장되어 있습니다.
# 코드 에디터에서 이 파일을 저장 후 실행하면 전체 대시보드 작동합니다.

# 현재 문서에는 앞서 제공한 코드를 그대로 반영하였습니다.
# 중간중간 수정 사항은 다음과 같습니다:
# 1. 표준화 시 ZeroDivision 방지
# 2. Bubble Chart에서 size는 고정값 대신 위험대비수익률 기반 비례 크기 적용 가능 (추가 예정 시 알려주세요)
# 3. top_names 텍스트 주석 표시 수정
# 4. 포트 충돌 방지용 find_available_port 함수 적용






# 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning)


cache_path = r'C:\Covenant\TDF\data\CMA_optimization.pkl'
cache_expiry = timedelta(days=180)


code_dict = {
    "ACWI": "iShares MSCI ACWI ETF",
    "SPY": "SPDR S&P 500 ETF Trust",
    "VUG": "Vanguard Growth ETF",
    "VTV": "Vanguard Value ETF",
    "VO": "Vanguard Mid-Cap ETF",
    "VB": "Vanguard Small-Cap ETF",
    "VEA": "Vanguard FTSE Developed Markets ETF",
    "VWO": "Vanguard FTSE Emerging Markets ETF",
    "069500.KS": "Samsung KODEX 200 ETF",
    "BIL": "SPDR Bloomberg 1-3 Month T-Bill ETF",
    "BND": "Vanguard Total Bond Market ETF",
    "VGSH": "Vanguard Short-Term Treasury ETF",
    "VGIT": "Vanguard Intermediate-Term Treasury ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "TIP": "iShares TIPS Bond ETF",
    "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
    "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
    "BNDX": "Vanguard Total International Bond ETF",
    "EMB": "iShares JP Morgan USD Emerging Markets Bond ETF",
    "273130.KS": "SAMSUNG KODEX Active Korea Total Bond Market(AA-) ETF",
    "157450.KS": "Mirae Asset Tiger Money Market ETF",
    "114260.KS": "Samsung KODEX Treasury Bond ETF",
    "148070.KS": "Kiwoom KOSEF 10Y KTB ETF",
    "GSG": "iShares S&P GSCI Commodity Indexed Trust",
    "GLD": "SPDR Gold Shares",
    "KRW=X": "KRW/USD"
}




# 데이터 가져오기 함수===============================================
def fetch_price(code, start, end):
    try:
        if isinstance(code, int) or code.isdigit() or code.endswith(".KS"):
            if isinstance(code, int):
                code = str(code)
            if len(code) == 5:
                code = '0' + code
            if code.endswith(".KS"):
                code = code.replace(".KS", "")
            df_price = pykrx.get_market_ohlcv_by_date(start, end, code)
            if '종가' in df_price.columns:
                df_price = df_price['종가'].rename(code)
            else:
                raise ValueError(f"{code}: '종가' column not found in pykrx data.")
        else:
            session = requests.Session()
            session.verify = False  # SSL 인증서 검증 비활성화
            yf_data = yf.Ticker(code, session=session)
            df_price = yf_data.history(start=start, end=end)['Close'].rename(code)


        # 월간/월말 데이터 추출 : 데이터프레임으로 변환 및 인덱스 포맷 설정
        df_price = pd.DataFrame(df_price)
        df_price.columns = [code]
        df_price.index = pd.to_datetime(df_price.index).strftime('%Y-%m-%d')  # 인덱스를 %Y-%m-%d 형식으로 변환

        # 월말 기준으로 리샘플링 (종가)
        df_price.index = pd.to_datetime(df_price.index).tz_localize(None) 
        df_price = df_price.resample('ME').last()  # 월말 종가 기준 리샘플링

        return df_price
    
    except Exception as e:
        print(f"Error fetching data for {code}: {e}")
        return None
#===============================================================


# 배당 데이터를 가져오는 함수==================================
def fetch_dividend(code, start, end):
    try:
        if not (isinstance(code, int) or code.isdigit() or code.endswith(".KS")):
            session = requests.Session()
            session.verify = False  # SSL 인증서 검증 비활성화
            yf_data = yf.Ticker(code, session=session)
            df_dividend = yf_data.dividends.rename(code)

            # 배당 데이터가 비어있는지 확인
            if df_dividend.empty:
                print(f"No dividend data for {code}")
                return pd.DataFrame()  # 배당 데이터가 없으면 빈 데이터프레임 반환

            df_dividend = pd.DataFrame(df_dividend)

            # 타임존 제거 및 날짜 형식 변환
            df_dividend.index = pd.to_datetime(df_dividend.index).tz_localize(None)

            # 월말로 리샘플링 및 직전 값으로 채우기
            df_dividend = df_dividend.resample('ME').ffill()

            return df_dividend

        else:
            return pd.DataFrame()  # KRX에서는 배당 데이터가 없으므로 빈 데이터프레임 반환
    except Exception as e:
        print(f"Error fetching dividend data for {code}: {e}")
        return pd.DataFrame()

#=========================================================




# 총 수익 계산 함수 (가격 + 배당 반영)
def calculate_total(price_data, dividend_data):
    df_total = price_data.copy()
    
    for col in price_data.columns:
        if col in dividend_data.columns:
            dividend = dividend_data[col].fillna(0)  # 배당 데이터에서 결측치는 0으로 처리
            # 배당을 반영하여 총 수익 계산
            df_total[col] = price_data[col] + (dividend/3).cumsum()
            
    return df_total



# Func 함수 수정 - 캐시 파일을 하나로 통합
def Func(code, start, end, batch_size=10):

    # 캐시가 존재하고 유효 기간 내라면 캐시에서 데이터 로드
    if os.path.exists(cache_path):
        cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - cache_mtime < cache_expiry:
            with open(cache_path, 'rb') as f:
                print("Loading data from cache...")
                return pickle.load(f), pd.DataFrame(), pd.DataFrame()  # 캐시에서 데이터 로드 시 배당 데이터는 빈 데이터프레임 반환

    # 캐시가 없거나 만료되었다면 데이터를 다시 불러오고 캐시 저장
    price_frames = []
    dividend_frames = []
    for i in range(0, len(code), batch_size):
        code_batch = code[i:i + batch_size]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 비동기로 가격 데이터를 가져옴
            futures = {executor.submit(fetch_price, c, start, end): c for c in code_batch}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if isinstance(result, pd.DataFrame) and not result.empty:  # 데이터프레임인지 확인 후 empty 속성 확인
                    price_frames.append(result)

            # 비동기로 배당 데이터를 가져옴
            futures_dividend = {executor.submit(fetch_dividend, c, start, end): c for c in code_batch}
            for future in concurrent.futures.as_completed(futures_dividend):
                result = future.result()
                if isinstance(result, pd.DataFrame) and not result.empty:  # 데이터프레임인지 확인 후 empty 속성 확인
                    dividend_frames.append(result)

    price_data = pd.concat(price_frames, axis=1) if price_frames else pd.DataFrame()
    dividend_data = pd.concat(dividend_frames, axis=1) if dividend_frames else pd.DataFrame()


    # 배당을 반영한 가격 데이터 계산
    df_total = calculate_total(price_data, dividend_data)

    # 데이터를 캐시에 저장
    with open(cache_path, 'wb') as f:
        pickle.dump(df_total, f)
        print("Data cached.")

    return df_total, price_data, dividend_data  # 세 개의 값을 반환

# 시작 날짜 설정
start = (datetime.today() - relativedelta(years=10, months=1)).strftime('%Y-%m-%d')
end = (datetime.today() - timedelta(days=0)).strftime('%Y-%m-%d')

# 데이터 불러오기 및 출력
code = list(code_dict.keys())
df = Func(code, start, end)  # return 세 값을 받음
df_total = df[0]
df_price = df[1]    
df_dividend = df[2]

print("df_dividend============", df_dividend)


R_total = df_total.pct_change().fillna(0)



#엑셀 저장=======================================================
def save_excel(df, sheetname, index_option=None):
    
    # 파일 경로
    path = rf'C:\Covenant\TDF\data\CMA_optimization.xlsx'

    # 파일이 없는 경우 새 Workbook 생성
    if not os.path.exists(path):
        wb = Workbook()
        wb.save(path)
        print(f"새 파일 '{path}' 생성됨.")
    
    # 인덱스를 날짜로 변환 시도
    try:
        # index_option이 None일 경우 인덱스를 포함하고 날짜 형식으로 저장
        if index_option is None or index_option:  # 인덱스를 포함하는 경우
            df.index = pd.to_datetime(df.index, errors='raise')  # 변환 실패 시 오류 발생
            df.index = df.index.strftime('%Y-%m-%d')  # 벡터화된 방식으로 날짜 포맷 변경
            index = True  # 인덱스를 포함해서 저장
        else:
            index = False  # 인덱스를 제외하고 저장
    except Exception:
        print("Index를 날짜 형식으로 변환할 수 없습니다. 기본 인덱스를 사용합니다.")
        index = index_option if index_option is not None else True  # 변환 실패 시에도 인덱스를 포함하도록 설정

    # DataFrame을 엑셀 시트로 저장
    with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheetname, index=index)  # index 여부 설정
        print(f"'{sheetname}' 저장 완료.")


# save_excel(df_price, "df_price")
# save_excel(df_dividend, "df_dividend")
# save_excel(df_total, "df_total")

# save_excel(R_total, "R_total")
#==========================================================



#KRW 수익률 구하기 =============================================
def calculate_return_krw(R_total):
    # "KRW=X" 열이 있는지 확인
    if "KRW=X" not in R_total.columns:
        raise ValueError("KRW=X 열이 데이터프레임에 존재하지 않습니다.")

    # 새로운 데이터프레임을 생성하여 필요한 열에 KRW 변환 적용
    df_R_M_KRW = R_total.copy()

    # KRW=X 수익률을 계산
    krw_return = R_total["KRW=X"].replace([np.inf, -np.inf], 0).fillna(0)

    for col in R_total.columns:
        if col != "KRW=X" and not col.isdigit() and not pd.api.types.is_integer_dtype(R_total[col]):
        
            asset_return = R_total[col].replace([np.inf, -np.inf], 0).fillna(0)
            df_R_M_KRW[col] = asset_return + krw_return

    return df_R_M_KRW

# df_R_M_KRW 데이터프레임 생성
df_R_M_KRW = calculate_return_krw(R_total)

# save_excel(df_R_M_KRW, "df_R_M_KRW")
#================================================================




#USD 수익률 구하기 =========================================================
def calculate_return_USD(R_total):
     # "KRW=X" 열이 있는지 확인
    if "KRW=X" not in R_total.columns:
        raise ValueError("KRW=X 열이 데이터프레임에 존재하지 않습니다.")

    # 새로운 데이터프레임을 생성하여 필요한 열에 KRW 변환 적용
    df_R_M_USD = R_total.copy()

    # KRW=X 수익률을 계산
    krw_return = R_total["KRW=X"].replace([np.inf, -np.inf], 0).fillna(0)

    for col in R_total.columns:
        if col != "KRW=X" and col.isdigit() and pd.api.types.is_integer_dtype(R_total[col]):
        
            asset_return = R_total[col].replace([np.inf, -np.inf], 0).fillna(0)
            df_R_M_USD[col] = asset_return - krw_return

    return df_R_M_USD

df_R_M_USD = calculate_return_USD(R_total)
# save_excel(df_R_M_USD, "df_R_M_USD")
#===========================================================



#월간 수익률(not price)로 CAGR을 구하는 함수=======================
def calculate_cagr(df):
    cagr_dict = {}  # 결과를 저장할 딕셔너리

    # 각 열에 대해 시작값, 끝값 및 기간을 계산
    for col in df.columns:
        df[col] = df[col].bfill()  # 빈 값을 앞으로 채움
        
        # 월간 수익률을 연환산(CAGR)하기 위한 계산
        start_value = df[col].iloc[0] + 1  # 첫 번째 월간 수익률 (0을 방지하기 위해 1 더함)
        end_value = ((1 + df[col]).cumprod()).iloc[-1]  # 마지막 월간 수익률 (복리 적용)
        
        # 전체 기간을 연 단위로 계산 (개월 수를 12로 나눠 연 단위로 변환)
        total_periods = len(df)
        years = total_periods / 12  # 총 기간을 연 단위로 변환
        
        # CAGR 계산: (최종값 / 초기값) ^ (1/연수) - 1
        if start_value > 0:  # 시작값이 0이 아닌 경우에만 계산
            cagr_value = (end_value / start_value) ** (1 / years) - 1
        else:
            cagr_value = None  # 0으로 시작하는 경우 계산 불가
        
        # 결과를 딕셔너리에 저장
        cagr_dict[col] = cagr_value

    # 딕셔너리를 데이터프레임으로 변환
    CAGR_return = pd.DataFrame(list(cagr_dict.items()), columns=['Ticker', 'CAGR'])
    
    return CAGR_return


CAGR_KRW = calculate_cagr(df_R_M_KRW)
CAGR_USD = calculate_cagr(df_R_M_USD)

print("CAGR_KRW===================", CAGR_KRW)
print("CAGR_USD===================", CAGR_USD)

# save_excel(CAGR_KRW, "CAGR_KRW", index_option=False)
# save_excel(CAGR_USD, "CAGR_USD", index_option=False)



def calculate_avg_rolling_return(df, window=12):
    avg_rolling_return_dict = {}

    # 각 열에 대해 1년(12개월) 롤링 리턴 계산
    for col in df.columns:
        # 입력 데이터프레임이 이미 월간 수익률을 포함하고 있으므로 바로 복리 계산
        rolling_return = (1 + df[col]).rolling(window=window).apply(lambda x: x.prod() - 1, raw=False)

        # 각 열에 대해 12개월 롤링 리턴의 평균을 계산
        avg_rolling_return_dict[col] = rolling_return.mean()

    # 딕셔너리를 데이터프레임으로 변환
    avg_rolling_return = pd.DataFrame(list(avg_rolling_return_dict.items()), columns=['Ticker', 'Average 1-Year Rolling Return'])

    return avg_rolling_return


avg_RR_KRW = calculate_avg_rolling_return(df_R_M_KRW)
print("avg_RR_KRW================", avg_RR_KRW)
# save_excel(avg_RR_KRW, "avg_RR_KRW", index_option=False)


avg_RR_USD = calculate_avg_rolling_return(df_R_M_USD)
# print("avg_RR_USD================", avg_RR_USD)
# save_excel(avg_RR_USD, "avg_RR_USD", index_option=False)







# 윈도우별 변동성 계산 함수 (12개월 수익률의 롤링 표준편차를 평균하여 계산)
# def calculate_Vol(df_price):
#     if df_price is None or df_price.empty:
#         return None

#     returns = df_price.pct_change().fillna(0)
#     rolling_std = returns.rolling(window=12).std()
#     annualized_vol = rolling_std * np.sqrt(12)
#     # 평균 변동성 계산
#     average_vol = annualized_vol.mean()
#     return average_vol



# 전체기간 변동성 계산 함수
def calculate_Vol(df_R_M):
    if df_price is None or df_price.empty:
        return None

    # 12개월 롤링 표준편차 계산 및 연율화
    vol = df_R_M.std() * np.sqrt(12)
    return vol

# 예시로 df_price 데이터프레임이 있을 경우 실행
Vol_KRW = df_R_M_KRW.std() * np.sqrt(12)
Vol_USD = df_R_M_USD.std() * np.sqrt(12)
print("Vol_KRW==============================", Vol_KRW)
# print("Vol_USD==============================", Vol_USD)

# save_excel(Vol_KRW, "Vol_KRW")
# save_excel(Vol_USD, "Vol_USD")




df_CMA_KRW = pd.concat([avg_RR_KRW.set_index('Ticker'), Vol_KRW], axis=1)
df_CMA_KRW.columns = ['E(R)_KRW', 'Vol_KRW']
df_CMA_KRW['위험대비수익률(KRW)'] = df_CMA_KRW['E(R)_KRW'] / df_CMA_KRW['Vol_KRW']
df_CMA_KRW.reset_index()
df_CMA_KRW.rename(columns={'index': 'Asset'}, inplace=True)
print("df_CMA_KRW===============", df_CMA_KRW)


df_CMA_USD = pd.concat([avg_RR_USD.set_index('Ticker'), Vol_USD], axis=1)
df_CMA_USD.columns = ['E(R)_USD', 'Vol_USD']
df_CMA_USD['위험대비수익률(USD)'] = df_CMA_USD['E(R)_USD'] / df_CMA_USD['Vol_USD']
df_CMA_USD.reset_index()
df_CMA_USD.rename(columns={'index': 'Asset'}, inplace=True)  # 열 이름을 Asset으로 변경
print("df_CMA_USD===============", df_CMA_USD)



def format_percent(df):
    df_formatted = df.copy()
    last_col = df_formatted.columns[-1]  # 마지막 열 이름

    for col in df_formatted.columns:
        df_formatted[col] = pd.to_numeric(df_formatted[col], errors='coerce').astype(float)
        if col == last_col:
            df_formatted[col] = df_formatted[col].apply(lambda x: "{:.1f}".format(x))  # 마지막 열은 소수점 1자리
        else:
            df_formatted[col] = df_formatted[col].apply(lambda x: "{:.2%}".format(x))  # 나머지는 퍼센트
    return df_formatted


df_CMA_KRW = format_percent(df_CMA_KRW)
df_CMA_USD = format_percent(df_CMA_USD)




# 포트폴리오의 월간 수익률을 최대화하는 함수======================

def optimize_Glide(df_R_M_KRW, BM_list, target_volatility=0.08, TE_target=0.1, total_weight=1):
    
    # 자산들의 월간 수익률 데이터만 추출
    returns = df_R_M_KRW[BM_list].copy()
    
    
    # 월간 ACWI 수익률 데이터
    BM_return = (
                 df_R_M_KRW["ACWI"]*0.6 + df_R_M_KRW["BND"]*0.4
                ).copy()
    
    
    MP_return = (
        (
        ( df_R_M_KRW["VUG"] + df_R_M_KRW["VTV"] )*0.674
        + df_R_M_KRW["VEA"]*0.2187
        + df_R_M_KRW["VWO"]*0.117
        )*0.6
        
        + df_R_M_KRW["273130"]*0.4
        ).copy()
    
    # 평균 수익률 벡터
    Glide_return = returns.mean()
    
    # 월간 수익률 공분산 행렬
    cov_matrix = returns.cov()
    
    # 초기 비중을 동일하게 설정
    num_assets = len(BM_list)
    initial_weights = np.ones(num_assets) / num_assets
    
    # 제약 조건 함수 정의
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 12, weights)))  # 연간 변동성 (월간 수익률 변동성 * sqrt(12))
    
    def TE(weights):
        portfolio_returns = np.dot(weights, returns.T)
        TE = np.sqrt(((portfolio_returns - BM_return)**2).mean()) * np.sqrt(12)  # 연간 트래킹 에러
        return TE
    
    # 포트폴리오의 예상 수익률
    def portfolio_return(weights):
        return np.dot(weights, Glide_return)
    
    # 제약 조건 설정
    constraints = [
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - total_weight},  # 비중 합계는 total_weight이어야 함
        {'type': 'ineq', 'fun': lambda weights: target_volatility - portfolio_volatility(weights)},  # 변동성 <= 목표 변동성
        {'type': 'ineq', 'fun': lambda weights: TE_target - TE(weights)}  # 트래킹 에러 <= 목표 트래킹 에러
    ]
    
    # 자산의 비중은 0 이상이어야 함 (숏 포지션 허용 X)
    bounds = [(0, total_weight) for _ in range(num_assets)]
    
    # 목적 함수는 수익률을 최대화하는 것이므로 음수로 만들어서 minimize 함수로 최대화
    def objective(weights):
        return -portfolio_return(weights)
    
    options = {'disp': True, 'maxiter': 500}
    optimized_result = minimize(
        objective, 
        initial_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints, 
        options=options) 


    # 최적화된 비중을 소수점 5자리로 반올림
    if optimized_result.success:
        optimized_weights = np.round(optimized_result.x, 5)  # 소수점 5자리로 반올림
        df_weights = pd.DataFrame({'Asset': BM_list, 'Weight': optimized_weights})
        return df_weights  # 데이터프레임으로 반환
    else:
        raise ValueError("Optimization did not converge")

#===============================================================









# Glidepath 주식/채권 투자비중 뽑아내기 : ACWI와 한국채권===========================================
BM_list = ["ACWI", "273130"]

# 빈티지 리스트와 해당 target_volatility 값을 딕셔너리로 설정
Vintage = {
    "TIF": 0.04,
    "2030": 0.05,
    "2035": 0.055,
    "2040": 0.06,
    "2045": 0.065,
    "2050": 0.07,
    "2055": 0.075,
    "2060": 0.08
}


# 최적화된 비중을 저장할 리스트=============================
results = []

# 빈티지별로 최적화 실행
for vintage, volatility in Vintage.items():
    optimized_weights = optimize_Glide(
        df_R_M_KRW, 
        BM_list=BM_list,  # ACWI와 BND를 최적화
        target_volatility=volatility,  # 빈티지에 따른 target_volatility 값 사용
        TE_target=0.5
    )
    
    # 각 빈티지에서 최적화된 ACWI와 BND 비중을 추출
    W_ACWI = optimized_weights.loc[optimized_weights['Asset'] == 'ACWI', 'Weight'].values[0]
    W_BND = optimized_weights.loc[optimized_weights['Asset'] == '273130', 'Weight'].values[0]
    
    # 결과를 딕셔너리 형태로 저장
    results.append({
        'Vintage': vintage,
        'W_ACWI': W_ACWI,
        '273130': W_BND
    })

# 결과를 데이터프레임으로 변환
df_Glide = pd.DataFrame(results)

# 출력
print("df_Glide=================", df_Glide)
#=================================================================




# df_interpolated = df_Glide.interpolate(method='polynomial', order=2)




def optimize_MP(df_R_M_KRW, MP_list, TE_target=0.05, total_weight=1):
    # 자산들의 월간 수익률 데이터만 추출
    returns = df_R_M_KRW[MP_list].copy()
    
    MP_return = (
        (df_R_M_KRW["VUG"] + df_R_M_KRW["VTV"]) * 0.674
        + df_R_M_KRW["VEA"] * 0.2187
        + df_R_M_KRW["VWO"] * 0.117
        ) * 0.95  # 주식 부분
    + df_R_M_KRW["GLD"] * 0.05  # GLD
    

    # 평균 수익률 벡터
    expected_return = returns.mean()

    # 월간 수익률 공분산 행렬
    cov_matrix = returns.cov()

    # 초기 비중을 동일하게 설정
    num_assets = len(MP_list)
    initial_weights = np.ones(num_assets) / num_assets

    # 제약 조건 함수 정의
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 12, weights)))  # 연간 변동성 (월간 수익률 변동성 * sqrt(12))
    
    def TE(weights):
        portfolio_returns = np.dot(weights, returns.T)
        TE = np.sqrt(((portfolio_returns - MP_return*total_weight)**2).mean()) * np.sqrt(12)  # 연간 트래킹 에러
        return TE

    # 포트폴리오의 변동성 대비 수익률 계산
    def portfolio_sharpe(weights):
        return np.dot(weights, expected_return) / portfolio_volatility(weights)  # 변동성 대비 수익률 (샤프 비율)

    # 제약 조건 설정 (비중 합계는 total_weight이어야 하고, 트래킹 에러는 목표 이하)
    constraints = [
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - total_weight},  # 비중 합계는 total_weight이어야 함
        {'type': 'ineq', 'fun': lambda weights: TE_target - TE(weights)},
        {'type': 'ineq', 'fun': lambda weights: total_weight * 0.05 - weights[MP_list.index("GLD")]},  # GLD <= total_weight * 0.05
        # {'type': 'ineq', 'fun': lambda weights: weights[MP_list.index("VUG")] - weights[MP_list.index("VTV")] },  # VUG >= VTV

    ]

    # 자산의 비중은 0 이상 total_weight 이하이어야 함 (숏 포지션 허용 X)
    bounds = [(0, total_weight) for _ in range(num_assets)]


    # 목적 함수는 샤프 비율(변동성 대비 수익률)을 최대화해야 함 -> minimize에서 -를 붙여서 최대화
    def objective(weights):
        return -portfolio_sharpe(weights)

    options = {'disp': True, 'maxiter': 500}
    optimized_result = minimize(
        objective, 
        initial_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints, 
        options=options)

    # 최적화된 비중을 소수점 5자리로 반올림
    if optimized_result.success:
        optimized_weights = np.round(optimized_result.x, 5)  # 소수점 5자리로 반올림
        df_weights = pd.DataFrame({'Asset': MP_list, 'Weight': optimized_weights})
        return df_weights  # 데이터프레임으로 반환
    else:
        raise ValueError("Optimization did not converge")

#===============================================================



MP_list=["VUG", "VTV", "VEA", "VWO", "GLD"]



# 빈티지 리스트
vintages = df_Glide['Vintage'].unique()  # 빈티지 목록 (예: 'TIF', '2030', '2035' 등)

# 최적화 결과를 저장할 리스트
results = []

# 각 빈티지에 대해 최적화 수행 및 '273130' 자산 추가
for vintage in vintages:
    # 각 빈티지별 ACWI 비중을 가져와서 최적화 수행
    W_Equity = optimize_MP(
        df_R_M_KRW, 
        MP_list=MP_list, 
        TE_target=0.1,
        total_weight= df_Glide.loc[df_Glide['Vintage'] == vintage, 'W_ACWI'].values[0]
    )
    
    # '273130' 자산 추가 및 Weight 값 설정
    W_Equity.loc[len(W_Equity)] = ['273130', df_Glide.loc[df_Glide['Vintage'] == vintage, '273130'].values[0]]
    
    # 빈티지명 추가
    W_Equity['Vintage'] = vintage
    
    # 결과를 리스트에 추가
    results.append(W_Equity)

# 결과를 하나의 데이터프레임으로 병합
df_final = pd.concat(results).reset_index(drop=True)
df_final = df_final.pivot(index='Asset', columns='Vintage', values='Weight')

# 행과 열의 순서를 MP_list와 Vintage 순서대로 재배열
df_final = df_final.reindex(index=MP_list, columns=list(Vintage.keys()))
df_final.loc['합계'] = df_final.sum()

# 최종 결과 출력
print(df_final)




# # df_weights_table을 변환된 데이터로 적용
df_weights_table = format_percent(df_final)
df_weights_table = df_weights_table.reset_index()   #인덱스도 열로 리셋


# save_excel(df_weights_table, "df_weights_table")









# df 형태는 : 0 범례 1 E(R)_KRW  2 Vol_KRW  3위험대비수익률(KRW)

# 데이터의 이름 가져오기
legend_krw = df_CMA_KRW.iloc[0].tolist()  # 첫 번째 행은 그래프의 범례로 사용될 이름들을 포함합니다.
legend_usd = df_CMA_USD.iloc[0].tolist()

# 실제 데이터 가져오기
# df 형태는 : 0 E(R)_KRW  1 Vol_KRW  2 위험대비수익률(KRW)
df_data_krw = df_CMA_KRW.iloc[1:, :]  # 데이터의 첫 번째 행은 범례이므로 제외합니다.
df_data_usd = df_CMA_USD.iloc[1:, :]


# 컬럼명 변경 (기존 열 이름 → 새 열 이름으로 명시적으로 지정)
df_data_krw.columns = ['기대수익률', '변동성', '수익률/위험']
df_data_usd.columns = ['기대수익률', '변동성', '수익률/위험']


# %문자 제거하고 퍼센트를 제거하고 숫자로 변환 후 100으로 나눔
df_data_krw = df_data_krw.apply(
    lambda col: pd.to_numeric(col.str.replace('%', ''), errors='coerce') / 100
    if col.dtypes == 'object' else col
)

df_data_usd = df_data_usd.apply(
    lambda col: pd.to_numeric(col.str.replace('%', ''), errors='coerce') / 100
    if col.dtypes == 'object' else col
)

print("df_data_krw==================", df_data_krw)



# 컬럼 이름 저장
col_krw = '수익률/위험'
col_usd = '수익률/위험'


# 최소/최대값 계산 및 버블 사이즈
min_krw, max_krw = df_data_krw[col_krw].min(), df_data_krw[col_krw].max()
min_usd, max_usd = df_data_usd[col_usd].min(), df_data_usd[col_usd].max()


size_scale = 300
if max_krw == min_krw:
    bubble_sizes_krw = [30 for _ in df_data_krw[col_krw]]  # 최소 기본 크기
else:
    bubble_sizes_krw = ((df_data_krw[col_krw] - min_krw) / (max_krw - min_krw) * size_scale).tolist()

if max_usd == min_usd:
    bubble_sizes_usd = [30 for _ in df_data_usd[col_usd]]
else:
    bubble_sizes_usd = ((df_data_usd[col_usd] - min_usd) / (max_usd - min_usd) * size_scale).tolist()




print("df_data_krw***********************", df_data_krw)


trace_krw = go.Scatter(
    x=df_data_krw['변동성'],  # 가로축 데이터:
    y=df_data_krw['기대수익률'],  # 세로축 데이터: 
    mode='markers',
    marker=dict(
        size=bubble_sizes_krw,  # 버블의 크기: 
        color='#3762AF',  # 1번 그래프 버블 색상  
    )
)

trace_usd = go.Scatter(
    x=df_data_usd['변동성'],  # 가로축 데이터: 
    y=df_data_usd['기대수익률'],  # 세로축 데이터: 
    mode='markers',
    marker=dict(
        size=bubble_sizes_usd,  # 버블의 크기: 데이터 테이블의 5번째 열의 표준화된 값으로 설정
        color='#630',  # 2번 그래프 버블 색상
    )
)



# 레이아웃 생성
layout = go.Layout(
    # title='자산군별 위험대비수익률',
    xaxis=dict(
        title='변동성',
        range=[0,0.35],
        tickformat='.1%',   # y축의 범위를 0부터 시작하도록 설정합니다.
    ),  # 가로축 레이블
    yaxis=dict(
        title='기대수익률',  # y축의 제목을 설정합니다.
        range=[0, None],
        tickformat='.1%',   # y축의 범위를 0부터 시작하도록 설정합니다.
    ),
    width=700,  # 그래프의 가로 크기
    height=500,  # 그래프의 세로 크기
    margin=dict(l=50, r=100, t=1, b=1),  # 마진 설정
)


# 그래프 생성
fig_krw = go.Figure(data=[trace_krw], layout=layout)
fig_usd = go.Figure(data=[trace_usd], layout=layout)






# 데이터프레임을 버블 크기 열을 기준으로 내림차순 정렬합니다.
df_sorted_krw = df_data_krw.sort_values(by=df_data_krw.columns[2], ascending=False)
df_sorted_usd = df_data_usd.sort_values(by=df_data_usd.columns[2], ascending=False)

# 상위 7개의 데이터를 추출합니다.
top_names_krw = df_sorted_krw.head(7).index
top_names_usd = df_sorted_usd.head(7).index


# 상위 7개의 이름을 그래프에 표시합니다.
text_annotations_krw = []
text_annotations_usd = []

# KRW용 텍스트 어노테이션
for name in top_names_krw:
    x_value = df_sorted_krw.loc[name, df_sorted_krw.columns[1]]   # 변동성 (x축)
    y_value = df_sorted_krw.loc[name, df_sorted_krw.columns[0]]   # 수익률 (y축)
    annotation = go.Scatter(
        x=[x_value],
        y=[y_value],
        mode='text',
        text=name,
        showlegend=False,
        textposition='middle right',
        textfont=dict(size=10, color='black')
    )
    text_annotations_krw.append(annotation)

# USD용 텍스트 어노테이션
for name in top_names_usd:
    x_value = df_sorted_usd.loc[name, df_sorted_usd.columns[1]]   # 변동성 (x축)
    y_value = df_sorted_usd.loc[name, df_sorted_usd.columns[0]]   # 수익률 (y축)
    annotation = go.Scatter(
        x=[x_value],
        y=[y_value],
        mode='text',
        text=name,
        showlegend=False,
        textposition='middle right',
        textfont=dict(size=10, color='black')
    )
    text_annotations_usd.append(annotation)


# 그래프 데이터에 텍스트 어노테이션을 추가합니다.
fig_krw.add_traces(text_annotations_krw)
fig_usd.add_traces(text_annotations_usd)





# Flask 서버 생성
server = Flask(__name__)
app = dash.Dash(__name__, server=server)
app.title = 'TDF_Optimization'



# 라인 그래프 생성 함수=========================================


# 스타일 설정 딕셔너리
graph_style = {
    'width': '60%', 
    'height': '450px', 
    'margin': 'auto',
    'display': 'flex',
    'justify-content': 'center',  # 가로 방향 가운데 정렬
    'text-align': 'center',
    'align-items': 'center'  # 세로 방향 가운데 정렬
}



# 앱 레이아웃 정의
app.layout = html.Div([
    html.H3("Optimized Weights Table", 
        style={'margin': 'auto', 'textAlign': 'center'}  # 가운데 정렬
    ),

    dash_table.DataTable(
        id='weights-table',
        columns=[{'name': col, 'id': col} for col in df_weights_table.columns],
        data=df_weights_table.to_dict('records'),
        style_table={
            'overflowX': 'auto', 
            'width': '60%',  # 테이블 너비를 75%로 설정
            'margin': 'auto'},  # 가운데 정렬
        style_cell={'textAlign': 'center', 'font-family': 'Arial', 'fontSize': '13px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#3762AF', 'color': 'white'},
    ),


    # 구분 선 표시
    # html.Hr(),

    html.H3("Portfolio Weight", 
        style={'margin': 'auto', 'textAlign': 'center'}  # 가운데 정렬
    ),

    dcc.Graph(
        id='line-chart',
        figure={
            'data': [
                go.Scatter(
                    x=df_final.columns,  # 포트폴리오 기간
                    y=df_final.loc[asset],  # 인덱스로 자산 가중치 추출
                    mode='lines+markers',
                    name=asset
                ) for asset in df_final.index  if asset != '합계'  # 🔥 '합계' 제외
            ],
            'layout': go.Layout(
                title='MP Weights',
                xaxis={'title': 'Vintage'},
                yaxis={'title': 'Weight', 'tickformat': '.0%'},
                template='plotly_white'
            )
        }, 
        style=graph_style
    ),



    # df_CMA_KRW 테이블 추가
    html.H3("CMA KRW Table", 
        style={'margin': 'auto', 'textAlign': 'center'}  # 가운데 정렬
    ),
    dash_table.DataTable(
        id='cma-krw-table',
        columns=[{'name': col, 'id': col} for col in df_CMA_KRW.reset_index().columns],
        data=df_CMA_KRW.reset_index().to_dict('records'),  # 인덱스 열을 포함하여 데이터를 표시
        style_table={
            'overflowX': 'auto', 
            'width': '60%',  # 테이블 너비를 75%로 설정
            'margin': 'auto'},  # 가운데 정렬
        style_cell={'textAlign': 'center', 'font-family': 'Arial', 'fontSize': '13px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#3762AF', 'color': 'white'},
    ),

    # df_CMA_USD 테이블 추가
    html.H3("CMA USD Table", 
        style={'margin': 'auto', 'textAlign': 'center'}  # 가운데 정렬
    ),
    dash_table.DataTable(
        id='cma-usd-table',
        columns=[{'name': col, 'id': col} for col in df_CMA_USD.reset_index().columns],
        data=df_CMA_USD.reset_index().to_dict('records'),  # 인덱스 열을 포함하여 데이터를 표시
        style_table={
            'overflowX': 'auto', 
            'width': '60%',  # 테이블 너비를 60%로 설정
            'margin': 'auto'  # 가운데 정렬
        },
        style_cell={'textAlign': 'center', 'font-family': 'Arial', 'fontSize': '13px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#3762AF', 'color': 'white'},
    ),



    html.Div([
        # 첫 번째 그래프
        html.Div([
            html.H3('2025 LTCMA(KRW))', style={'text-align': 'center'}),
            dcc.Graph(
                id='bubble-chart-krw',
                figure=fig_krw,
                style={'width': '70vh', 'height': 'auto'}  # 그래프에 스타일을 적용합니다.
            )
        ], style={'display': 'inline-block', 'margin-right': '20px'}),  # 그래프를 가로로 정렬합니다.

        # 두 번째 그래프
        html.Div([
            html.H3('2025 LTCMA(USD))', style={'text-align': 'center'}),
            dcc.Graph(
                id='bubble-chart-usd',
                figure=fig_usd,
                style={'width': '70vh', 'height': 'auto'}  # 그래프에 스타일을 적용합니다.
            )
        ], style={'display': 'inline-block'}),  # 그래프를 가로로 정렬합니다.
    ], style={
        'margin': 'auto',
        'justifyContent': 'center',
        'textAlign': 'center',
    }),





])



# 기본 포트 설정 ============================= 여러개 실행시 충돌 방지

DEFAULT_PORT = 8051

def find_available_port(start_port=DEFAULT_PORT, max_attempts=10):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))  # 실제 바인딩을 시도
                return port  # 사용 가능한 포트 반환
            except OSError:
                continue  # 이미 사용 중이면 다음 포트 확인
    raise RuntimeError("사용 가능한 포트를 찾을 수 없습니다.")

port = find_available_port()
print(f"사용 중인 포트: {port}")  # 디버깅용


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=port)

# ==================================================================
