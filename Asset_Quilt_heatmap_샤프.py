import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pickle
import os
import yfinance as yf
from pykrx import stock as pykrx
import requests
import concurrent.futures
from openpyxl import Workbook
import numpy as np
from io import BytesIO
import base64
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import random



#엑셀 저장=======================================================
def save_excel(df, sheetname, index_option=None):

    path = rf'C:\Covenant\data\Asset_Quilt_heatmap.xlsx'
    # 파일이 없는 경우 새 Workbook 생성
    if not os.path.exists(path):
        wb = Workbook()
        wb.save(path)
        print(f"새 파일 '{path}' 생성됨.")
    
    # 인덱스를 날짜로 변환 시도
    try:
        # index_option이 None일 경우 인덱스를 포함하고 날짜 형식으로 저장
        if index_option is None or index_option:  # 인덱스를 포함하는 경우
            df.index = pd.to_datetime(df.index, errors='coerce')
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

    #저장하고 원래대로 돌려놓음
    df.index = pd.to_datetime(df.index, errors='coerce')

# =================================================================


# 주식 및 채권 코드 정의
주식 = {
    '미국성장주': 'VUG',
    '미국가치주': 'VTV',
    '선진국주식': 'VEA',
    '이머징주식': 'VWO',
    '독일주식': 'EWG',
    '일본주식': 'EWJ',
    '중국주식': 'MCHI',
    '한국주식': '105190.KS',
    
    '금': 'GLD',
    '원유': 'USO',
}


채권 = {
    '한국채권': '273130.KS',
    # '한국국고채10년': '365780.KS',
}


BM = {
    '글로벌주식' : 'ACWI', 
    '글로벌채권' : 'BND',
    # '원/달러 환율' : 'KRW=X',
    # '자산배분' : ''
}

code_dict = {**주식, **채권, **BM}
code = list(set(code_dict.values()))



# 캐싱 경로 및 만료 시간 설정
cache_price = r'C:\Covenant\data\연도별자산군별수익률.pkl'
cache_expiry = timedelta(days=30)


# 데이터 가져오기 함수
def fetch_data(code, start, end):
    try:
        if isinstance(code, int) or code.isdigit() :
            if len(code) == 5:
                code = '0' + code
            df_price = pykrx.get_market_ohlcv_by_date(start, end, code)['종가']
        else:
            session = requests.Session()
            session.verify = False  # SSL 인증서 검증 비활성화
            yf_data = yf.Ticker(code, session=session)
            df_price = yf_data.history(start=start, end=end)['Close']
            df_price = df_price.tz_localize(None)  # 타임존 제거

        df_price = pd.DataFrame(df_price)
        df_price.columns = [code]
        df_price.index = pd.to_datetime(df_price.index).strftime('%Y-%m-%d')  # 인덱스를 문자열 형식으로 변환
        df_price = df_price.sort_index(ascending=True)
        
        
        return df_price
    
    except Exception as e:
        print(f"Error fetching data for {code}: {e}")
        return None


# 캐시 데이터 로딩 및 데이터 병합 처리 함수
def Func(code, start, end, batch_size=10):
    if os.path.exists(cache_price):
        cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_price))
        if datetime.now() - cache_mtime < cache_expiry:
            with open(cache_price, 'rb') as f:
                print("Loading data from cache...")
                return pickle.load(f)

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
    price_data = price_data.sort_index(ascending=True)
    print("price_data=================\n", price_data)

    with open(cache_price, 'wb') as f:
        pickle.dump(price_data, f)
        print("Data cached.")

    return price_data



start = (datetime.today() - relativedelta(years=7) - timedelta(days=1) ).strftime('%Y-%m-%d')
end = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')


# ETF 가격 데이터 가져오기
ETF_price = Func(code, start, end, 30)
ETF_price = ETF_price.ffill()
ETF_price.index = pd.to_datetime(ETF_price.index)  # 인덱스를 DatetimeIndex로 변환
ETF_price['자산배분'] = ETF_price['ACWI']*0.6 + ETF_price['BND']*0.4
# save_excel(ETF_price, 'ETF_price', index_option=False)


# df_지수.index = pd.to_datetime(df_지수.index, errors='coerce')  # 날짜 형식으로 변환
# df_지수.index = df_지수.index.strftime('%Y-%m-%d')


ETF_R = ETF_price.pct_change()
print("ETF_R=================\n", ETF_R)





# 연도별 수익률 계산==========================================
연도별수익률 = ETF_price.resample('YE').last()
# save_excel(연도별수익률, '연도별수익률', index_option=False)


연도별수익률 = 연도별수익률.pct_change()
연도별수익률.index = 연도별수익률.index.strftime('%Y')  #그룹바이를 안하면 연도 형식지정



# '자산배분'을 별도로 처리하려면, 연도별수익률.columns의 값이 '자산배분'인 경우를 예외 처리해주면 됩니다.
# 연도별수익률의 열 이름을 code_dict의 키로 설정
연도별수익률.columns = [
    list(code_dict.keys())[list(code_dict.values()).index(col)] if col in code_dict.values() else col
    for col in 연도별수익률.columns
]


# 딕셔너리 코드 값에 포함된 자산군만 필터링 (자산배분 제외) - 중복 제거
valid_columns = list(dict.fromkeys([col for col in 연도별수익률.columns if col in code_dict])) + ['자산배분']

# 유효한 자산군만 포함한 연도별수익률
연도별수익률 = 연도별수익률[valid_columns]

print("연도별수익률==============", 연도별수익률)


연도별수익률 = 연도별수익률.T






# 각 열을 내림차순 정렬
sorted_columns = {}

# 각 열을 순차적으로 처리하여 내림차순 정렬
for col in 연도별수익률.columns:
    # 열을 내림차순 정렬
    sorted_column = 연도별수익률[[col]].sort_values(by=col, ascending=False)
    
    # 정렬된 열을 별도의 DataFrame으로 저장
    sorted_columns[col] = sorted_column

# 결과 확인
for col, sorted_수익률 in sorted_columns.items():
    print(f"sorted_수익률_{col}=================\n", sorted_수익률)

print("sorted_columns=================\n", sorted_columns)
# sorted_columns


    



# 수익률_table_data를 위한 리스트 초기화
수익률_table_data = []

# 각 열에 대해 내림차순으로 정렬된 데이터프레임을 처리
# 각 열을 별도로 처리하여 자산군과 수익률을 순차적으로 저장
for i in range(len(sorted_columns[next(iter(sorted_columns))])):  # 첫 번째 열에서 데이터 길이를 기준으로 반복
    table_row = {}  # 각 행을 위한 빈 딕셔너리
    
    # 각 열에 대해 순차적으로 자산군과 수익률을 추가
    for col, sorted_수익률 in sorted_columns.items():
        row = sorted_수익률.iloc[i]  # 각 열의 i번째 행을 가져옴
        table_row['자산군'] = row.name  # 자산군을 '자산군' 열에 추가, row.name은 인덱스 값을 가져옴
        table_row[col] = f'{row.name}\n{row[col]*100:.2f}%'  # 자산군명과 수익률을 텍스트로 추가
        
    # 수익률_table_data에 해당 행 추가
    수익률_table_data.append(table_row)


print("수익률_table_data=================\n", 수익률_table_data)




# 수익률_table_data를 데이터프레임으로 변환
df_수익률 = pd.DataFrame(수익률_table_data)
print("df_수익률=================\n", df_수익률)

# DataTable 열 이름 설정 (연도별 수익률의 열 이름 사용)
columns = [{'name': f'{col}년', 'id': col} for col in 연도별수익률.columns]

# unique(): '자산군' 열에 나타나는 중복되지 않은 고유한 값들을 반환합니다.
# 자산군 리스트 추출
asset_classes = df_수익률['자산군'].unique()

asset_color = {
    '자산배분': '#000000',  # 다크 블루 (신뢰감, 고급스러움)
    '글로벌주식': '#4A90E2',  # 푸른색 (세련되고 현대적인 느낌)
    '글로벌채권': '#1F78D1',  # 다크 청록색 (안정적이고 신뢰감 있는 느낌)
    '미국성장주': '#e74c3c',  # 강렬한 붉은색 (도전적이고 현대적인 느낌)
    '미국가치주': '#8B4513',  # 다크 브라운 (안정성, 신뢰성)
    '선진국주식': '#D4AF37',  # 밝은 회색 (세련된 느낌, 가벼운 톤)
    '이머징주식': '#7F8C8D',  # 어두운 회색 (모던하고 고급스러운 느낌)
    '한국주식': '#2980B9',  # 진한 파란색 (안정적이고 신뢰감 있는 느낌)
    '중국주식': '#CD7F32',  # 네이비 블루 (신뢰감, 안정성)
    '일본주식': '#1ABC9C',  # 민트색 (산뜻하고 현대적인 느낌)
    '독일주식': '#16A085',  # 청록색 (세련된 느낌)
    '한국국고채10년': '#E67E22',  # 오렌지 (온화하고 따뜻한 느낌)
    '금': '#F1C40F',  # 금색 (고급스러움, 부유함)
    # '금_다크': '#D4AF37',  # 다크 골드 (고급스럽고 안정감)
    # '금_라이트': '#F9E79F',  # 라이트 골드 (밝고 부드러운 느낌)
    # '금_오렌지': '#E67E22',  # 오렌지 (따뜻하고 온화한 느낌)
    # '금_브론즈': '#CD7F32',  # 브론즈 (세련되고 고급스러운 느낌)
    '원유': '#34495E',  # 다크 회색 (강렬하고 안정적인 느낌)
    '한국채권': '#4682B4',  # 스틸 블루 (안정감과 신뢰감을 주는 색상)
    '원/달러 환율': '#F39C12',  # 청록색 (자연적이고 신뢰감 있는 느낌)
}



# asset_color = {
#     '자산배분': '#000000',  # 다크 블루 (신뢰감, 고급스러움)
#     '글로벌주식': '#4A90E2',  # 푸른색 (세련되고 현대적인 느낌)
#     '글로벌채권': '#1F78D1',  # 다크 청록색 (안정적이고 신뢰감 있는 느낌)
#     '미국성장주': '#E74C3C',  # 강렬한 붉은색 (도전적이고 현대적인 느낌)
#     '미국가치주': '#8B4513',  # 다크 브라운 (안정성, 신뢰성)
#     '선진국주식': '#BDC3C7',  # 밝은 회색 (세련된 느낌, 가벼운 톤)
#     '이머징주식': '#7F8C8D',  # 어두운 회색 (모던하고 고급스러운 느낌)
#     '한국주식': '#2980B9',  # 진한 파란색 (안정적이고 신뢰감 있는 느낌)
#     '중국주식': '#400120',  # 네이비 블루 (신뢰감, 안정성)
#     '일본주식': '#1ABC9C',  # 민트색 (산뜻하고 현대적인 느낌)
#     '독일주식': '#16A085',  # 청록색 (세련된 느낌)
#     '한국국고채10년': '#E67E22',  # 오렌지 (온화하고 따뜻한 느낌)
#     '금': '#F1C40F',  # 금색 (고급스러움, 부유함)
#     '원유': '#34495E',  # 다크 회색 (강렬하고 안정적인 느낌)
#     '한국채권': '#4682B4',  # 스틸 블루 (안정감과 신뢰감을 주는 색상)
#     '원/달러 환율': '#F39C12',  # 청록색 (자연적이고 신뢰감 있는 느낌)
# }


# asset_color = {
#     '자산배분': '#000000',  
#     '글로벌주식': '#003366',  # 보라색 (고급스러운 느낌)
#     '글로벌채권': '#3498db',  # 청록색 (세련된 느낌)
#     '미국성장주': '#e74c3c',  # 붉은색 (강렬하고 현대적인 느낌)
#     '미국가치주': '#8B4513',  # 노란색 (따뜻하고 친근한 느낌)
#     '선진국주식': '#95a5a6',  # 회색 (세련된 느낌)
#     '이머징주식': '#7f8c8d',  # 어두운 회색 (모던하고 고급스러운 느낌)
#     '한국주식': '#2980b9',  # 파란색 (깊이 있고 안정적인 느낌)
#     '중국주식': '#8e44ad',  # 진한 보라색 (독특하고 강렬한 느낌)
#     '일본주식': '#1abc9c',  # 민트색 (산뜻하고 현대적인 느낌)
#     '독일주식': '#16a085',  # 청록색 (세련된 느낌)
#     '한국국고채10년': '#e67e22',  # 주황색 (온화하고 따뜻한 느낌)
#     '금': '#f1c40f',  # 밝은 노란색 (부유함과 고급스러움)
#     '원유': '#34495e',  # 짙은 회색 (강렬하고 안정적인 느낌)
#     '한국채권': '#3682B4',  # 연두색 (신선하고 생동감 있는 느낌)
#     '원/달러 환율': '#f39c12',  # 청록색 (자연적이고 신뢰감 있는 느낌)
# }


# asset_color = {
#     '자산배분': '#000000',  # 검정색 (전문성)
#     '글로벌주식': '#003366',  # 네이비 블루 (신뢰감, 안정성)
#     '글로벌채권': '#333333',  # 차콜 그레이 (세련된 느낌)
#     '미국성장주': '#e74c3c',  # 붉은색 (강렬하고 현대적인 느낌)
#     '미국가치주': '#8B4513',  # 다크브라운 (고급스러운 느낌)
#     '선진국주식': '#006400',  # 다크 그린 (안정성)
#     '이머징주식': '#7f8c8d',  # 어두운 회색 (모던하고 고급스러운 느낌)
#     '한국주식': '#2980b9',  # 스카이 블루 (청렴한 느낌)
#     '중국주식': '#8e44ad',  # 진한 보라색 (독특하고 강렬한 느낌)
#     '일본주식': '#1abc9c',  # 민트색 (산뜻하고 현대적인 느낌)
#     '독일주식': '#16a085',  # 청록색 (세련된 느낌)
#     '한국국고채10년': '#FFD700',  # 골드 (고급스러움과 풍요)
#     '금': '#FFD700',  # 골드 (고급스러움과 풍요)
#     '원유': '#34495e',  # 짙은 회색 (강렬하고 안정적인 느낌)
#     '한국채권': '#4682B4',  # 스틸 블루 (신뢰감과 안정감)
#     '원/달러 환율': '#f39c12',  # 오렌지 (신뢰감과 자산 가치)
# }





# 변동성 계산=======================================================
ETF_R_W = ETF_price.resample('W').last().pct_change()
ETF_R_W = ETF_R_W.dropna(how='all', axis=1)

ETF_R_W.index = pd.to_datetime(ETF_R_W.index, errors='coerce')
ETF_R_W.index = ETF_R_W.index.strftime('%Y-%m-%d')  # 벡터화된 방식으로 날짜 포맷 변경


print("ETF_R_W=====================", ETF_R_W)
save_excel(ETF_R_W, 'ETF_R_W', index_option=None)


변동성 = ETF_R_W.groupby(ETF_R_W.index.year).std()*np.sqrt(52)

# 연도별수익률의 열 이름을 code_dict의 키로 설정
변동성.columns = [
    list(code_dict.keys())[list(code_dict.values()).index(col)] if col in code_dict.values() else col
    for col in 변동성.columns
]





# 딕셔너리 코드 값에 포함된 자산군만 필터링 
변동성_columns = [col for col in 변동성.columns if col in code_dict] + ['자산배분']
변동성 = 변동성[valid_columns]

변동성 = 변동성.T


# # # Then, you can define the columns for the volatility table
# columns_변동성 = [{'name': col, 'id': col} for col in 변동성.columns]  # Skip '자산군' column


# print("변동성=====================", 변동성)





# 각 열을 내림차순 정렬
변동성_sorted_columns = {}

# 각 열을 순차적으로 처리하여 내림차순 정렬
for col in 변동성.columns[1:]:
    # 열을 내림차순 정렬
    sorted_column = 변동성[[col]].sort_values(by=col, ascending=False)
    변동성_sorted_columns[col] = sorted_column

# 결과 확인
for col, sorted_변동성 in 변동성_sorted_columns.items():
    print(f"sorted_변동성_{col}=================\n", sorted_변동성)





# 변동성 DataFrame 생성
변동성_table_data = []

# 각 열에 대해 내림차순으로 정렬된 데이터프레임을 처리
# 각 열을 별도로 처리하여 자산군과 수익률을 순차적으로 저장
for i in range(len(변동성_sorted_columns[next(iter(변동성_sorted_columns))])):  # 첫 번째 열에서 데이터 길이를 기준으로 반복
    table_row = {}  # 각 행을 위한 빈 딕셔너리
    
    # 각 열에 대해 순차적으로 자산군과 수익률을 추가
    for col, sorted_변동성 in 변동성_sorted_columns.items():
        row = sorted_변동성.iloc[i]  # 각 열의 i번째 행을 가져옴
        table_row['자산군'] = row.name  # 자산군을 '자산군' 열에 추가, row.name은 인덱스 값을 가져옴
        table_row[col] = f'{row.name}\n{row[col]*100:.2f}%'  # 자산군명과 수익률을 텍스트로 추가
        
    # 수익률_table_data에 해당 행 추가
    변동성_table_data.append(table_row)
print("변동성_table_data=================\n", 변동성_table_data)

# 수익률_table_data를 데이터프레임으로 변환
df_변동성 = pd.DataFrame(변동성_table_data)
print("df_변동성==============", df_변동성)









# 샤프 지수 계산=======================================================
# # 연도별수익률의 열을 문자열로 변환
# 연도별수익률.columns = 연도별수익률.columns.astype(str)

# # 변동성의 열을 문자열로 변환
변동성.columns = 변동성.columns.astype(str)

# 공통 열 찾기
common_columns = 연도별수익률.columns.intersection(변동성.columns)
print(f"공통 열: {common_columns}")  # 공통 열 출력

# 공통 열이 없으면 종료
if common_columns.empty:
    print("공통 열이 없습니다. 확인이 필요합니다.")
else:
    # 공통 열로 데이터프레임을 merge
    merged_df = pd.merge(연도별수익률[common_columns], 변동성[common_columns], 
                         left_index=True, right_index=True, suffixes=('_수익률', '_변동성'))

    # merge 후 결과 출력
    print(f"merge된 데이터프레임:\n{merged_df.head()}")  # merged_df가 제대로 만들어졌는지 확인

    # 샤프 지수 계산 (수익률 / 변동성)
    샤프 = merged_df[[col + '_수익률' for col in common_columns]].values / merged_df[[col + '_변동성' for col in common_columns]].values

    # 샤프 지수를 새로운 DataFrame으로 저장
    샤프_df = pd.DataFrame(샤프, index=merged_df.index, columns=[col + '년' for col in common_columns])

    # 결과 출력
    print("샤프_df=================", 샤프_df)
    


# 각 열을 내림차순 정렬
sorted_샤프 = {}

# 각 열을 순차적으로 처리하여 내림차순 정렬
for col in 샤프_df.columns:
    sorted_column = 샤프_df[[col]].sort_values(by=col, ascending=False)
    sorted_샤프[col] = sorted_column

# 결과 확인
for col, df in sorted_샤프.items():
    print(f"sorted_샤프_{col}=================\n", df)


# 샤프_table_data 생성
샤프_table_data = []

# 정렬된 데이터프레임 기반으로 데이터 생성
for i in range(len(next(iter(sorted_샤프.values())))):  # 첫 번째 열의 길이 기준
    table_row = {}
    for col, df in sorted_샤프.items():
        row = df.iloc[i]  # i번째 행 가져오기
        table_row[col] = f"{row.name}\n{row[col]:.2f}"
    샤프_table_data.append(table_row)

# DataFrame 변환
df_sorted_샤프 = pd.DataFrame(샤프_table_data)

# 결과 출력
print("df_sorted_샤프=================", df_sorted_샤프)



# 결과 출력
print("샤프_table_data=================")
for row in 샤프_table_data:
    print(row)

sharpe_columns = [{'name': col, 'id': col} for col in df_sorted_샤프.columns]





# 스타일 설정
style_data_conditional = []

# 각 열에 포함된 자산군을 찾아 색상을 적용
for col in df_수익률.columns[1:] :  # 첫 번째 컬럼은 자산군이므로 제외
    for asset in asset_classes:
        style_data_conditional.append({
            'if': {
                'filter_query': f'{{{col}}} contains "{asset}"',
                'column_id': col
            },
            'backgroundColor': asset_color.get(asset, '#FFFFFF'),  # 기본 색상은 흰색
            'color': 'white',  # 텍스트 색상은 흰색
        })


# 샤프_table_data 스타일 설정
style_data_conditional_sharpe = []

# 샤프_table_data에 포함된 자산군 색상 지정
for col in df_sorted_샤프.columns:
    for asset, color in asset_color.items():
        style_data_conditional_sharpe.append({
            'if': {
                'filter_query': f'{{{col}}} contains "{asset}"',
                'column_id': col
            },
            'backgroundColor': color,
            'color': 'white'
        })






# Dash 앱 생성
app = dash.Dash(__name__)

# Dash 레이아웃 설정
app.layout = html.Div(
    style={'width': '80%', 'margin': 'auto'},
    children=[
        html.H3("연도별 자산군별 수익률", style={'textAlign': 'center'}),
        
        dash_table.DataTable(
            id='asset_return_table',
            data=수익률_table_data,  # 테이블 데이터를 DataTable에 전달
            columns=columns[1:], # 첫번째 컬럼은 Nan이므로 제외
            style_table={'height': 'auto', 'overflowY': 'auto'},  # 테이블 높이 설정
            style_cell={
                'textAlign': 'center', 
                'verticalAlign': 'middle',  # 텍스트를 세로로 중앙 정렬
                'height': '60px',  # height를 3%로 설정
                'lineHeight': '2',  # 텍스트 높이를 2배로 설정
                'minWidth': '20px', 
                'maxWidth': '20px',
                'whiteSpace': 'pre-line',  # 줄바꿈을 위해 pre-line을 사용
                'border': 'none',  # 셀에 테두리 추가
                # 'fontWeight': 'bold',
            },
            style_header={
                'backgroundColor': 'white', 
                'fontWeight': 'bold',
                'border': 'none',  # 헤더 테두리 제거
                'font-size': '16px',  # 헤더 폰트 크기 설정
                },
            style_data_conditional=style_data_conditional  # 조건부 스타일 추가
        ),


        html.H1(" ", style={'textAlign': 'center'}),  # 제목 추가


# 변동성 테이블 추가
        html.H3("연간 자산군 변동성", style={'textAlign': 'center'}),  # 제목 추가
        dash_table.DataTable(
            id='annual_volatility_table',
            data=변동성_table_data,  # 연간 변동성 데이터
            columns=columns[1:],  # 연간 변동성 테이벌 첫째열 제거
            style_table={'height': 'auto', 'overflowY': 'auto'},  # 테이블 높이 설정
            style_cell={
                'textAlign': 'center', 
                'verticalAlign': 'middle',  # 텍스트를 세로로 중앙 정렬
                'height': '60px',  # height를 3%로 설정
                'lineHeight': '2',  # 텍스트 높이를 2배로 설정
                'minWidth': '20px', 
                'maxWidth': '20px',
                'whiteSpace': 'pre-line',  # 줄바꿈을 위해 pre-line을 사용
                'border': 'none',  # 셀에 테두리 추가
            },
            style_header={
                'backgroundColor': 'white', 
                'fontWeight': 'bold',
                'border': 'none',  # 헤더 테두리 제거
                'font-size': '16px',  # 헤더 폰트 크기 설정
                },
            style_data_conditional=style_data_conditional  # 조건부 스타일 추가
        ),


# 위험대비수익률 테이블 추가
        html.H3("위험대비수익률", style={'textAlign': 'center'}),  # 제목 추가
        dash_table.DataTable(
            id='sharpe_table',
            data=샤프_table_data,  
            columns=sharpe_columns[1:],  
            style_table={'height': 'auto', 'overflowY': 'auto'},  # 테이블 높이 설정
            style_cell={
                'textAlign': 'center', 
                'verticalAlign': 'middle',  # 텍스트를 세로로 중앙 정렬
                'height': '60px',  # height를 3%로 설정
                'lineHeight': '2',  # 텍스트 높이를 2배로 설정
                'minWidth': '20px', 
                'maxWidth': '20px',
                'whiteSpace': 'pre-line',  # 줄바꿈을 위해 pre-line을 사용
                'border': 'none',  # 셀에 테두리 추가
            },
            style_header={
                'backgroundColor': 'white', 
                'fontWeight': 'bold',
                'border': 'none',  # 헤더 테두리 제거
                'font-size': '16px',  # 헤더 폰트 크기 설정
                },
            style_data_conditional=style_data_conditional_sharpe    # 조건부 스타일 추가
        )


    ]
)


# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')
