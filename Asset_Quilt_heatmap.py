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


# =================================================================


# 주식 및 채권 코드 정의
주식 = {
    '미국성장주': 'VUG',
    '미국가치주': 'VTV',
    '선진국주식': 'VEA',
    '이머징주식': 'VWO',
    '독일주식': 'EWG',
    '일본주식': 'EWJ',
    '중국주식': 'CNYA',
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
    '원/달러 환율' : 'KRW=X',
    '자산배분' : ''
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
# save_excel(ETF_price, 'ETF_price', index_option=None)


# df_지수.index = pd.to_datetime(df_지수.index, errors='coerce')  # 날짜 형식으로 변환
# df_지수.index = df_지수.index.strftime('%Y-%m-%d')


ETF_R = ETF_price.pct_change()
print("ETF_R=================\n", ETF_R)

# BM_R ={}
# BM_R['자산배분'] = ETF_R['ACWI']*0.6 + ETF_R['BND']*0.4
# ETF_R['자산배분'] = BM_R['자산배분']


# print("ETF_R=================\n", ETF_R)

연도별수익률 = ETF_price.resample('YE').last()
연도별수익률['자산배분'] = 연도별수익률['ACWI']*0.6 + 연도별수익률['BND']*0.4
# save_excel(연도별수익률, '연도별수익률', index_option=False)


연도별수익률 = 연도별수익률.pct_change()
연도별수익률.index = 연도별수익률.index.strftime('%Y')



# '자산배분'을 별도로 처리하려면, 연도별수익률.columns의 값이 '자산배분'인 경우를 예외 처리해주면 됩니다.
# 연도별수익률의 열 이름을 code_dict의 키로 설정
연도별수익률.columns = [
    list(code_dict.keys())[list(code_dict.values()).index(col)] if col in code_dict.values() else col
    for col in 연도별수익률.columns
]

# 딕셔너리 코드 값에 포함된 자산군만 필터링 (자산배분 제외)
valid_columns = [col for col in 연도별수익률.columns if col in code_dict]

# 유효한 자산군만 포함한 연도별수익률
연도별수익률 = 연도별수익률[valid_columns]

# if '자산배분' in 연도별수익률.columns:
#     연도별수익률 = pd.concat([연도별수익률, 연도별수익률['자산배분']], axis=1)


연도별수익률 = 연도별수익률.T

# print("연도별수익률.index==============", 연도별수익률.index)





# 각 열을 내림차순 정렬
sorted_columns = {}

# 각 열을 순차적으로 처리하여 내림차순 정렬
for col in 연도별수익률.columns:
    # 열을 내림차순 정렬
    sorted_column = 연도별수익률[[col]].sort_values(by=col, ascending=False)
    
    # 정렬된 열을 별도의 DataFrame으로 저장
    sorted_columns[col] = sorted_column

# 결과 확인
for col, sorted_df in sorted_columns.items():
    print(f"sorted_df_{col}=================\n", sorted_df)

#sorted_df들이 데이터프레임으로 정의 되었음
    



# table_data를 위한 리스트 초기화
table_data = []

# 각 열에 대해 내림차순으로 정렬된 데이터프레임을 처리
# 각 열을 별도로 처리하여 자산군과 수익률을 순차적으로 저장
for i in range(len(sorted_columns[next(iter(sorted_columns))])):  # 첫 번째 열에서 데이터 길이를 기준으로 반복
    table_row = {}  # 각 행을 위한 빈 딕셔너리
    
    # 각 열에 대해 순차적으로 자산군과 수익률을 추가
    for col, sorted_df in sorted_columns.items():
        row = sorted_df.iloc[i]  # 각 열의 i번째 행을 가져옴
        table_row['자산군'] = row.name  # 자산군을 '자산군' 열에 추가, row.name은 인덱스 값을 가져옴
        table_row[col] = f'{row.name}\n{row[col]*100:.2f}%'  # 자산군명과 수익률을 텍스트로 추가
        
    # table_data에 해당 행 추가
    table_data.append(table_row)
print("table_data=================\n", table_data)

# table_data를 데이터프레임으로 변환
df_table = pd.DataFrame(table_data)

# 결과 확인
print("df_table=================\n", df_table)



# DataTable 열 이름 설정 (연도별 수익률의 열 이름 사용)
columns = [{'name': f'{col}년', 'id': col} for col in 연도별수익률.columns]


asset_classes = df_table['자산군'].unique()


asset_color = {
    '자산배분': '#2C3E50',  # 다크 블루 (신뢰감, 고급스러움)
    '글로벌주식': '#4A90E2',  # 푸른색 (세련되고 현대적인 느낌)
    '글로벌채권': '#1F78D1',  # 다크 청록색 (안정적이고 신뢰감 있는 느낌)
    '미국성장주': '#E74C3C',  # 강렬한 붉은색 (도전적이고 현대적인 느낌)
    '미국가치주': '#8B4513',  # 다크 브라운 (안정성, 신뢰성)
    '선진국주식': '#BDC3C7',  # 밝은 회색 (세련된 느낌, 가벼운 톤)
    '이머징주식': '#7F8C8D',  # 어두운 회색 (모던하고 고급스러운 느낌)
    '한국주식': '#2980B9',  # 진한 파란색 (안정적이고 신뢰감 있는 느낌)
    '중국주식': '#8E44AD',  # 진한 보라색 (독특하고 강렬한 느낌)
    '일본주식': '#1ABC9C',  # 민트색 (산뜻하고 현대적인 느낌)
    '독일주식': '#16A085',  # 청록색 (세련된 느낌)
    '한국국고채10년': '#E67E22',  # 오렌지 (온화하고 따뜻한 느낌)
    '금': '#F1C40F',  # 금색 (고급스러움, 부유함)
    '원유': '#34495E',  # 다크 회색 (강렬하고 안정적인 느낌)
    '한국채권': '#4682B4',  # 스틸 블루 (안정감과 신뢰감을 주는 색상)
    '원/달러 환율': '#F39C12',  # 청록색 (자연적이고 신뢰감 있는 느낌)
}


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







# 스타일 설정
style_data_conditional = []

# 각 열에 포함된 자산군을 찾아 색상을 적용
for col in df_table.columns[1:]:  # 첫 번째 컬럼은 자산군이므로 제외
    for asset in asset_classes:
        style_data_conditional.append({
            'if': {
                'filter_query': f'{{{col}}} contains "{asset}"',
                'column_id': col
            },
            'backgroundColor': asset_color.get(asset, '#FFFFFF'),  # 기본 색상은 흰색
            'color': 'white',  # 텍스트 색상은 흰색
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
            data=table_data,  # 테이블 데이터를 DataTable에 전달
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
                'font-size': '3',  # 헤더 폰트 크기 설정
                },
            style_data_conditional=style_data_conditional  # 조건부 스타일 추가
        )
    ]
)


# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')
