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




# 주식 및 채권 코드 정의
주식 = {
    'US Growth': 'VUG',
    'US Value': 'VTV',
    'DM ex US': 'VEA',
    'EM': 'VWO',
    'Gold': 'GLD',
}

채권 = {
    'KIS종합채권': '273130.KS',
    'KIS국고채10년': '365780.KS'
}

BM = {
    'ACWI' : 'ACWI', 
    'BND' : 'BND',
    '원/달러 환율' : 'KRW=X',
}

code_dict = {**주식, **채권, **BM}
code = list(code_dict.values()) 

# 캐싱 경로 및 만료 시간 설정
cache_price = r'C:\Covenant\TDF\data\디폴트옵션_백테스트_price.pkl'
cache_expiry = timedelta(days=1)

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






# ETF포커스_MP_Release_2024.xlsx 파일에서 df_MP 생성
path1 = r'C:\Covenant\TDF\data\ETF포커스_MP_Release_2024.xlsx'

try:
    df_MP = pd.read_excel(path1, sheet_name='Summary')
    print("Summary 시트 읽기 성공")
except Exception as e:
    print(f"파일 읽기 오류: {e}")

# df_MP 전처리
df_MP = df_MP.iloc[1:, 1:9]  # 필요 없는 행과 열 제외
df_MP.columns = df_MP.iloc[0]  # 첫 번째 행을 열 이름으로 설정
df_MP = df_MP.drop(1).reset_index(drop=True)  # 첫 번째 행 삭제 및 인덱스 리셋

# 주식, 채권 비율 계산 후 컬럼 추가
df_Glide = pd.DataFrame()
df_Glide['주식'] = df_MP[list(주식.keys())].sum(axis=1)
df_Glide['채권'] = df_MP[list(채권.keys())].sum(axis=1)

# 빈티지 열 처리
df_MP['Vintage'] = pd.to_numeric(df_MP['Vintage'], errors='coerce')
df_MP['Vintage'] = df_MP['Vintage'].fillna(0).astype(int)

print("df_MP================", df_MP)


def generate_df_weight(vintage_target, test_period, df_MP):
    # 현재 연도
    today = pd.Timestamp.today()
    current_year = today.year

    # df_weight 시작 날짜: test_period 연도 전
    df_weight_start = today - pd.DateOffset(years=test_period)

    # 빈 데이터프레임 생성 (인덱스를 df_weight_start부터 오늘까지의 일간 날짜로 설정하고, df_MP의 컬럼을 적용)
    df_weight = pd.DataFrame(
        index=pd.date_range(start=df_weight_start, end=today, freq='D'),
        columns=df_MP.columns
    )

    # 날짜 인덱스를 새로운 'Date' 열로 추가
    df_weight['Date'] = df_weight.index

    # DatetimeIndex에서 연도를 추출하여 'T' 값 계산
    df_weight['T'] = vintage_target + (current_year - df_weight.index.year)

    # df_MP와 병합
    df_weight = df_weight.merge(df_MP, left_on='T', right_on='Vintage', how='left')

    # 병합된 데이터프레임에서 'Vintage', 'T' 열 삭제
    df_weight.drop(columns=['Vintage', 'T'], inplace=True, errors='ignore')

    # 열 이름을 정리 (필요에 따라 _x, _y 제거)
    df_weight.columns = df_weight.columns.str.replace('_y', '', regex=False)
    df_weight.drop(columns=df_weight.columns[df_weight.columns.str.contains('_x')], inplace=True)

    df_weight.set_index('Date', inplace=True)
    df_weight.index = df_weight.index.strftime('%Y-%m-%d')
    df_weight = df_weight.bfill()

    print("df_weight================\n", df_weight)

    return df_weight






#엑셀 저장=======================================================
def save_excel(df, sheetname, index_option=None):
    
    # 파일 경로
    path = rf'C:\Covenant\TDF\data\TDF_디폴트옵션_백테스트.xlsx'

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






# Dash 앱 생성
app = dash.Dash(__name__)
app.title = 'TDF 백테스트'


# 레이아웃 정의
app.layout = html.Div(
    style={'width': '60%', 'margin': 'auto'},
    children=[
        html.H3("TDF 백테스트", style={'textAlign': 'center'}),

        # 빈티지 선택 드롭다운
        dcc.Dropdown(
            id='vintage-target',
            options=[{'label': str(year), 'value': year} for year in range(2025, 2081, 5)],
            value=2035,  # 기본값 설정
            placeholder="빈티지 연도 선택",
            style={'width': '50%', 'margin': '0 auto', 'textAlign': 'center'}
        ),

        html.Label('테스트 기간 (연도):', style={'textAlign': 'center'}),

        # 슬라이더를 감싸는 Div에 스타일 적용
        html.Div(
            dcc.Slider(
                id='test-period',
                min=1,
                max=20,
                step=1,
                value=20,  # 기본값 설정
                marks={i: f'{i}' for i in range(1, 21)},  # 슬라이더 마커 설정 (1~20)
                tooltip={'always_visible': True, 'placement': 'bottom'},  # 툴팁 항상 표시
                included=False  # 슬라이더 선택 값만 표시
            ),
            style={'width': '80%', 'margin': '0 auto'}
        ),

        # 결과 그래프
        dcc.Graph(id='line-graph'),

    ]
)




@app.callback(
    Output('line-graph', 'figure'),
    [Input('vintage-target', 'value'),
     Input('test-period', 'value')]  # 테스트 기간 입력을 추가
)
def update_graph(vintage_target, test_period):
    # 시작 날짜 및 종료 날짜 정의
    start = (datetime.today() - relativedelta(years=test_period)).strftime('%Y-%m-%d')
    end = (datetime.today() - timedelta(days=0)).strftime('%Y-%m-%d')


    # ETF 가격 데이터 가져오기
    ETF_price = Func(code, start, end, 30)
    ETF_price = ETF_price.ffill()

    

    # KIS종합채권 지수 데이터프레임으로 채우기===================
    path2 = rf'C:\Covenant\TDF\data\TDF_디폴트옵션_백테스트.xlsx'

    df_KIS = pd.read_excel(path2, sheet_name='KIS종합채권지수')
    df_KIS = df_KIS.iloc[:, 3:5]
    df_KIS = df_KIS.set_index(df_KIS.columns[0])
    df_KIS = df_KIS.sort_index(ascending=True)



    df_지수 = pd.read_excel(path2, sheet_name='지수')
    df_지수 = df_지수.iloc[7:, 0:7]
    df_지수.columns =  df_지수.iloc[0]
    df_지수 = df_지수.drop(df_지수.index[0])
    
    df_지수.set_index('Dates', inplace=True)
    df_지수.index = pd.to_datetime(df_지수.index, errors='coerce')  # 날짜 형식으로 변환

    # 인덱스를 원하는 날짜 형식으로 변환
    df_지수.index = df_지수.index.strftime('%Y-%m-%d')
    
    print("df_지수=================\n", df_지수)

    
    df_ETF_R = ETF_price.pct_change()
    df_KIS_R = df_KIS.pct_change()
    df_지수_R = df_지수.pct_change()

    print("df_ETF_R=================\n", df_ETF_R)
    print("df_KIS_R=================\n", df_KIS_R)
    print("df_지수_R================\n", df_지수_R)




    # 수익률 데이터 병합
    df_R = pd.merge(df_ETF_R, df_KIS_R, left_index=True, right_index=True, how='left')
    df_R = pd.merge(df_R, df_지수_R, left_index=True, right_index=True, how='left')
    df_R = df_R[df_R.index >= start]


    # # NaN 채우기 후 infer_objects()를 사용하여 데이터 타입을 추론
    # df_R['273130.KS'] = df_R['273130.KS'].fillna(df_R['KIS종합채권지수'])
    # df_R['365780.KS'] = df_R['365780.KS'].fillna(df_R['KIS종합채권지수'])
    # df_R['VUG'] = df_R['VUG'].fillna(df_R['US Growth'])
    # df_R['VTV'] = df_R['VTV'].fillna(df_R['US Value'])
    # df_R['VEA'] = df_R['VEA'].fillna(df_R['DM ex US'])
    # df_R['VWO'] = df_R['VWO'].fillna(df_R['EM'])



    # 0이나 Nan 값이 있으면 대체
    df_R['ACWI'] = df_R['ACWI'].mask(df_R['ACWI'].isin([0]) | df_R['ACWI'].isna(), df_R['ACWI_지수'])
    df_R['BND'] = df_R['BND'].mask(df_R['BND'].isin([0]) | df_R['BND'].isna(), df_R['BGABI'])
    df_R['273130.KS'] = df_R['273130.KS'].mask(df_R['273130.KS'].isin([0]) | df_R['273130.KS'].isna(), df_R['KIS종합채권지수'])
    df_R['365780.KS'] = df_R['365780.KS'].mask(df_R['365780.KS'].isin([0]) | df_R['365780.KS'].isna(), df_R['KIS종합채권지수'])
    df_R['VUG'] = df_R['VUG'].mask(df_R['VUG'].isin([0]) | df_R['VUG'].isna(), df_R['US Growth'])
    df_R['VTV'] = df_R['VTV'].mask(df_R['VTV'].isin([0]) | df_R['VTV'].isna(), df_R['US Value'])
    df_R['VEA'] = df_R['VEA'].mask(df_R['VEA'].isin([0]) | df_R['VEA'].isna(), df_R['DM ex US'])
    df_R['VWO'] = df_R['VWO'].mask(df_R['VWO'].isin([0]) | df_R['VWO'].isna(), df_R['EM'])

    print("df_R=====================\n", df_R)
    


    
    
    # print("df_ETF_R=====================\n", df_ETF_R)
                                
    
    BM_R_US = df_R['ACWI']*0.6 + df_R['BND']*0.4
    BM_R_KIS = df_R['ACWI']*0.6 + df_R['273130.KS']*0.4
    cum_BM_US = (1 + BM_R_US).cumprod() - 1
    cum_BM_KIS = (1 + BM_R_KIS).cumprod() - 1

    print("cum_BM_US================", cum_BM_US)
    print("cum_BM_KIS================", cum_BM_KIS)






    # # df_weight 생성
    df_weight = generate_df_weight(vintage_target, test_period, df_MP)

    # NaN 값을 채운 후, object dtype을 명시적으로 추론
    df_weight = df_weight[list(주식.keys()) + list(채권.keys())].fillna(0)

    # 이후 df_R와 df_weight_filtered 곱하기
    mapping = {key: value for key, value in zip(code_dict.keys(), code_dict.values())}

    # df_weight의 열 이름을 code_dict의 values로 변경
    df_weight.columns = df_weight.columns.map(mapping)

    df_ctr = (df_weight * df_R[df_weight.columns]).fillna(0)
    # 모든 값이 0이 아닌 행만을 지정
    df_ctr = df_ctr[(df_ctr != 0).any(axis=1)]

    # df_cleaned는 모든 값이 0인 행이 제거된 DataFrame

    
    print("df_ctr================\n", df_ctr)


    # # print("df_R.columns===============\n", df_R.columns)
    # # print("df_weight.columns===============\n", df_weight.columns)
    # # print("df_ctr.columns===============\n", df_ctr.columns)


    # 결과 출력
    port_R = df_ctr.sum(axis=1)
    print("port_R================\n", port_R)

    cum_port = (1 + port_R).cumprod() - 1


    # Stat 요약 모듈
    def stat(df_R, start, interval='w'): 
        # start를 날짜 형식으로 변환
        stat_start = pd.to_datetime(start)
        df_R.index = pd.to_datetime(df_R.index)

        # start 이후의 날짜만 필터링
        df_filtered = df_R[df_R.index >= stat_start]

        # interval에 따른 w 값을 설정 ('w'면 5, 'm'이면 20)
        if interval == 'w':
            interval = 5
        elif interval == 'm':
            interval = 20

        # 매 w번째 행을 선택하고 오름차순 정렬한 데이터로 변동성 계산
        vol = (df_filtered.sort_index(ascending=False).iloc[::interval].sort_index(ascending=True)).std() * np.sqrt(52)

        # df_filtered의 평균 계산
        mean = df_filtered.rolling(window=250).mean()

        return vol, mean

    vol, mean = stat(port_R, '2010-01-01', 'w')
    print("vol==========\n",vol, "mean============\n", mean)


    # save_excel(ETF_price, f'ETF_price', index_option=None)
    # save_excel(df_weight, 'df_weight', index_option=None)
    # save_excel(df_R, 'df_R', index_option=None)
    # save_excel(df_ctr, 'df_ctr', index_option=None)
    # save_excel(port_R, 'port_R', index_option=None)
    # save_excel(cum_port, 'cum_port', index_option=None)
    # save_excel(BM_R_US, 'BM_R_US', index_option=None)




    # 그래프 생성
    figure = {
        'data': [
            go.Scatter(x=cum_port.index, y=cum_port, mode='lines', name=f'ETF 포커스 {vintage_target}'),
            go.Scatter(x=cum_BM_US.index, y=cum_BM_US, mode='lines', name=f'BM : ACWI*60% + BND*40%'),
            go.Scatter(x=cum_BM_KIS.index, y=cum_BM_KIS, mode='lines', name=f'BM : ACWI*60% + KIS종합*40%'),
        ],
        'layout': go.Layout(
            title=f'TDF 백테스트 - Vintage {vintage_target} (테스트 기간: {test_period}년)',            
            xaxis={'title': 'Date', 'tickformat': '%Y-%m-%d'},
            yaxis={'title': '누적 수익률', 'tickformat': ',.0%'},
        )
    }

    #cum_port는 Series 객체로, DataFrame처럼 columns 속성이 없습니다. 이를 해결하려면 cum_port를 DataFrame으로 변환한 후 dash_table.DataTable에 사용
    cum_port = cum_port.reset_index()
    return figure
        
      

# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')
