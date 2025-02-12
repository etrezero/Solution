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
import random




BM_dict = {
    
    "VUG" : 18.2,
    "VTV": 11.1,
    "VEA": 2.5,
    "VWO": 2.5,
    "GLD": 1.8,
    "356540": 64,   #ACE KIS종합채권(AA-이상)액티브

    "ACWI": 0,
    "BND": 0,

    # "273130": 0,   #KODEX 종합채권(AA-이상)액티브
}




# code_dict에 포함된 모든 티커 리스트
code = list(BM_dict.keys())

print("code==================", code)


# 캐싱 경로 및 만료 시간 설정
cache_price = r'C:\Covenant\TDF\data\TIF_포커스_백테스트_월지급식_price.pkl'
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












# #엑셀 저장=======================================================
# def save_excel(df, sheetname, index_option=None):
    
#     # 파일 경로
#     path = rf'C:\Covenant\TDF\data\TDF_디폴트옵션_백테스트.xlsx'

#     # 파일이 없는 경우 새 Workbook 생성
#     if not os.path.exists(path):
#         wb = Workbook()
#         wb.save(path)
#         print(f"새 파일 '{path}' 생성됨.")
    
#     # 인덱스를 날짜로 변환 시도
#     try:
#         # index_option이 None일 경우 인덱스를 포함하고 날짜 형식으로 저장
#         if index_option is None or index_option:  # 인덱스를 포함하는 경우
#             df.index = pd.to_datetime(df.index, errors='raise')  # 변환 실패 시 오류 발생
#             df.index = df.index.strftime('%Y-%m-%d')  # 벡터화된 방식으로 날짜 포맷 변경
#             index = True  # 인덱스를 포함해서 저장
#         else:
#             index = False  # 인덱스를 제외하고 저장
#     except Exception:
#         print("Index를 날짜 형식으로 변환할 수 없습니다. 기본 인덱스를 사용합니다.")
#         index = index_option if index_option is not None else True  # 변환 실패 시에도 인덱스를 포함하도록 설정

#     # DataFrame을 엑셀 시트로 저장
#     with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
#         df.to_excel(writer, sheet_name=sheetname, index=index)  # index 여부 설정
#         print(f"'{sheetname}' 저장 완료.")






# Dash 앱 생성
app = dash.Dash(__name__)
app.title = 'TIF 월지급식 백테스트'








# 레이아웃 정의
app.layout = html.Div(
    style={'width': '50%', 'margin': 'auto', 'height': '800px'},  # 높이 설정
    children=[
        html.H3("TIF 백테스트", style={'textAlign': 'center'}),

        html.Label('테스트 기간 (연도):', style={'textAlign': 'center'}),

        # 슬라이더를 감싸는 Div에 스타일 적용
        html.Div(
            dcc.Slider(
                id='test-period',
                min=1,
                max=20,
                step=5,
                value=15,  # 기본값 설정
                marks={i: f'{i}' for i in range(0, 21, 5)},  # 5년 단위로 마크 설정
                tooltip={'always_visible': True, 'placement': 'bottom'},  # 툴팁 항상 표시
                included=False  # 슬라이더 선택 값만 표시
            ),
            style={'width': '80%', 'margin': '0 auto'}
        ),

        html.H3("", style={'textAlign': 'center'}),

        # 분배율 슬라이더 추가 (0% ~ 10%)
        html.Label('분배율 (연%):', style={'textAlign': 'center'}),
        html.Div(
            dcc.Slider(
                id='distribution-rate',
                min=0,
                max=10,
                step=0.5,
                value=3,  # 기본값 설정
                marks={i: f'{i}%' for i in range(0, 11, 1)},  # 0% ~ 10% 범위로 마크 설정
                tooltip={'always_visible': True, 'placement': 'bottom'}
            ),
            style={'width': '80%', 'margin': '0 auto'}
        ),

        # 결과 그래프
        dcc.Graph(id='line-graph'),

        dcc.Graph(id='monte-carlo-graph'),


    ]
)




@app.callback(
    [Output('line-graph', 'figure'),
     Output('monte-carlo-graph', 'figure')],
    [Input('test-period', 'value'),
     Input('distribution-rate', 'value')]
)
def update_graph(test_period, distribution_rate):
    start = (datetime.today() - relativedelta(years=test_period)).strftime('%Y-%m-%d')
    end = (datetime.today() - timedelta(days=0)).strftime('%Y-%m-%d')

    # ETF 가격 데이터 가져오기
    ETF_price = Func(code, start, end, 30)
    ETF_price = ETF_price.ffill()

    ETF_price.index = pd.to_datetime(ETF_price.index)

    # KIS종합채권 지수 데이터프레임으로 채우기===================
    df_R = ETF_price.pct_change()
    df_R = df_R[df_R.index >= start]
    
    BM_R_US = df_R['ACWI']*0.4 + df_R['BND']*0.6
    BM_R_KIS = df_R['ACWI']*0.4 + df_R['356540']*0.6




    # 분배율 적용: 연간 분배율을 일별로 나눠서 차감
    monthly_distribution = distribution_rate / 100 / 12  # 연간 분배율을 월별로 나누기

    # 월말 날짜를 찾아서 해당 날짜에만 분배금 차감
    month_ends = ETF_price.index.to_period('M').to_timestamp('M')  # 월말 날짜를 'M'으로 추출

    # 월말 날짜에 분배율 차감
    for month_end in month_ends.unique():  # 고유한 월말 날짜만 반복
        if month_end in BM_R_US.index:  # 월말 날짜가 cum_simulations_df의 인덱스에 존재하면
            BM_R_US.loc[month_end] -= monthly_distribution  # 월말에 분배금 차감

        if month_end in BM_R_KIS.index:  # 월말 날짜가 cum_BM_US 인덱스에 존재하면
            BM_R_KIS.loc[month_end] -= monthly_distribution



    cum_BM_US = (1 + BM_R_US).cumprod() - 1
    cum_BM_KIS = (1 + BM_R_KIS).cumprod() - 1

    # df_weight 생성 (df_R의 열 이름과 인덱스를 그대로 사용)
    df_weight = pd.DataFrame(index=df_R.index, columns=df_R.columns)

    # BM_dict의 키에 포함된 열만 필터링
    columns_to_keep = [col for col in df_weight.columns if col in BM_dict.keys()]

    # 해당 열들만 남긴 새로운 df_weight
    df_weight = df_weight[columns_to_keep]

    # 이제 df_weight의 열 이름을 BM_dict의 키로 설정
    df_weight.columns = list(BM_dict.keys())

    # BM_dict에 따라 각 ETF 티커에 투자 비중을 할당
    for ticker in df_weight.columns:
        df_weight[ticker] = BM_dict.get(ticker, 0)  # 만약 BM_dict에 없으면 0으로 설정

    # df_weight와 df_R 곱하기
    df_ctr = (df_weight/100 * df_R[df_weight.columns]).fillna(0)

    # 모든 값이 0이 아닌 행만을 필터링
    df_ctr = df_ctr[(df_ctr != 0).any(axis=1)]

    # 결과 출력
    port_R = df_ctr.sum(axis=1)
    port_R = port_R.dropna().replace([np.inf, -np.inf], 0)
    port_R.index = pd.to_datetime(port_R.index)

    # 월말 날짜에 분배율 차감
    for month_end in month_ends.unique():  # 고유한 월말 날짜만 반복
        if month_end in port_R.index:  # 월말 날짜가 port_R의 인덱스에 존재하면
            port_R.loc[month_end] -= monthly_distribution  # 해당 월말에 분배금 차감


    cum_port = (1 + port_R).cumprod() - 1


    # 몬테카를로 시뮬레이션
    num_simulations = 50  # 시뮬레이션 횟수
    num_days = len(port_R)  # 시뮬레이션 기간 (일 수)
    initial_investment = 1  # 초기 투자 금액

    # 샘플링할 수익률
    daily_returns = port_R.values



    # 메트로폴리스-헤이스팅스 샘플링 함수
    def metropolis_hastings(daily_returns, num_samples, proposal_width):
        # Initial sample
        current_sample = np.random.choice(daily_returns)
        
        samples = [current_sample]
        
        for _ in range(num_samples):
            # Propose a new sample from the proposal distribution
            proposed_sample = np.random.normal(current_sample, proposal_width)
            
            # Calculate the probabilities of the current and proposed sample
            current_prob = np.sum(daily_returns == current_sample) / len(daily_returns)
            proposed_prob = np.sum(daily_returns == proposed_sample) / len(daily_returns)
            
            # Acceptance ratio
            acceptance_ratio = min(1, proposed_prob / current_prob)
            
            # Accept or reject the proposed sample
            if np.random.rand() < acceptance_ratio:
                current_sample = proposed_sample
            
            samples.append(current_sample)
        
        return np.array(samples)



    # 시뮬레이션 함수 (MCMC를 적용한 버전)
    def monte_carlo_simulation_mcmc(initial_investment, daily_returns, num_simulations, num_days, proposal_width=0.01):
        simulations = []
        
        for _ in range(num_simulations):
            simulation = [initial_investment]
            
            for i in range(num_days):
                # MCMC 방법을 사용하여 daily_return 샘플링
                daily_return_samples = metropolis_hastings(daily_returns, 1, proposal_width)
                daily_return = daily_return_samples[0]  # MCMC에서 추출한 첫 번째 샘플
                
                simulation.append(simulation[-1] * (1 + daily_return))
            
            simulations.append(simulation)
        
        return np.array(simulations)


    # 시뮬레이션 실행
    simulations = monte_carlo_simulation_mcmc(initial_investment, daily_returns, num_simulations, num_days)

    # 시뮬레이션 결과 누적 수익률
    cum_simulations = (simulations / initial_investment - 1)


    # 몬테카를로 시뮬레이션 결과를 데이터프레임으로 변환
    cum_simulations_df = pd.DataFrame(cum_simulations[:, :-1].T, index=cum_port.index)  # Exclude the last row of cum_simulations
    cum_simulations_df.columns = [f'Simulation {i+1}' for i in range(num_simulations)]  # 시뮬레이션 번호로 열 이름 지정
    print("cum_simulations_df==================", cum_simulations_df )
    

    # 월말 날짜에 분배율 차감
    for month_end in month_ends.unique():  # 고유한 월말 날짜만 반복
        if month_end in cum_simulations_df.index:  # 월말 날짜가 cum_simulations_df의 인덱스에 존재하면
            cum_simulations_df.loc[month_end] *= (1 - monthly_distribution)  # 월말에 분배금 차감
            # cum_simulations.loc[month_end] -= monthly_distribution  # 해당 월말에 분배금 차감

        if month_end in cum_BM_US.index:  # 월말 날짜가 cum_BM_US 인덱스에 존재하면
            cum_BM_US.loc[month_end] *= (1 - monthly_distribution)  # 월말에 분배금 차감


    avg_final_return = np.mean(cum_simulations[:, -1])  # 최종 수익률의 평균




    # 포트폴리오 그래프
    portfolio_figure = {
        'data': [
            go.Scatter(x=cum_port.index, y=cum_port, mode='lines', name=f'TIF 월지급식 : 연{distribution_rate}% 월지급'),
            go.Scatter(x=cum_BM_US.index, y=cum_BM_US, mode='lines', name=f'BM(ACWI*40% + BND*60%) : 연{distribution_rate}% 월지급'),
            # go.Scatter(x=cum_BM_KIS.index, y=cum_BM_KIS, mode='lines', name=f'BM : ACWI*40% + KIS종합*60%'),
        ],
        'layout': go.Layout(
            title=f'TIF 월지급식 Back-Test (테스트 기간: {test_period}년)',            
            xaxis={'title': 'Date', 'tickformat': '%Y-%m-%d'},
            yaxis={'title': '누적 수익률', 'tickformat': ',.0%'}
        )
    }


    # 시뮬레이션 결과 그래프 그리기
    cum_simulations_figure = {
        'data': [
            go.Scatter(
                x=cum_simulations_df.index, 
                y=cum_simulations_df.iloc[:, i],  # i번째 열을 선택
                mode='lines', 
                line=dict(width=0.5, color='lightgray'),
                showlegend=False
            ) for i in range(num_simulations)
        ] + [
            go.Scatter(
                x=cum_simulations_df.index, 
                y=np.mean(cum_simulations_df.values, axis=1),  # 평균 누적 수익률을 선으로 표시
                mode='lines', 
                line=dict(color='blue', width=2),
                name='평균 누적 수익률'
            )
        ],
        'layout': go.Layout(
            title=f'MCMC 시뮬레이션 결과',
            xaxis={'title': 'Date', 'tickformat': '%Y-%m-%d'},
            yaxis={'title': '누적 수익률', 'tickformat': ',.0%'}
        )
    }


    return portfolio_figure, cum_simulations_figure
        





# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
