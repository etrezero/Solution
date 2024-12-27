import pandas as pd
import numpy as np
import openpyxl
import os
from openpyxl import Workbook
import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests

import pandas as pd
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yaml




def load_config(config_path='config.yaml'):
    """
    Config 파일에서 API 키를 로드합니다.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            api_key = config.get('ECOS_key', None)
            return api_key
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error reading {config_path}: {e}")
        return None


config_path = r'C:\Covenant\config.yaml'
ECOS_API_KEY = load_config(config_path)




# ECOS API 호출 함수
def ECOS_StatisticSearch(stat_code, Frequency, item_code, start, end):
    """
    한국은행 ECOS API를 사용하여 데이터 가져오기
    """
    # Frequency에 따른 날짜 형식 변환
    if Frequency == 'A':  # 연간 데이터
        start_date = start.strftime('%Y')
        end_date = end.strftime('%Y')
    elif Frequency == 'Q':  # 분기 데이터
        start_date = start.strftime('%Y') + 'Q' + str((start.month - 1) // 3 + 1)
        end_date = end.strftime('%Y') + 'Q' + str((end.month - 1) // 3 + 1)
    elif Frequency == 'M':  # 월간 데이터
        start_date = start.strftime('%Y%m')
        end_date = end.strftime('%Y%m')
    elif Frequency == 'D':  # 일간 데이터
        start_date = start.strftime('%Y%m%d')
        end_date = end.strftime('%Y%m%d')
    else:
        raise ValueError("Invalid Frequency. Use 'A', 'Q', 'M', or 'D'.")
    # API 호출 URL 구성
    url = (
        f"http://ecos.bok.or.kr/api/StatisticSearch/"
        f"{ECOS_API_KEY}/json/kr/1/100/"
        f"{stat_code}/{Frequency}/{start_date}/{end_date}/{item_code}"
    )
    
    try:
        # SSL 인증서를 무시하기 위해 verify=False 추가
        response = requests.get(url, verify=False)
        response.raise_for_status()
        data = response.json()
        
        if 'StatisticSearch' in data:
            rows = data['StatisticSearch']['row']
            df = pd.DataFrame(rows)

            # 'TIME' 열 처리
            if Frequency == 'M':
                df['TIME'] = pd.to_datetime(df['TIME'], format='%Y%m')  # %Y%m 형식으로 변환
                df['TIME'] = df['TIME'].dt.to_period('M').dt.strftime('%Y%m')  # 매월 1일로 설정하고 형식을 유지
            else:
                df['TIME'] = pd.to_datetime(df['TIME'])



            df.set_index('TIME', inplace=True)
            df = df[['DATA_VALUE']].astype(float)
            
            return df
        else:
            print("No data found in response.")
            return None
    except Exception as e:
        print(f"Error fetching ECOS data: {e}")
        return None



# 여러 지표 데이터를 병합하는 함수 수정
def get_ECOS(item_dict, Frequency, start, end):
    merged_df = pd.DataFrame()

    for stat_code, items in item_dict.items():
        for item_code, column_name in items:
            df = ECOS_StatisticSearch(stat_code, Frequency, item_code, start, end)
            if df is not None:
                df.rename(columns={'DATA_VALUE': column_name}, inplace=True)
                if merged_df.empty:
                    merged_df = df
                else:
                    merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, how='outer')
    
    return merged_df




path = rf'C:\Covenant\data\ECON_ECOS.xlsx'

#엑셀 저장=======================================================
def save_excel(df, sheetname, index_option=None):
    
    # 파일 경로
    path = rf'C:\Covenant\data\ECON_ECOS.xlsx'

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









# 테스트 데이터 가져오기
if __name__ == "__main__":


    경제성장률 = {
        '902Y015': [
                    ('KOR', '한국'),
                    ('USA', '미국'),
                    ('CAN', '캐나다'),
                    ('AUS', '호주'),
                    ('CHN', '중국'),
                    ('FRA', '프랑스'),
                    ('DEU', '독일'),
                    ('JPN', '일본'),
                    ('GBR', '영국'),
        ],
    }



    소비자물가지수 = {
        '902Y008': [
                    ('KR', '한국'),
                    ('US', '미국'),
                    ('CA', '캐나다'),
                    ('AU', '호주'),
                    ('CN', '중국'),
                    ('FR', '프랑스'),
                    ('DE', '독일'),
                    ('JP', '일본'),
                    ('GB', '영국'),
        ],
    }



    생산자물가지수 = {
        '902Y007': [
                    ('KR', '한국'),
                    ('US', '미국'),
                    ('CA', '캐나다'),
                    ('AU', '호주'),
                    ('CN', '중국'),
                    ('FR', '프랑스'),
                    ('DE', '독일'),
                    ('JP', '일본'),
                    ('GB', '영국'),
        ],
    }



    
    수출 = {
        '902Y012': [
                    ('KR', '한국'),
                    ('US', '미국'),
                    ('CA', '캐나다'),
                    ('AU', '호주'),
                    ('CN', '중국'),
                    ('FR', '프랑스'),
                    ('DE', '독일'),
                    ('JP', '일본'),
                    ('GB', '영국'),
        ],
    }



    환율 = {
        '731Y003': [
                    ('0000003', '한국'),
        ],

        '731Y002': [
                    ('0000013', '캐나다'),
                    ('0000008', '$/호주'),
                    ('0000027', '중국'),
                    ('0000005', '프랑스'),
                    ('0000004', '독일'),
                    ('0000002', '일본'),
                    ('0000003', '$/유로'),
                    ('0000012', '$/영국'),
                    ('0000035', '베트남'),
        ],

    }




    
    외환보유액 = {
        '902Y014': [
                    ('KR', '한국'),
                    ('US', '미국'),
                    ('CA', '캐나다'),
                    ('AU', '호주'),
                    ('CN', '중국'),
                    ('FR', '프랑스'),
                    ('DE', '독일'),
                    ('JP', '일본'),
                    ('GB', '영국'),
        ],
    }


    산업생산지수 = {
        '902Y020': [
                    ('KOR', '한국'),
                    ('USA', '미국'),
                    ('CAN', '캐나다'),
                    ('AUS', '호주'),
                    ('CHN', '중국'),
                    ('FRA', '프랑스'),
                    ('DEU', '독일'),
                    ('JPN', '일본'),
                    ('GBR', '영국'),
        ],
    }



    실업률 = {
        '902Y021': [
                    ('KOR', '한국'),
                    ('USA', '미국'),
                    ('CAN', '캐나다'),
                    ('AUS', '호주'),
                    ('CHN', '중국'),
                    ('FRA', '프랑스'),
                    ('DEU', '독일'),
                    ('JPN', '일본'),
                    ('GBR', '영국'),
        ],
    }



    대외채무 = {
        '311Y004': [
                    ('A000000', '대외채무(USD mil.)'),
        ],
    }

    
    대외채권 = {
        '311Y005': [
                    ('B000000', '대외채권(USD mil.)'),
        ],
    }


    
    순대외채권 = {
        '311Y006': [
                    ('C000000', '순대외채권(USD mil.)'),
        ],
    }



    평균임금 = {
        '901Y086': [
                    ('I68A', '전직종'),
        ],
    }


    주택시가총액 = {
        '291Y424': [
                    ('101', '주택시가총액(십억원)'),
        ],
    }



    금융기관유동성 = {
        '101Y018': [
                    ('BBLS00', 'M1(십억원)'),
                    ('A110000', '중앙은행(십억원)'),
        ],

        '101Y003': [
                    ('BBHS00', 'M2(십억원)'),
                    ('BBHS01', '현금(십억원)'),
                    ('BBHS02', '요구불예금(십억원)'),
                    ('BBHS03', '입출식저축성예금(십억원)'),
                    ('BBHS04', 'MMF(십억원)'),
                    ('BBHS05', '2년미만 정기예금(십억원)'),
                    ('BBHS06', '수익증권(십억원)'),
        ],
    }





    # Term 정의 (A: 연간, Q: 분기, M: 월간, D: 일간)
    

    # 시작과 종료 날짜 설정
    start = datetime.today() - relativedelta(years=10)
    end = datetime.today() - relativedelta(months=1)

    
    df_경제성장률 = get_ECOS(경제성장률, 'Q', start, end).ffill()
    df_소비자물가지수 = get_ECOS(소비자물가지수, 'Q', start, end).ffill()
    df_생산자물가지수 = get_ECOS(생산자물가지수, 'Q', start, end).ffill()
    df_수출 = get_ECOS(수출, 'Q', start, end).ffill()
    df_외환보유액 = get_ECOS(외환보유액, 'Q', start, end).ffill()
    df_산업생산지수 = get_ECOS(산업생산지수, 'Q', start, end).ffill()
    df_실업률 = get_ECOS(실업률, 'Q', start, end).ffill()
    df_대외채무 = get_ECOS(대외채무, 'Q', start, end).ffill()
    df_대외채권 = get_ECOS(대외채권, 'Q', start, end).ffill()
    df_순대외채권 = get_ECOS(순대외채권, 'Q', start, end).ffill()
    
    

    
    df_평균임금 = get_ECOS(평균임금, 'A', start, end).ffill()
    df_주택시가총액 = get_ECOS(주택시가총액, 'A', start, end).ffill()
        


    df_금융기관유동성 = get_ECOS(금융기관유동성, 'M', start, end).ffill()


    df_환율 = get_ECOS(환율, 'D', start, end).ffill()
        

    # 리스트와 이름을 함께 저장 (선택 사항)
    df_dict = [
        ("경제성장률", df_경제성장률),

        ("소비자물가지수", df_소비자물가지수),
        ("생산자물가지수", df_생산자물가지수),

        ("수출", df_수출),
        ("외환보유액", df_외환보유액),

        ("산업생산지수", df_산업생산지수),
        ("실업률", df_실업률),

        ("대외채무", df_대외채무),
        ("대외채권", df_대외채권),
        ("순대외채권", df_순대외채권),


        ("평균임금", df_평균임금),
        ("주택시가총액", df_주택시가총액),

        ("금융기관유동성", df_금융기관유동성),

        ("환율", df_환율),
    ]

    # 데이터프레임 리스트 확인
    for name, df in df_dict:
        print(f"\n{name}\n", df)



app = Dash(__name__)



# 레이아웃 정의
app.layout = html.Div([
    html.H1("ECON_ECOS", style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': name, 'value': name} for name, _ in df_dict],
        value='경제성장률',
        placeholder="데이터 선택",
        style={'width': '30%', 'margin': 'auto'}
    ),
    dcc.Graph(id='line-graph', style={'width': '60%', 'margin': 'auto'}),
    
    html.Button("EXCEL", id='save-button', style={'margin': '20px auto', 'display': 'block'}),
    html.Div(id='save-status', style={'textAlign': 'center', 'marginTop': '10px'}),
])




# 콜백 함수 정의
@app.callback(
    Output('line-graph', 'figure'),
    [Input('dropdown', 'value')]
)
def update_graph(selected_name):
    # 선택된 데이터프레임 가져오기
    selected_df = next((df for name, df in df_dict if name == selected_name), None)
    
    if selected_df is not None:
        fig = go.Figure()
        for column in selected_df.columns:
            fig.add_trace(go.Scatter(x=selected_df.index, y=selected_df[column], mode='lines', name=column))
        fig.update_layout(
            title=f"{selected_name} 데이터",
            xaxis_title="시간",
            yaxis_title="값",
            yaxis=dict(tickformat=",.0f"),  # y축 값을 소수점 한 자리로 표시
            template="plotly_white"
        )


        return fig
    else:
        # 선택된 데이터가 없을 경우 빈 그래프 반환
        return go.Figure()




# 콜백: 엑셀로 저장
@app.callback(
    Output('save-status', 'children'),
    [Input('save-button', 'n_clicks')],
    [State('dropdown', 'value')]
)
def save_to_excel_btn(n_clicks, selected_name):
    if n_clicks > 0 and selected_name:
        # 선택된 데이터프레임 가져오기
        selected_df = next((df for name, df in df_dict if name == selected_name), None)
        if selected_df is not None:
            # 엑셀 파일로 저장
            save_excel(selected_df, selected_name)
            return f"'{selected_name}' 데이터를 {path}에 엑셀 파일로 저장했습니다."
    return "[저장 버튼]을 클릭해주세요."


# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=False,host='0.0.0.0') 



