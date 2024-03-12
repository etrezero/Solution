import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import requests
import yaml
import os

app = dash.Dash(__name__)

# TE API 키 가져오기
yaml_path = 'C:\\Users\\서재영\\Documents\\Python Scripts\\koreainvestment-autotrade-main\\config.yaml'
with open(yaml_path, encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)
TE_key = _cfg['TE_key']

# URL과 해당 지표의 특징을 딕셔너리로 매칭
indicator_mapping = {
    'GDP Growth Rate': 'https://api.tradingeconomics.com/historical/country/United%20States/indicator/GDP%20Growth%20Rate?',
    'Unemployment Rate': 'https://api.tradingeconomics.com/historical/country/United%20States/indicator/Unemployment%20Rate?',
    'Inflation Rate': 'https://api.tradingeconomics.com/historical/country/United%20States/indicator/Inflation%20Rate?',
    'Interest Rate': 'https://api.tradingeconomics.com/historical/country/United%20States/indicator/Interest%20Rate?',
    'Balance of Trade': 'https://api.tradingeconomics.com/historical/country/United%20States/indicator/Balance%20of%20Trade?',
    'Government Debt to GDP': 'https://api.tradingeconomics.com/historical/country/United%20States/indicator/Government%20Debt%20to%20GDP?',
    'Manufacturing PMI': 'https://api.tradingeconomics.com/historical/country/United%20States/indicator/Manufacturing%20PMI?',
    'Services PMI': 'https://api.tradingeconomics.com/historical/country/United%20States/indicator/Services%20PMI?',
    'Productivity': 'https://api.tradingeconomics.com/historical/country/United%20States/indicator/Productivity?',
    'Wage Growth': 'https://api.tradingeconomics.com/historical/country/United%20States/indicator/Wage%20Growth?',
    'Population': 'https://api.tradingeconomics.com/historical/country/United%20States/indicator/Population?',
    'Core Inflation Rate': 'https://api.tradingeconomics.com/historical/country/United%20States/indicator/Core%20Inflation%20Rate?',
    'Producer Prices Change': 'https://api.tradingeconomics.com/historical/country/United%20States/indicator/Producer%20Prices%20Change?'
}

# TE API에 요청 보내고 데이터프레임 생성하는 함수
def fetch_data_and_create_df(indicator_name):
    endpoint = indicator_mapping[indicator_name]
    params = {'c': TE_key}
    response = requests.get(endpoint, params=params)
    data = response.json()
    df = pd.DataFrame(data)
    df['DateTime'] = pd.to_datetime(df['DateTime'])  # datetime 열을 날짜 형식으로 변환
    df['DateTime'] = df['DateTime'].dt.strftime('%Y-%m-%d')  # 원하는 형식으로 포맷팅
    return df

# 그래프의 마지막 값(value)와 해당 datetime 가져오는 함수
def get_last_value_and_datetime(df):
    last_value = df.iloc[-1]['Value']
    last_datetime = df.iloc[-1]['DateTime']
    return last_value, last_datetime

indicator_dropdown_style = {
    'width': '40%',  # 가로 크기를 30%로 지정
    'position': 'absolute',  # 위치를 절대값으로 설정
    'top': '60px',  # 위쪽 여백을 조절하여 우측 상단으로 이동
    'right': '150px', # 오른쪽 여백을 조절하여 우측 상단으로 이동
    'z-index' : '9999',
    'background-color' : '#4BACC6',
    'font-weight' : 'bold',
}

# 인디케이터 드롭다운 설정
indicator_dropdown = dcc.Dropdown(
    id='indicator-dropdown',
    options=[{'label': indicator_name, 'value': indicator_name} for indicator_name in indicator_mapping.keys()],
    value='Producer Prices Change',  # 기본 선택값
    style=indicator_dropdown_style
)

# 테이블 우측 상단에 다운로드 버튼을 추가하는 함수
def create_download_link(df, filename):
    link = html.A(
        'Download Data',
        id='download-link',
        download=f'{filename}.xlsx',
        href='',
        target='_blank',
        style={
            'width': '15%',
            'height' : '4%', 
            'position': 'absolute',  # 위치를 절대값으로 설정
            'top': '550px',  # 위쪽 여백을 조절하여 우측 상단으로 이동
            'right': '300px', # 오른쪽 여백을 조절하여 우측 상단으로 이동
            'z-index' : '9999',
            'background-color' : '#4BACC6',
            'font-weight' : 'bold',
            'text-align': 'center',
        }
    )
    return link

# Dash 애플리케이션 레이아웃 설정
app.layout = html.Div([
    html.H1('Macro Indicator', style={'text-align': 'center'}),

    html.Div([
        html.H2(''),
        indicator_dropdown
    ]),
    
    html.Div(id='graph-table-container')
])

# 인디케이터 선택에 따라 그래프와 테이블을 업데이트하는 콜백 함수
@app.callback(
    Output('graph-table-container', 'children'),
    [Input('indicator-dropdown', 'value')]
)
def update_graph_table(indicator_name):
    # 데이터 가져오기
    df = fetch_data_and_create_df(indicator_name)
    # 그래프의 마지막 값(value)와 해당 datetime 가져오기
    last_value, last_datetime = get_last_value_and_datetime(df)
    
    # 그래프 생성
    graph = dcc.Graph(
        id='producer-prices-graph',
        figure={
            'data': [
                {'x': df['DateTime'], 'y': df['Value'], 'type': 'line', 'name': indicator_name},
                {'x': [last_datetime], 'y': [last_value], 'mode': 'markers+text', 'text': f'{last_datetime}:[{last_value}]', 'textposition': 'top center', 'marker': {'size': 10}, 'name': ''}
            ],
            'layout': {
                'title': f'{indicator_name}',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Value'}
            }
        }, 
        style={
            'width': '90%', 
            'margin': '0 auto', 
            # 'border': '1px solid black',
            'text-align' : 'center',
        },
    )
    
    # 테이블 생성
    table = html.Table(
        style={'width': '80%', 
               'margin': '0 auto', 
               'border': '1px solid black',
               'text-align' : 'center',
               },
        children=[
            html.Thead(html.Tr([html.Th(col) for col in df.columns])),
            html.Tbody([
                html.Tr([
                    html.Td(df.tail(10).iloc[i][col]) for col in df.columns
                ]) for i in range(len(df.tail(10)))
            ])
        ]
    )
    
    # 다운로드 링크 생성
    download_link = create_download_link(df, f'TE_{indicator_name.replace(" ", "_").lower()}')
    
    return [graph, html.H2('Data Table', style={'text-align': 'center'}), table, download_link]

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
