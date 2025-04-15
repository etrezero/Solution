import pandas as pd
from fredapi import Fred
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
import yaml
import urllib3
from flask import Flask
import socket
from dash import no_update

urllib3.disable_warnings()

# 설정 및 API 키 불러오기
yaml_path = r'C:/Covenant/config.yaml'
with open(yaml_path, encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)
FRED_key = _cfg['FRED_key']
fred = Fred(api_key=FRED_key)

start0 = (datetime.today() - relativedelta(years=20)).strftime('%Y-%m-%d')
end0 = datetime.today().strftime('%Y-%m-%d')






# ?앱 초기화 ============================================================
external_stylesheets = ["https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"]

server = Flask(__name__)
app = Dash(__name__, 
           external_stylesheets=external_stylesheets, 
           server = server)
app.title = "ECON FRED"




# 레이아웃
app.layout = html.Div([
    html.Div([
        html.H2("ECON FRED", className="text-center my-4 fw-bold text-primary"),

        html.Div([
            html.Label("태그로 시리즈 검색", className="form-label fw-semibold"),
            dcc.Input(
                id='tag-input', 
                type='text', 
                value='gdp', 
                debounce=True,
                placeholder='예: gdp, inflation, employment 등', 
                className="form-control mb-3",
                style={"width": "200px", "margin-left" : "1px"}
            ),

            html.Label("데이터 주기 선택", className="form-label fw-semibold"),
            dcc.Dropdown(
                id='freq-dropdown',
                options=[
                    {'label': '일간 (Daily)', 'value': 'd'},
                    {'label': '주간 (Weekly)', 'value': 'w'},
                    {'label': '월간 (Monthly)', 'value': 'm'},
                    {'label': '분기별 (Quarterly)', 'value': 'q'},
                    {'label': '연간 (Annual)', 'value': 'a'}
                ],
                value='q',
                className='mb-4',
                style={"width": "200px", "margin-left" : "1px"}
            ),

            html.Label("시리즈 선택", className="form-label fw-semibold"),
            dcc.Dropdown(id='series-dropdown', className='mb-4', 
                         style={"width": "110%"})
        ], className="container", 
        style={"width": "80%", "margin-left" : "1px"}),


        html.Div([
            dcc.Graph(id='my-graph', style={"height": "50vh"}),
            html.Div(id='error-message', className="text-danger text-center mt-2")  # 에러 메시지 표시
        ], className="container mb-5"),

        html.Div([
            html.H4('최근 10개 데이터', className='fw-bold mt-4 mb-3'),
            dash_table.DataTable(
                id='my-table',
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                style_cell={
                    'textAlign': 'center',
                    'padding': '10px',
                    'font-family': 'Segoe UI'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f2f2f2'
                    }
                ]
            )
        ], className="container")
    ], style={"width": "50%", "margin": "auto"})
], className="bg-light")




# 🔍 태그 검색 → 드롭다운 업데이트
@app.callback(
    Output('series-dropdown', 'options'),
    Output('series-dropdown', 'value'),
    Input('tag-input', 'value')
)
def update_series_list(tag):
    if not tag or len(tag.strip()) < 2:
        return [], None

    url = "https://api.stlouisfed.org/fred/series/search"
    params = {
        "api_key": FRED_key,
        "search_text": tag,
        "file_type": "json",
        "limit": 50
    }
    response = requests.get(url, params=params, verify=False)
    data = response.json().get('seriess', [])

    options = [{'label': f"{item['title']} ({item['id']})", 'value': item['id']} for item in data]
    default_value = options[0]['value'] if options else None

    return options, default_value






@app.callback(
    Output('my-graph', 'figure'),
    Output('my-table', 'data'),
    Output('my-table', 'columns'),
    Output('error-message', 'children'),  # 추가
    Input('series-dropdown', 'value'),
    Input('freq-dropdown', 'value'),
)
def update_graph_and_table(series_id, freq):
    if not series_id:
        return go.Figure(), [], [], ""

    try:
        series_title = fred.get_series_info(series_id)['title']
        df = fred.get_series(
            series_id,
            observation_start=start0,
            observation_end=end0,
            frequency=freq
        )
        df = pd.DataFrame({series_id: df}).dropna()
        df.index = pd.to_datetime(df.index)

        last_value = df[series_id].iloc[-1]
        last_date = df.index[-1].strftime('%Y-%m-%d')

        fig = go.Figure([
            go.Scatter(x=df.index, y=df[series_id], mode='lines', name=series_title)
        ])
        fig.update_layout(
            title=series_title,
            title_font_size=20,
            margin=dict(l=20, r=20, t=50, b=20),
            template="plotly_white",
            annotations=[{
                'x': last_date, 'y': last_value, 'xref': 'x', 'yref': 'y',
                'text': f'{last_date}: {last_value:,.2f}', 'showarrow': True, 'arrowhead': 2,
                'ax': 20, 'ay': -30
            }]
        )

        recent = df.tail(10).reset_index()
        recent.columns = ['Date', 'Value']
        recent['Date'] = recent['Date'].dt.strftime('%Y-%m-%d')
        return fig, recent.to_dict('records'), [{'name': i, 'id': i} for i in recent.columns], ""

    except ValueError as e:
        msg = str(e)
        if "Bad Request" in msg and "frequency" in msg:
            return go.Figure(), [], [], html.Div(
                "⚠️ 선택한 시리즈는 해당 주기를 지원하지 않습니다. 다른 주기를 선택해주세요.",
                style={"color": "red", "fontWeight": "bold", "marginTop": "20px"}
            )
        return go.Figure(), [], [], html.Div(f"에러 발생: {msg}", style={"color": "red"})




if __name__ == '__main__':
    def find_available_port(start=8050, attempts=10):
        for port in range(start, start + attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("0.0.0.0", port))
                    return port
                except:
                    continue
        raise RuntimeError("포트를 찾을 수 없습니다.")

    port = find_available_port()
    print(f"서버 실행 중: http://localhost:{port}")
    app.run(debug=True, host="0.0.0.0", port=port)