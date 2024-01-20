# 필요한 패키지 임포트
from dash import Dash, dcc, html, dash_table, Input, Output
from dash.dash_table.Format import Format, Scheme
import plotly.express as px
import pandas as pd
import openpyxl
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning)

# 파일 경로와 시트 이름 정의
file_path = r'C:\Users\서재영\Documents\Python Scripts\data\TDF모니터링.xlsm'
sheet_name = 'Sheet1'

# 앱 초기화
app = Dash(__name__)

# 엑셀 파일 내 특정 셀 범위에서 데이터를 읽어오는 함수
def read_data_from_excel(file_path, sheet_name, cell_range):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    sheet = wb[sheet_name]
    data = []
    for row in sheet[cell_range]:
        data.append([cell.value for cell in row])
    df = pd.DataFrame(data[1:], columns=data[0])
    return df

# 맞춤형 옵션으로 데이터 테이블을 생성하는 함수
def create_data_table(file_path, sheet_name, cell_range, percent_columns=None, font_size=14, header_style=None):
    df = read_data_from_excel(file_path, sheet_name, cell_range)
    
    # 열의 너비 계산
    num_columns = len(df.columns)
    column_width = "{}%".format(100 / num_columns)

    cell_style = {
        'whiteSpace': 'normal',
        'textAlign': 'center',
        'fontSize': 14, #font_size,
        'width': column_width,  # 모든 열에 동일한 너비 적용
        'minWidth': column_width,  # 최소 너비 설정
        'maxWidth': column_width,  # 최대 너비 설정
    }

    # 첫번째 열을 텍스트로 표시
    columns = [{"name": i, "id": i, "type": "text"} for i in df.columns]

    if percent_columns:
        for idx in percent_columns:
            adjusted_idx = idx - 1
            if 0 <= adjusted_idx < len(columns):
                columns[adjusted_idx]['type'] = 'numeric'
                columns[adjusted_idx]['format'] = Format(precision=2, scheme=Scheme.percentage)

    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=columns,
        page_size=10,
        style_cell=cell_style,
        style_header=header_style,
    )

# 테이블 1의 데이터 불러오기
df_table1 = read_data_from_excel(file_path, sheet_name, 'A4:G12')

# 테이블 2, 3, 4의 데이터 불러오기
df_table2 = read_data_from_excel(file_path, sheet_name, 'I4:O12')
df_table3 = read_data_from_excel(file_path, sheet_name, 'Q4:U13')
df_table4 = read_data_from_excel(file_path, sheet_name, 'W4:AA13')

# 그래프 형식 (Line, Bar, Dot, Pie) 목록
graph_types = ['Line', 'Bar', 'Dot', 'Pie']

# 앱 레이아웃 설정
app.layout = html.Div([
    html.Div(children='TDF 모니터링 from TDF2 Dash'),

    # 첫 번째 테이블과 그래프
    html.Div([
        html.Div([html.H4('Table 1'), create_data_table(file_path, sheet_name, 'A4:G12', [2, 3, 4, 5, 6, 7], 10, {'backgroundColor': 'lightblue', 'textAlign': 'center'})], className='포커스 수익률'),
        html.Div([
            dcc.Dropdown(
                id='yaxis-column-table1',
                options=[{'label': i, 'value': i} for i in df_table1.columns[1:]],  # 첫 번째 열 제외
                value=df_table1.columns[1]
            ),
            dcc.Dropdown(
                id='graph-type-table1',
                options=[{'label': i, 'value': i} for i in graph_types],
                value='Line'
            ),
            dcc.Graph(id='line-graph-table1')
        ])
    ]),

    # 나머지 테이블 및 그래프
    html.Div([
        html.Div([html.H4('Table 2'), create_data_table(file_path, sheet_name, 'I4:O12', [], 10, {'backgroundColor': 'lightblue', 'textAlign': 'center'})], className='포커스 Rank'),
        html.Div([
            dcc.Dropdown(
                id='yaxis-column-table2',
                options=[{'label': i, 'value': i} for i in df_table2.columns[1:]],  # 첫 번째 열 제외
                value=df_table2.columns[1]
            ),
            dcc.Dropdown(
                id='graph-type-table2',
                options=[{'label': i, 'value': i} for i in graph_types],
                value='Line'
            ),
            dcc.Graph(id='line-graph-table2')
        ])
    ], className='grid-item'),

    html.Div([
        html.Div([html.H4('Table 3'), create_data_table(file_path, sheet_name, 'Q4:U13', [3, 5], 10, {'backgroundColor': 'lightblue', 'textAlign': 'center'})], className='설정액_시장전체'),
        html.Div([
            dcc.Dropdown(
                id='yaxis-column-table3',
                options=[{'label': i, 'value': i} for i in df_table3.columns[1:]],  # 첫 번째 열 제외
                value=df_table3.columns[1]
            ),
            dcc.Dropdown(
                id='graph-type-table3',
                options=[{'label': i, 'value': i} for i in graph_types],
                value='Line'
            ),
            dcc.Graph(id='line-graph-table3')
        ])
    ], className='grid-item'),

    html.Div([
        html.Div([html.H4('Table 4'), create_data_table(file_path, sheet_name, 'W4:AA13', [3, 5], 10, {'backgroundColor': 'lightblue', 'textAlign': 'center'})], className='설정액_포커스'),
        html.Div([
            dcc.Dropdown(
                id='yaxis-column-table4',
                options=[{'label': i, 'value': i} for i in df_table4.columns[1:]],  # 첫 번째 열 제외
                value=df_table4.columns[1]
            ),
            dcc.Dropdown(
                id='graph-type-table4',
                options=[{'label': i, 'value': i} for i in graph_types],
                value='Line'
            ),
            dcc.Graph(id='line-graph-table4')
        ])
    ], className='grid-item')
])

# 콜백 함수 정의
@app.callback(
    Output('line-graph-table1', 'figure'),
    [Input('yaxis-column-table1', 'value'),
     Input('graph-type-table1', 'value')]
)
def update_graph_table1(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        fig = px.scatter(df_table1, x=df_table1.columns[0], y=yaxis_column_name)
    elif graph_type == 'Bar':
        fig = px.bar(df_table1, x=df_table1.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = px.pie(df_table1, names=df_table1.columns[0], values=yaxis_column_name)
    else:
        fig = px.line(df_table1, x=df_table1.columns[0], y=yaxis_column_name)
    return fig

@app.callback(
    Output('line-graph-table2', 'figure'),
    [Input('yaxis-column-table2', 'value'),
     Input('graph-type-table2', 'value')]
)
def update_graph_table2(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        fig = px.scatter(df_table2, x=df_table2.columns[0], y=yaxis_column_name)
    elif graph_type == 'Bar':
        fig = px.bar(df_table2, x=df_table2.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = px.pie(df_table2, names=df_table2.columns[0], values=yaxis_column_name)
    else:
        fig = px.line(df_table2, x=df_table2.columns[0], y=yaxis_column_name)
    return fig

@app.callback(
    Output('line-graph-table3', 'figure'),
    [Input('yaxis-column-table3', 'value'),
     Input('graph-type-table3', 'value')]
)
def update_graph_table3(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        fig = px.scatter(df_table3, x=df_table3.columns[0], y=yaxis_column_name)
    elif graph_type == 'Bar':
        fig = px.bar(df_table3, x=df_table3.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = px.pie(df_table3, names=df_table3.columns[0], values=yaxis_column_name)
    else:
        fig = px.line(df_table3, x=df_table3.columns[0], y=yaxis_column_name)
    return fig

@app.callback(
    Output('line-graph-table4', 'figure'),
    [Input('yaxis-column-table4', 'value'),
     Input('graph-type-table4', 'value')]
)
def update_graph_table4(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        fig = px.scatter(df_table4, x=df_table4.columns[0], y=yaxis_column_name)
    elif graph_type == 'Bar':
        fig = px.bar(df_table4, x=df_table4.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = px.pie(df_table4, names=df_table4.columns[0], values=yaxis_column_name)
    else:
        fig = px.line(df_table4, x=df_table4.columns[0], y=yaxis_column_name)
    return fig

# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=True)
