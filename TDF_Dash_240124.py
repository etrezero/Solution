# 필요한 패키지 임포트
from dash import Dash, dcc, html, dash_table, Input, Output
from dash.dash_table.Format import Format, Scheme
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import openpyxl
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning)

# 파일 경로와 시트 이름 정의
file_path = r'C:\Users\서재영\Documents\Python Scripts\data\TDF 템플릿.xlsx'
sheet_name = '요약'

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
        'fontSize': font_size,
        'width': column_width,  # 모든 열에 동일한 너비 적용
        'minWidth': column_width,  # 최소 너비 설정
        'maxWidth': column_width,  # 최대 너비 설정
    }

    # 첫 번째 열을 텍스트로 표시
    columns = [{"name": i, "id": i, "type": "text" if i == df.columns[0] else "numeric"} for i in df.columns]

    # 모든 열을 정수 숫자로 설정 (첫 번째 열은 텍스트로 설정됨)
    for column in columns[0:]:
        column['format'] = Format(precision=0, scheme=Scheme.fixed)

    # 퍼센트 컬럼으로 지정한 열만 소수점 2자리까지 보이도록 설정
    if percent_columns:
        for idx in percent_columns:
            adjusted_idx = idx - 1
            if 0 <= adjusted_idx < len(columns):
                columns[adjusted_idx]['format'] = Format(precision=2, scheme=Scheme.percentage)

    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=columns,
        page_size=10,
        style_cell=cell_style,
        style_header={'color': 'White', 'fontweight': 'Bold', 'background-color': 'darkblue'},  # 헤더 스타일 변경
    )


# 테이블의 데이터 불러오기
df_table1 = read_data_from_excel(file_path, sheet_name, 'A3:G11')
df_table2 = read_data_from_excel(file_path, sheet_name, 'I3:O11')
df_table3 = read_data_from_excel(file_path, sheet_name, 'Q3:U11')
df_table4 = read_data_from_excel(file_path, sheet_name, 'W3:AA12')
df_table5 = read_data_from_excel(file_path, sheet_name, 'AC3:AK9')
df_table6 = read_data_from_excel(file_path, sheet_name, 'AW43:BE58')

# 그래프 형식 (Line, Bar, Dot, Pie) 목록
graph_types = ['Line', 'Bar', 'Dot', 'Pie']

# 그리드 아이템 CSS <A구역 : B구역>
grid_item_style = {
    'display': 'inline-block',  # 인라인 블록 요소로 배치
    'width': '49%',  # 화면 절반을 차지
    'justifyContent': 'center',  # 수평 가운데 정렬
    'verticalAlign': 'top',  # 상단 정렬
    # 'border': '1px solid black',  # 바더 라인 추가
    'padding' : 5
}

# 공통 스타일 설정(테이블, 그래프, 드롭다운 공통)
common_style = {
    'margin': '0 auto',
    'width': '70%',
}

# 그래프 스타일 설정
graph_style = {
    # 'display': 'inline-block',
    'width': '105%',
}

# 드롭다운 스타일 설정
dropdown_style = {
    'width': '40%',
    'display': 'inline-block',
    'marginLeft': '10px',
    'verticalAlign': 'top',
    'textAlign': 'center',
}

# 공통 셀 범위 정의
common_cell_range = {
    'table1': 'A3:G11',
    'table2': 'I3:O11',
    'table3': 'Q3:U12',
    'table4': 'W3:AA12',
    'table5': 'AC3:AK9',
    'table6': 'AW43:BE58'
}

# 앱 레이아웃 설정
app.layout = html.Div([
    html.Div(children='한국투자 TDF ETF 포커스 모니터링', style={'fontSize': 25, 'textAlign': 'center', 'position': 'sticky', 'top': '0', 'zIndex': '100','background-color': 'white'}),    # 홀수 테이블1 및 그래프1
    html.Div([
        html.Div([html.H3('수익률'), create_data_table(file_path, sheet_name, common_cell_range['table1'], [2, 3, 4, 5, 6, 7], 10, {'backgroundColor': 'blue', 'textAlign': 'center'})], className='포커스 수익률', style=common_style),
        html.Div([
            dcc.Graph(id='line-graph-table1', style=graph_style),
            dcc.Dropdown(
                id='yaxis-column-table1',
                options=[{'label': i, 'value': i} for i in df_table1.columns[1:]],  # 첫 번째 열 제외
                value=df_table1.columns[1],
                style={**dropdown_style, 'width': '40%'},
            ),
            dcc.Dropdown(
                id='graph-type-table1',
                options=[{'label': i, 'value': i} for i in graph_types],
                value='Dot',
                style={**dropdown_style, 'marginLeft': '5px'},
            ),
        ], style=common_style)  # 그래프 크기
    ], className='grid-item', style=grid_item_style),

    # 짝수 테이블2 및 그래프2
    html.Div([
        html.Div([html.H3('변동성'), create_data_table(file_path, sheet_name, common_cell_range['table2'], [2, 3, 4, 5, 6, 7], 10, {'backgroundColor': 'blue', 'textAlign': 'center'})], className='포커스 Rank', style=common_style),
        html.Div([
            dcc.Graph(id='line-graph-table2', style=graph_style),
            dcc.Dropdown(
                id='yaxis-column-table2',
                options=[{'label': i, 'value': i} for i in df_table2.columns[1:]],  # 첫 번째 열 제외
                value=df_table2.columns[1],
                style={**dropdown_style, 'width': '40%'},
            ),
            dcc.Dropdown(
                id='graph-type-table2',
                options=[{'label': i, 'value': i} for i in graph_types],
                value='Dot',
                style={**dropdown_style, 'marginLeft': '5px'},
            ),
        ], style=common_style) # 그래프 크기
    ], className='grid-item', style=grid_item_style),

    # 홀수 테이블3 및 그래프3
    html.Div([
        html.Div([html.H3('설정액<시장전체>'), create_data_table(file_path, sheet_name, common_cell_range['table3'], [3, 5], 10, {'backgroundColor': 'blue', 'textAlign': 'center'})], className='설정액_시장전체', style=common_style),
        html.Div([
            dcc.Graph(id='line-graph-table3', style=graph_style),
            dcc.Dropdown(
                id='yaxis-column-table3',
                options=[{'label': i, 'value': i} for i in df_table3.columns[1:]],  # 첫 번째 열 제외
                value=df_table3.columns[1],
                style={**dropdown_style, 'width': '40%'},
            ),
            dcc.Dropdown(
                id='graph-type-table3',
                options=[{'label': i, 'value': i} for i in graph_types],
                value='Bar',
                style={**dropdown_style, 'marginLeft': '5px'},
            ),
        ], style=common_style) # 그래프 크기
    ], className='grid-item', style=grid_item_style),

    # 짝수 테이블4 및 그래프4
    html.Div([
        html.Div([html.H3('설정액<ETF포커스>'), create_data_table(file_path, sheet_name, common_cell_range['table4'], [3, 5], 10, {'backgroundColor': 'blue', 'textAlign': 'center'})], className='설정액_포커스', style=common_style),
        html.Div([
            dcc.Graph(id='line-graph-table4', style=graph_style),
            dcc.Dropdown(
                id='yaxis-column-table4',
                options=[{'label': i, 'value': i} for i in df_table4.columns[1:]],  # 첫 번째 열 제외
                value=df_table4.columns[1],
                style={**dropdown_style, 'width': '40%'},
            ),
            dcc.Dropdown(
                id='graph-type-table4',
                options=[{'label': i, 'value': i} for i in graph_types],
                value='Bar',
                style={**dropdown_style, 'marginLeft': '5px'},
            ),
        ], style=common_style) # 그래프 크기
    ], className='grid-item', style=grid_item_style),
    
    
    # 짝수 테이블5 및 그래프5
    html.Div([
        # html.Div([html.H3('자산배분'), create_data_table(file_path, sheet_name, common_cell_range['table5'], [2,3,4,5,6,7,8,9], 10, {'backgroundColor': 'blue', 'textAlign': 'center'})], className='자산배분', style=common_style),

        html.Div([
            dcc.Graph(id='line-graph-table5', style=graph_style),
            dcc.Dropdown(
                id='yaxis-column-table5',
                options=[{'label': i, 'value': i} for i in df_table5.columns[1:]],  # 첫 번째 열 제외
                value=df_table5.columns[1],
                style={**dropdown_style, 'width': '40%'},
            ),
            dcc.Dropdown(
                id='graph-type-table5',
                options=[{'label': i, 'value': i} for i in graph_types],
                value='Pie',
                style={**dropdown_style, 'marginLeft': '5px'},
            ),
        ], style=common_style) # 그래프 크기
    ], className='grid-item', style=grid_item_style),
    
    
    
    # 짝수 테이블6 및 그래프6
    html.Div([
        # html.Div([html.H3('투자비중'), create_data_table(file_path, sheet_name, common_cell_range['table6'], [2,3,4,5,6,7,8,9], 10, {'backgroundColor': 'blue', 'textAlign': 'center'})], className='자산배분', style=common_style),
        html.Div([
            dcc.Graph(id='line-graph-table6', style=graph_style),
            dcc.Dropdown(
                id='yaxis-column-table6',
                options=[{'label': i, 'value': i} for i in df_table6.columns[1:]],  # 첫 번째 열 제외
                value=df_table6.columns[1],
                style={**dropdown_style, 'width': '40%'},
            ),
            dcc.Dropdown(
                id='graph-type-table6',
                options=[{'label': i, 'value': i} for i in graph_types],
                value='Pie',
                style={**dropdown_style, 'marginLeft': '5px'},
            ),
        ], style=common_style) # 그래프 크기
    ], className='grid-item', style=grid_item_style),

])

# 콜백 함수 정의
@app.callback(
    Output('line-graph-table1', 'figure'),
    [Input('yaxis-column-table1', 'value'),
     Input('graph-type-table1', 'value')]
)
def update_graph_table1(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = go.Figure(data=go.Scatter(
            x=df_table1[df_table1.columns[0]].astype(str),
            y=df_table1[yaxis_column_name],
            mode='markers',  # 마커 모드 사용
            marker=dict(
                size=15,  # 데이터 마크 크기 조절 (원하는 크기로 설정)
                opacity=0.8,
                line=dict(width=2, color='DarkSlateGrey')
            )
        ))
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
        
    elif graph_type == 'Bar':
        fig = px.bar(df_table1, x=df_table1.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = px.pie(df_table1, names=df_table1.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = px.line(df_table1, x=df_table1.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    return fig



@app.callback(
    Output('line-graph-table2', 'figure'),
    [Input('yaxis-column-table2', 'value'),
     Input('graph-type-table2', 'value')]
)

def update_graph_table2(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = go.Figure(data=go.Scatter(
            x=df_table2[df_table1.columns[0]],
            y=df_table2[yaxis_column_name] ,  # 백분율로 표시하기 위해 100을 곱함
            mode='markers',  # 마커 모드 사용
            marker=dict(
                size=15,  # 데이터 마크 크기 조절 (원하는 크기로 설정)
                opacity=0.8,
                line=dict(width=2, color='DarkSlateGrey')
            )
        ))
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 소수점 1자리 백분율로 설정
        
    elif graph_type == 'Bar':
        fig = px.bar(df_table2, x=df_table2.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 소수점 1자리 백분율로 설정
        
    elif graph_type == 'Pie':
        fig = px.pie(df_table2, names=df_table2.columns[0], values=yaxis_column_name)
    else:
        fig = px.line(df_table2, x=df_table2.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 소수점 1자리 백분율로 설정
        
    return fig


@app.callback(
    Output('line-graph-table3', 'figure'),
    [Input('yaxis-column-table3', 'value'),
     Input('graph-type-table3', 'value')]
)
def update_graph_table3(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = go.Figure(data=go.Scatter(
            x=df_table1[df_table1.columns[0]],
            y=df_table1[yaxis_column_name],
            mode='markers',  # 마커 모드 사용
            marker=dict(
                size=10,  # 데이터 마크 크기 조절 (원하는 크기로 설정)
                opacity=0.8,
                line=dict(width=2, color='DarkSlateGrey')
            )
        ))
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
        
    elif graph_type == 'Bar':
        fig = px.bar(df_table3, x=df_table3.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
        
    elif graph_type == 'Pie':
        fig = px.pie(df_table3, names=df_table3.columns[0], values=yaxis_column_name)
    else:
        fig = px.line(df_table3, x=df_table3.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
        
    return fig

@app.callback(
    Output('line-graph-table4', 'figure'),
    [Input('yaxis-column-table4', 'value'),
     Input('graph-type-table4', 'value')]
)
def update_graph_table4(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = go.Figure(data=go.Scatter(
            x=df_table1[df_table1.columns[0]],
            y=df_table1[yaxis_column_name],
            mode='markers',  # 마커 모드 사용
            marker=dict(
                size=10,  # 데이터 마크 크기 조절 (원하는 크기로 설정)
                opacity=0.8,
                line=dict(width=2, color='DarkSlateGrey')
            )
        ))
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
        
    elif graph_type == 'Bar':
        fig = px.bar(df_table4, x=df_table4.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
        
    elif graph_type == 'Pie':
        fig = px.pie(df_table4, names=df_table4.columns[0], values=yaxis_column_name)
    else:
        fig = px.line(df_table4, x=df_table4.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
        
    return fig


@app.callback(
    Output('line-graph-table5', 'figure'),
    [Input('yaxis-column-table5', 'value'),
     Input('graph-type-table5', 'value')]
)
def update_graph_table5(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = go.Figure(data=go.Scatter(
            x=df_table1[df_table1.columns[0]],
            y=df_table1[yaxis_column_name],
            mode='markers',  # 마커 모드 사용
            marker=dict(
                size=10,  # 데이터 마크 크기 조절 (원하는 크기로 설정)
                opacity=0.8,
                line=dict(width=2, color='DarkSlateGrey')
            )
        ))
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
        
    elif graph_type == 'Bar':
        fig = px.bar(df_table5, x=df_table5.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
        
    elif graph_type == 'Pie':
        fig = px.pie(df_table5, names=df_table5.columns[0], values=yaxis_column_name)
    else:
        fig = px.line(df_table5, x=df_table5.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
        
    return fig

@app.callback(
    Output('line-graph-table6', 'figure'),
    [Input('yaxis-column-table6', 'value'),
     Input('graph-type-table6', 'value')]
)
def update_graph_table6(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = go.Figure(data=go.Scatter(
            x=df_table1[df_table1.columns[0]],
            y=df_table1[yaxis_column_name],
            mode='markers',  # 마커 모드 사용
            marker=dict(
                size=10,  # 데이터 마크 크기 조절 (원하는 크기로 설정)
                line=dict(width=2, color='DarkSlateGrey')
            )
        ))
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
        
    elif graph_type == 'Bar':
        fig = px.bar(df_table6, x=df_table6.columns[0].astype(str), y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
        
    elif graph_type == 'Pie':
        fig = px.pie(df_table6, names=df_table6.columns[0], values=yaxis_column_name)
    else:
        fig = px.line(df_table6, x=df_table6.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
        
    return fig


# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=True)