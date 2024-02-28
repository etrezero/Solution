# 필요한 패키지 임포트
from dash import Dash, dcc, html, dash_table, Input, Output
from dash.dash_table.Format import Format, Scheme
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import openpyxl
import warnings
from tqdm import tqdm
import os
from openpyxl import Workbook
import datetime


# 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning)



# 모니터링 ---------> temp

# 파일 경로 및 시트 이름 정의
path_TDF = r'C:/Covenant/data/0.TDF_모니터링.xlsx'
path_Temp_TDF = r'C:/Covenant/data/Temp_TDF.xlsx'
sheet_설정액 = '설정액'

# 지정된 시트를 읽어와서 데이터프레임으로 저장
df_설정액 = pd.read_excel(path_TDF, sheet_name=sheet_설정액)

# 파일이 없는 경우 새 Workbook 생성
if not os.path.exists(path_Temp_TDF):
    wb = Workbook()
    wb.save(path_Temp_TDF)
    print(f"새 파일 '{path_Temp_TDF}' 생성됨.")

# DataFrame을 엑셀 시트로 저장
with pd.ExcelWriter(path_Temp_TDF, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_설정액.to_excel(writer, sheet_name=sheet_설정액, index=False)

print(f"'{sheet_설정액}' 시트를 '{sheet_설정액}' 시트로 저장했습니다.")




# 앱 초기화
app = Dash(__name__)

# 엑셀 파일 내 특정 셀 범위에서 데이터를 읽어오는 함수
def read_data_from_excel(path_Temp_TDF, sheet_설정액, cell_range):
    wb = openpyxl.load_workbook(path_Temp_TDF, data_only=True)
    sheet = wb[sheet_설정액]
    data = []
    for row in sheet[cell_range]:
        data.append([cell.value for cell in row])
    df = pd.DataFrame(data[1:], columns=data[0])
    return df

    # df.columns = [str(col) for col in df.columns]


# 데이터 테이블을 생성하는 함수
def create_data_table(path, sheet_name, cell_range, percent_columns, font_size=None, header_style=None):
    df = read_data_from_excel(path_Temp_TDF, sheet_설정액, cell_range)
    
    # 데이터프레임 컬럼 이름 변환 함수 정의
    def convert_column_names(columns):
        converted_columns = []
        for column in columns:
            if isinstance(column, datetime.datetime):
                converted_columns.append(column.strftime('%Y년 %m월'))
            elif column == '':
                converted_columns.append(' ')
            else:
                converted_columns.append(column)
        return converted_columns
    
    # 데이터프레임 컬럼 이름 변환 적용
    df.columns = convert_column_names(df.columns)


    # 열의 너비 계산
    num_columns = len(df.columns)
    column_width = "{}%".format(100 / num_columns)
            
      
    # 첫 번째 열을 텍스트로 표시
    columns = [{"name": str(i), "id": str(i), "type": "text" if i == df.columns[0] else "numeric"} for i in df.columns]

    # 모든 열을 numeric으로 설정 
    for column in columns[1:]:
        column['type'] = 'numeric'
        column['format'] = Format(precision=0, scheme=Scheme.fixed, group=",")  # 모든 숫자 열의 소수 자릿수를 0으로 설정

    # 퍼센트 컬럼으로 지정한 열만 소수점 2자리까지 보이도록 설정
    if percent_columns:
        for idx in percent_columns:
            adjusted_idx = idx - 1
            if 0 <= adjusted_idx < len(columns):
                columns[adjusted_idx]['format'] = Format(precision=1, scheme=Scheme.percentage)

    cell_style = {
        'width': column_width,
        'minWidth': column_width,
        'maxWidth': column_width,
        'whiteSpace': 'normal',
        'textAlign': 'center',
        'verticalAlign': 'middle',
        'fontSize': '12px',
        # 'fontWeight': 'bold',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis'
    }
    
    # 헤더 스타일 변경
    style_header = {
        'color': 'White', 
        'fontweight': 'Bold', 
        'background-color': '#3762AF', #'darkblue'
        # 'background-color': '#4BACC6', #연한 비취색
    }

    style_data_conditional = [
        {
            'if': {
                'filter_query': '{{{0}}} contains "포커스"'.format(col_name),
                'column_id': str(col_name)  # 문자열로 변경*중요함
            },
            'color': 'white',  # 폰트 컬러를 그레이 블루로 설정합니다.
            'background-color': '#3762AF',  # 폰트 컬러를 그레이 블루로 설정합니다.
            'fontWeight': 'bold'  # 볼드체로 설정합니다.
        }
        for col_name in df.columns
    ] + [
        {
            'if': {
                'filter_query': '{{{0}}} contains "TRP"'.format(col_name),
                'column_id': str(col_name)  # 문자열로 변경*중요함
            },
            'color': 'white',  # 폰트 컬러를 그레이 블루로 설정합니다.
            'background-color': '#630',  # 폰트 컬러를 그레이 블루로 설정합니다.
            'fontWeight': 'bold'  # 볼드체로 설정합니다.
        }
        for col_name in df.columns
    ]

    
    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=columns,
        page_size=18,  #행 개수 넘어가면 페이지 넘어가
        style_cell=cell_style,  # 테이블 셀 텍스트 정렬을 중앙으로 설정
        style_header=style_header,
        style_data_conditional=style_data_conditional
    )



# 테이블 셀 범위 정의
table_cell = {
    'table1': 'AO28:BC31',
    'table2': 'S15:AL25',
    'table3': 'AO33:BC63',
    'table4': 'S3:AG13',
    'table5': 'AJ3:AX13',
    'table6': 'J3:Q13',
    'table7': 'J15:Q25',
}

percent_column = {
    '%table1' : [],
    '%table2' : [16,18,19,20],
    '%table3' : [],
    '%table4' : [],
    '%table5' : [],
    '%table6' : [],
    '%table7' : [],
}

# 그리드 아이템 CSS <A구역 : B구역>
grid_style = {
    # 'display': 'inline-block',  # 인라인 블록 요소로 배치
    'width': '100%',  # 화면 절반을 차지
    'margin': '0 auto',
    'verticalAlign': 'top',  # 상단 정렬
    # 'padding' :200,
    # 'paddingLeft': '100px', 
    # 'border': '1px solid black',  # 바더 라인 추가
}

# 테이블 스타일
table_style = {
    'margin': '0 auto',
    'margin-top': '2%',
    'width': '100%',
    'textAlign': 'center',
    # 'fontsize' : '30px',
    'z-index': 3,  # 이 요소가 다른 요소 위에 위치함
    # 'padding' :300,
    # 'leftpadding' :300,
    # 'rightpadding' :300,
}

# # 그래프 형식 (Line, Bar, Dot, Pie) 목록
# graph_types = ['Line', 'Bar', 'Dot', 'Pie','Heatmap','Histogram']

# # 그래프 컨테이너 스타일
# graph_container_style = {
#     'position': 'relative',  # 상대적 위치 설정
#     'margin': '0 auto',
#     'width': '90%',  # 컨테이너 너비
#     'z-index': 1,  # 이 요소가 다른 요소 위에 위치함
#     # 'border': '1px solid black',  # 바더 라인 추가
# }

# # 그래프 스타일 설정
# graph_style = {
#     'margin': '0 auto',
#     'margin-top': '0%',
#     'width': '85%',
#     'z-index': 1,  # 이 요소가 다른 요소 위에 위치함
# }


# # 드롭다운 컨테이너 스타일
# dropdown_container_style = {
#     'position': 'absolute',  # 절대적 위치 설정
#     'width': '15%',  # 각 드롭다운의 너비
#     'right': '0%',  # 오른쪽에서 5% 떨어진 곳에 위치
#     'top': '70%',  # 상단에서 50% 위치
#     'transform': 'translateY(-50%)',  # Y축으로 -50% 이동하여 중앙 정렬
#     'border': '0px solid black',  # 바더 라인 추가
#     'z-index': 3,  # 이 요소가 다른 요소 위에 위치함
# }

# # 개별 드롭다운 스타일
# dropdown_style = {
#     'width': '100%',  # 각 드롭다운의 너비
#     'marginRight': '1%',  # 드롭다운 사이의 간격
#     'textAlign': 'center',
# }


# 앱 레이아웃 설정
app.layout = html.Div([
    # html.Div(children='한국투자 TDF ETF 포커스 모니터링', 
    #          style={
    #              'fontSize': 25, 
    #              'textAlign': 'center', 
    #              'position': 'sticky', 
    #              'top': '0', 
    #              'zIndex': '100',
    #              'background-color': 'white'
    #          }),
    
    # 테이블 1 
    html.Div([
        # 테이블 컨테이너
        html.Div([
            html.H3('TDF 설정액 증감'),
            create_data_table(
                path_Temp_TDF, 
                sheet_설정액, 
                table_cell['table1'],
                percent_column['%table1'], 
                15, 
                {'backgroundColor': 'blue', 'textAlign': 'center'}
            )
        ], className='table', style=table_style),

        # # 그래프 컨테이너
        # html.Div([
        #     dcc.Graph(id='line-graph-table1', style=graph_style),

        #     # 드롭다운 컨테이너
        #     html.Div([
        #         dcc.Dropdown(
        #             id='yaxis-column-table1',
        #             options=[{'label': i, 'value': i} for i in df_table1.columns[1:]],
        #             value=df_table1.columns[5],
        #             style=dropdown_style
        #         ),
        #         dcc.Dropdown(
        #             id='graph-type-table1',
        #             options=[{'label': i, 'value': i} for i in graph_types],
        #             value='Dot',
        #             style=dropdown_style
        #         )
        #     ], className='dropdown', style=dropdown_container_style),
        # ], className='graph', style=graph_container_style),
    ], className='grid', style=grid_style),


    # 테이블 2
    html.Div([
        # 테이블 컨테이너
        html.Div([
            html.H3('TDF 설정액 증감(시장전체)'),
            create_data_table(
                path_Temp_TDF, 
                sheet_설정액, 
                table_cell['table2'],
                percent_column['%table2'], 
                15, 
                {'backgroundColor': 'blue', 'textAlign': 'center'}
            )
        ], className='table', style=table_style),

        # # 그래프 컨테이너
        # html.Div([
        #     dcc.Graph(id='line-graph-table2', style=graph_style),

        #     # 드롭다운 컨테이너
        #     html.Div([
        #         dcc.Dropdown(
        #             id='yaxis-column-table2',
        #             options=[{'label': i, 'value': i} for i in df_table2.columns[1:]],
        #             value=df_table2.columns[5],
        #             style=dropdown_style
        #         ),
        #         dcc.Dropdown(
        #             id='graph-type-table2',
        #             options=[{'label': i, 'value': i} for i in graph_types],
        #             value='Dot',
        #             style=dropdown_style
        #         )
        #     ], className='dropdown', style=dropdown_container_style),
        # ], className='graph', style=graph_container_style),
    ], className='grid', style=grid_style),


    # 테이블 3
    html.Div([
        # 테이블 컨테이너
        html.Div([
            html.H3('TDF 설정액 증감(운용사별)'),
            create_data_table(
                path_Temp_TDF, 
                sheet_설정액, 
                table_cell['table3'],
                percent_column['%table3'], 
                15, 
                {'backgroundColor': 'blue', 'textAlign': 'center'}
            )
        ], className='table', style=table_style),

        # # 그래프 컨테이너
        # html.Div([
        #     dcc.Graph(id='line-graph-table3', style=graph_style),

        #     # 드롭다운 컨테이너
        #     html.Div([
        #         dcc.Dropdown(
        #             id='yaxis-column-table3',
        #             options=[{'label': i, 'value': i} for i in df_table3.columns[1:]],
        #             value=df_table3.columns[1],
        #             style=dropdown_style
        #         ),
        #         dcc.Dropdown(
        #             id='graph-type-table3',
        #             options=[{'label': i, 'value': i} for i in graph_types],
        #             value='Pie',
        #             style=dropdown_style
        #         )
        #     ], className='dropdown', style=dropdown_container_style),
        # ], className='graph', style=graph_container_style),
    ], className='grid', style=grid_style),



    # 테이블 4 
    html.Div([
        # 테이블 컨테이너
        html.Div([
            html.H3('설정액(포커스)'),
            create_data_table(
                path_Temp_TDF, 
                sheet_설정액, 
                table_cell['table4'],
                percent_column['%table4'], 
                15, 
                {'backgroundColor': 'blue', 'textAlign': 'center'}
            )
        ], className='table', style=table_style),

        # # 그래프 컨테이너
        # html.Div([
        #     dcc.Graph(id='line-graph-table4', style=graph_style),

        #     # 드롭다운 컨테이너
        #     html.Div([
        #         dcc.Dropdown(
        #             id='yaxis-column-table4',
        #             options=[{'label': i, 'value': i} for i in df_table4.columns[1:]],
        #             value=df_table4.columns[1],
        #             style=dropdown_style
        #         ),
        #         dcc.Dropdown(
        #             id='graph-type-table4',
        #             options=[{'label': i, 'value': i} for i in graph_types],
        #             value='Pie',
        #             style=dropdown_style
        #         )
        #     ], className='dropdown', style=dropdown_container_style),
        # ], className='graph', style=graph_container_style),
    ], className='grid', style=grid_style),


    # 테이블 5
    html.Div([
        # 테이블 컨테이너
        html.Div([
            html.H3('설정액(TRP)'),
            create_data_table(
                path_Temp_TDF, 
                sheet_설정액, 
                table_cell['table5'],
                percent_column['%table5'], 
                15, 
            )
        ], className='table', style=table_style),

        # # 그래프 컨테이너
        # html.Div([
        #     dcc.Graph(id='line-graph-table5', style=graph_style),

        #     # 드롭다운 컨테이너
        #     html.Div([
        #         dcc.Dropdown(
        #             id='yaxis-column-table5',
        #             options=[{'label': i, 'value': i} for i in df_table5.columns[1:]],
        #             value=df_table5.columns[1],
        #             style=dropdown_style
        #         ),
        #         dcc.Dropdown(
        #             id='graph-type-table5',
        #             options=[{'label': i, 'value': i} for i in graph_types],
        #             value='Pie',
        #             style=dropdown_style
        #         )
        #     ], className='dropdown', style=dropdown_container_style),
        # ], className='graph', style=graph_container_style),
    ], className='grid', style=grid_style),


    # 테이블 6
    html.Div([
        # 테이블 컨테이너
        html.Div([
            html.H3('설정액(포커스)'),
            create_data_table(
                path_Temp_TDF, 
                sheet_설정액, 
                table_cell['table6'],
                percent_column['%table6'], 
                15, 
                {'backgroundColor': 'blue', 'textAlign': 'center'}
            )
        ], className='table', style=table_style),

        # # 그래프 컨테이너
        # html.Div([
        #     dcc.Graph(id='line-graph-table6', style=graph_style),

        #     # 드롭다운 컨테이너
        #     html.Div([
        #         dcc.Dropdown(
        #             id='yaxis-column-table6',
        #             options=[{'label': i, 'value': i} for i in df_table6.columns[1:]],
        #             value=df_table6.columns[6],
        #             style=dropdown_style
        #         ),
        #         dcc.Dropdown(
        #             id='graph-type-table6',
        #             options=[{'label': i, 'value': i} for i in graph_types],
        #             value='Bar',
        #             style=dropdown_style
        #         )
        #     ], className='dropdown', style=dropdown_container_style),
        # ], className='graph', style=graph_container_style),
    ], className='grid', style=grid_style),



# 테이블 7
    html.Div([
        # 테이블 컨테이너
        html.Div([
            html.H3('설정액(TRP)'),
            create_data_table(
                path_Temp_TDF, 
                sheet_설정액, 
                table_cell['table7'],
                percent_column['%table7'], 
                15, 
                {'backgroundColor': 'blue', 'textAlign': 'center'}
            )
        ], className='table', style=table_style),

        # # 그래프 컨테이너
        # html.Div([
        #     dcc.Graph(id='line-graph-table7', style=graph_style),

        #     # 드롭다운 컨테이너
        #     html.Div([
        #         dcc.Dropdown(
        #             id='yaxis-column-table7',
        #             options=[{'label': i, 'value': i} for i in df_table7.columns[1:]],
        #             value=df_table7.columns[6],
        #             style=dropdown_style
        #         ),
        #         dcc.Dropdown(
        #             id='graph-type-table6',
        #             options=[{'label': i, 'value': i} for i in graph_types],
        #             value='Bar',
        #             style=dropdown_style
        #         )
        #     ], className='dropdown', style=dropdown_container_style),
        # ], className='graph', style=graph_container_style),
    ], className='grid', style=grid_style),

])


# # 콜백 함수 정의
# @app.callback(
#     Output('line-graph-table1', 'figure'),
#     [Input('yaxis-column-table1', 'value'),
#      Input('graph-type-table1', 'value')]
# )
# def update_graph_table1(yaxis_column_name, graph_type):
#     if graph_type == 'Dot':
#         # Dot 그래프 생성
#         fig = go.Figure(data=go.Scatter(
#             x=df_table1[df_table1.columns[0]].astype(str),
#             y=df_table1[yaxis_column_name],
#             mode='markers',  # 마커 모드 사용
#             marker=dict(
#                 size=15,  # 데이터 마크 크기 조절 (원하는 크기로 설정)
#                 opacity=0.8,
#                 line=dict(width=2, color='DarkSlateGrey')
#             )
#         ))
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
#         fig.update_xaxes(tickformat="Value", tickangle=-45, tickmode='auto', nticks=12) # 여기서 nticks를 조정하여 간격 변경
        
#     elif graph_type == 'Bar':
#         fig = px.bar(df_table1, x=df_table1.columns[0], y=yaxis_column_name)
#     elif graph_type == 'Pie':
#         fig = px.pie(df_table1, names=df_table1.columns[0], values=yaxis_column_name)
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
#     else:
#         fig = px.line(df_table1, x=df_table1.columns[0], y=yaxis_column_name)
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

#             # 그래프의 배경을 투명하게 설정
#     fig.update_layout(
#         paper_bgcolor='rgba(0,0,0,0)',
#         # plot_bgcolor='rgba(0,0,0,0)',
#     )
#     return fig





# @app.callback(
#     Output('line-graph-table2', 'figure'),
#     [Input('yaxis-column-table2', 'value'),
#      Input('graph-type-table2', 'value')]
# )

# def update_graph_table2(yaxis_column_name, graph_type):
#     if graph_type == 'Dot':
#         # Dot 그래프 생성
#         fig = go.Figure(data=go.Scatter(
#             x=df_table2[df_table2.columns[0]].astype(str),
#             y=df_table2[yaxis_column_name],
#             mode='markers',  # 마커 모드 사용
#             marker=dict(
#                 size=15,  # 데이터 마크 크기 조절 (원하는 크기로 설정)
#                 opacity=0.8,
#                 line=dict(width=2, color='DarkSlateGrey')
#             )
#         ))
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
#         fig.update_xaxes(tickformat="Value", tickangle=-45, tickmode='auto', nticks=12) # 여기서 nticks를 조정하여 간격 변경
        
#     elif graph_type == 'Bar':
#         fig = px.bar(df_table2, x=df_table2.columns[0], y=yaxis_column_name)
#     elif graph_type == 'Pie':
#         fig = px.pie(df_table2, names=df_table2.columns[0], values=yaxis_column_name)
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
#     else:
#         fig = px.line(df_table2, x=df_table2.columns[0], y=yaxis_column_name)
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

#     # 그래프의 배경을 투명하게 설정
#     fig.update_layout(
#         paper_bgcolor='rgba(0,0,0,0)',
#         # plot_bgcolor='rgba(0,0,0,0)'
#     )

#     return fig



# @app.callback(
#     Output('line-graph-table3', 'figure'),
#     [Input('yaxis-column-table3', 'value'),
#      Input('graph-type-table3', 'value')]
# )

# def update_graph_table3(yaxis_column_name, graph_type):
#     if graph_type == 'Dot':
#         # Dot 그래프 생성
#         fig = go.Figure(data=go.Scatter(
#             x=df_G3[df_G3.columns[0]].astype(str),
#             y=df_G3[yaxis_column_name],
#             mode='markers',  # 마커 모드 사용
#             marker=dict(
#                 size=15,  # 데이터 마크 크기 조절 (원하는 크기로 설정)
#                 opacity=0.8,
#                 line=dict(width=2, color='DarkSlateGrey')
#             )
#         ))
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
#         fig.update_xaxes(tickformat="Value", tickangle=-45, tickmode='auto', nticks=12) # 여기서 nticks를 조정하여 간격 변경
        
#     elif graph_type == 'Bar':
#         fig = px.bar(df_G3, x=df_G3.columns[0], y=yaxis_column_name)
#     elif graph_type == 'Pie':
#         fig = px.pie(df_G3, names=df_G3.columns[0], values=yaxis_column_name)
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
#     else:
#         fig = px.line(df_G3, x=df_G3.columns[0], y=yaxis_column_name)
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

#     # 그래프의 배경을 투명하게 설정
#     fig.update_layout(
#         paper_bgcolor='rgba(0,0,0,0)',
#         # plot_bgcolor='rgba(0,0,0,0)',
#     )

#     return fig


# @app.callback(
#     Output('line-graph-table4', 'figure'),
#     [Input('yaxis-column-table4', 'value'),
#      Input('graph-type-table4', 'value')]
# )

# def update_graph_table4(yaxis_column_name, graph_type):
#     if graph_type == 'Dot':
#         # Dot 그래프 생성
#         fig = go.Figure(data=go.Scatter(
#             x=df_G4[df_G4.columns[0]].astype(str),
#             y=df_G4[yaxis_column_name],
#             mode='markers',  # 마커 모드 사용
#             marker=dict(
#                 size=15,  # 데이터 마크 크기 조절 (원하는 크기로 설정)
#                 opacity=0.8,
#                 line=dict(width=2, color='DarkSlateGrey')
#             )
#         ))
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
#         fig.update_xaxes(tickformat="Value", tickangle=-45, tickmode='auto', nticks=12) # 여기서 nticks를 조정하여 간격 변경
        
#     elif graph_type == 'Bar':
#         fig = px.bar(df_G4, x=df_G4.columns[0], y=yaxis_column_name)
#     elif graph_type == 'Pie':
#         fig = px.pie(df_G4, names=df_G4.columns[0], values=yaxis_column_name)
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
#     else:
#         fig = px.line(df_G4, x=df_G4.columns[0], y=yaxis_column_name)
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

#     # 그래프의 배경을 투명하게 설정
#     fig.update_layout(
#         paper_bgcolor='rgba(0,0,0,0)',
#         # plot_bgcolor='rgba(0,0,0,0)',
#     )

#     return fig


# @app.callback(
#     Output('line-graph-table5', 'figure'),
#     [Input('yaxis-column-table5', 'value'),
#      Input('graph-type-table5', 'value')]
# )

# def update_graph_table5(yaxis_column_name, graph_type):
#     if graph_type == 'Dot':
#         # Dot 그래프 생성
#         fig = go.Figure(data=go.Scatter(
#             x=df_table5[df_table5.columns[0]].astype(str),
#             y=df_table5[yaxis_column_name],
#             mode='markers',  # 마커 모드 사용
#             marker=dict(
#                 size=15,  # 데이터 마크 크기 조절 (원하는 크기로 설정)
#                 opacity=0.8,
#                 line=dict(width=2, color='DarkSlateGrey')
#             )
#         ))
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
#         fig.update_xaxes(tickformat="Value", tickangle=-45, tickmode='auto', nticks=12) # 여기서 nticks를 조정하여 간격 변경
        
#     elif graph_type == 'Bar':
#         fig = px.bar(df_table5, x=df_table5.columns[0], y=yaxis_column_name)
#     elif graph_type == 'Pie':
#         fig = px.pie(df_table5, names=df_table5.columns[0], values=yaxis_column_name)
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
#     else:
#         fig = px.line(df_table5, x=df_table5.columns[0], y=yaxis_column_name)
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

#     # 그래프의 배경을 투명하게 설정
#     fig.update_layout(
#         paper_bgcolor='rgba(0,0,0,0)',
#         # plot_bgcolor='rgba(0,0,0,0)',
#         yaxis=dict(domain=[0, 0.5]), #하단반쪽   cf)상단반쪽 [0.5,1]

#     )

#     return fig



# @app.callback(
#     Output('line-graph-table6', 'figure'),
#     [Input('yaxis-column-table6', 'value'),
#      Input('graph-type-table6', 'value')]
# )
# def update_graph_table6(yaxis_column_name, graph_type):
#     if graph_type == 'Dot':
#         # Dot 그래프 생성
#         fig = go.Figure(data=go.Scatter(
#             x=df_table6[df_table6.columns[0]].astype(str),
#             y=df_table6[yaxis_column_name],
#             mode='markers',  # 마커 모드 사용
#             marker=dict(
#                 size=15,  # 데이터 마크 크기 조절 (원하는 크기로 설정)
#                 opacity=0.8,
#                 line=dict(width=2, color='DarkSlateGrey')
#             )
#         ))
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
#         fig.update_xaxes(tickformat="Value", tickangle=-45, tickmode='auto', nticks=12) # 여기서 nticks를 조정하여 간격 변경
        
#     elif graph_type == 'Bar':
#         fig = px.bar(df_table6, x=df_table6.columns[0], y=yaxis_column_name)
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
#     elif graph_type == 'Pie':
#         fig = px.pie(df_table6, names=df_table6.columns[0], values=yaxis_column_name)
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
#     else:
#         fig = px.line(df_table6, x=df_table6.columns[0], y=yaxis_column_name)
#         fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

#     # 그래프의 배경을 투명하게 설정
#     fig.update_layout(
#         paper_bgcolor='rgba(0,0,0,0)',
#         # plot_bgcolor='rgba(0,0,0,0)',
#         height=700,
#     )

#     return fig

# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=True,host='0.0.0.0') 


    #http://192.168.194.140:8050