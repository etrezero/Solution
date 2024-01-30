# 필요한 패키지 임포트
from dash import Dash, dcc, html, dash_table, Input, Output
from dash.dash_table.Format import Format, Scheme
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import openpyxl



# 파일 경로와 시트 이름 정의
# file_path = r'C:\Users\USER\Desktop\Excel Project\templates\TDF_템플릿.xlsx'
file_path = r'C:\Users\서재영\Documents\Python Scripts\data\TDF_템플릿.xlsx'
sheet_name = '요약'



# 엑셀 파일 내 특정 셀 범위에서 데이터를 읽어오는 함수
def read_data_from_excel(file_path, sheet_name, cell_range):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    sheet = wb[sheet_name]
    data = []
    for row in sheet[cell_range]:
        data.append([cell.value for cell in row])
    df = pd.DataFrame(data[1:], columns=data[0])
    return df

# 데이터 테이블을 생성하는 함수
def create_data_table(file_path, sheet_name, cell_range, percent_columns, font_size=None, header_style=None):
    df = read_data_from_excel(file_path, sheet_name, cell_range)
    
    # 열의 너비 계산
    num_columns = len(df.columns)
    print(num_columns)
            
    style_cell = [
    {
        'if': {'column_id': col},
        'width': "{}%".format(100 / num_columns),  # 모든 열의 가로 크기를 동일하게 설정
        'minWidth': "{}%".format(100 / num_columns),
        'maxWidth': "{}%".format(100 / num_columns),
        'whiteSpace': 'normal',
        'textAlign': 'center',  # 수평 중앙 정렬
        'verticalAlign': 'middle',  # 수직 중앙 정렬
        'fontSize': '12px',
        'fontweight': 'bold',
        'overflow': 'hidden',  # 데이터가 셀을 벗어나지 않도록 설정
        'textOverflow': 'ellipsis',  # 데이터가 셀을 벗어나면 잘리고 생략 부호(...)를 표시
    }
    for col in df.columns
]

    # 첫 번째 열을 텍스트로 표시
    columns = [{"name": str(i), "id": str(i), "type": "text" if i == df.columns[0] else "numeric"} for i in df.columns]

    # 모든 열을 numeric으로 설정 (첫 번째 열은 텍스트로 설정됨)
    for column in columns[1:]:
        column['type'] = 'numeric'
        column['format'] = Format(precision=0, scheme=Scheme.fixed)  # 모든 숫자 열의 소수 자릿수를 0으로 설정

    # 퍼센트 컬럼으로 지정한 열만 소수점 2자리까지 보이도록 설정
    if percent_columns:
        for idx in percent_columns:
            adjusted_idx = idx - 1
            if 0 <= adjusted_idx < len(columns):
                columns[adjusted_idx]['format'] = Format(precision=1, scheme=Scheme.percentage)

    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=columns,
        page_size=9,  #행 개수 넘어가면 페이지 넘어가
        style_cell={'textAlign': 'center'},  # 테이블 셀 텍스트 정렬을 중앙으로 설정
        style_header={'color': 'White', 'fontweight': 'Bold', 'background-color': 'darkblue'},  # 헤더 스타일 변경
    )


# 테이블의 데이터 불러오기
df_table1 = read_data_from_excel(file_path, sheet_name, 'A3:G11')
df_table2 = read_data_from_excel(file_path, sheet_name, 'I3:O11')
df_table3 = read_data_from_excel(file_path, sheet_name, 'Q3:U11')
df_table4 = read_data_from_excel(file_path, sheet_name, 'W3:AA12')
df_table5 = read_data_from_excel(file_path, sheet_name, 'AC3:AK9')
df_table6 = read_data_from_excel(file_path, sheet_name, 'AW43:BE58')


# 마지막 행이 없는 데이터 - 그래프용
df_G1 = df_table1.iloc[:-1]
df_G2 = df_table2.iloc[:-1]
df_G3 = df_table3.iloc[:-1]
df_G4 = df_table4.iloc[:-1]
df_G5 = df_table5.iloc[:-1]
df_G6 = df_table6.iloc[:-1]

# 테이블 셀 범위 정의
table_cell = {
    'table1': 'A3:G11',
    'table2': 'I3:O11',
    'table3': 'Q3:U12',
    'table4': 'W3:AA12',
    'table5': 'AC3:AK9',
    'table6': 'AW43:BE58'
}

percent_column = {
    '%table1' : [2, 3, 4, 5, 6, 7],
    '%table2' : [2, 3, 4, 5, 6, 7],
    '%table3' : [3, 5],
    '%table4' : [3, 5],
    '%table5' : [2, 3, 4, 5, 6, 7, 8, 9],
    '%table6' : [2, 3, 4, 5, 6, 7, 8, 9],
}

# 그리드 아이템 CSS <A구역 : B구역>
grid_style = {
    'display': 'inline-block',  # 인라인 블록 요소로 배치
    'width': '49%',  # 화면 절반을 차지
    'height': '110vh', #더 높은 그리드에 맞추게
    'margin': '0 auto',
    'verticalAlign': 'top',  # 상단 정렬
    # 'padding' :'2%',
    # 'paddingLeft': '100px', 
    'border': '0px solid black',  # 바더 라인 추가
    'justify-content': 'space-between',
}

# 테이블 스타일
table_style = {
    'margin': '0 auto',
    'width': '70%',
    'textAlign': 'center',
    'z-index': 3,  # 이 요소가 다른 요소 위에 위치함
}

# 그래프 형식 (Line, Bar, Dot, Pie) 목록
graph_types = ['Line', 'Bar', 'Dot', 'Pie','Heatmap','Histogram']

# 그래프 컨테이너 스타일
graph_container_style = {
    'position': 'relative',  # 상대적 위치 설정
    'margin': '0 auto',
    'width': '90%',  # 컨테이너 너비
    'z-index': 1,  # 이 요소가 다른 요소 위에 위치함
    'border': '0px solid black',  # 바더 라인 추가
    'overflow': 'hidden',  # 오버플로우 숨기기

}

# 그래프 스타일 설정
graph_style = {
    'margin': '0 auto',
    'margin-top': '-5%',
    'width': '85%',
    'z-index': 0,  # 이 요소가 다른 요소 위에 위치함
}

# 드롭다운 컨테이너 스타일
dropdown_container_style = {
    'position': 'absolute',  # 절대적 위치 설정
    'width': '20%',  # 각 드롭다운의 너비
    'right': '0%',  # 오른쪽에서 5% 떨어진 곳에 위치
    'top': '70%',  # 상단에서 50% 위치
    'transform': 'translateY(-50%)',  # Y축으로 -50% 이동하여 중앙 정렬
    'border': '0px solid black',  # 바더 라인 추가
    'z-index': 3,  # 이 요소가 다른 요소 위에 위치함
}

# 개별 드롭다운 스타일
dropdown_style = {
    'width': '100%',  # 각 드롭다운의 너비
    'marginRight': '1%',  # 드롭다운 사이의 간격
    'textAlign': 'center',
}


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
        fig.update_xaxes(tickformat="Value", tickangle=-45, tickmode='auto', nticks=12) # 여기서 nticks를 조정하여 간격 변경
        
    elif graph_type == 'Bar':
        fig = px.bar(df_table1, x=df_table1.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = px.pie(df_table1, names=df_table1.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = px.line(df_table1, x=df_table1.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

            # 그래프의 배경을 투명하게 설정
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
    )


    return fig




def update_graph_table2(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = go.Figure(data=go.Scatter(
            x=df_table2[df_table2.columns[0]].astype(str),
            y=df_table2[yaxis_column_name],
            mode='markers',  # 마커 모드 사용
            marker=dict(
                size=15,  # 데이터 마크 크기 조절 (원하는 크기로 설정)
                opacity=0.8,
                line=dict(width=2, color='DarkSlateGrey')
            )
        ))
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
        fig.update_xaxes(tickformat="Value", tickangle=-45, tickmode='auto', nticks=12) # 여기서 nticks를 조정하여 간격 변경
        
    elif graph_type == 'Bar':
        fig = px.bar(df_table2, x=df_table2.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = px.pie(df_table2, names=df_table2.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = px.line(df_table2, x=df_table2.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

    # 그래프의 배경을 투명하게 설정
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig



def update_graph_table3(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = go.Figure(data=go.Scatter(
            x=df_G3[df_G3.columns[0]].astype(str),
            y=df_G3[yaxis_column_name],
            mode='markers',  # 마커 모드 사용
            marker=dict(
                size=15,  # 데이터 마크 크기 조절 (원하는 크기로 설정)
                opacity=0.8,
                line=dict(width=2, color='DarkSlateGrey')
            )
        ))
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
        fig.update_xaxes(tickformat="Value", tickangle=-45, tickmode='auto', nticks=12) # 여기서 nticks를 조정하여 간격 변경
        
    elif graph_type == 'Bar':
        fig = px.bar(df_G3, x=df_G3.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = px.pie(df_G3, names=df_G3.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = px.line(df_G3, x=df_G3.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

    # 그래프의 배경을 투명하게 설정
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def update_graph_table4(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = go.Figure(data=go.Scatter(
            x=df_G4[df_G4.columns[0]].astype(str),
            y=df_G4[yaxis_column_name],
            mode='markers',  # 마커 모드 사용
            marker=dict(
                size=15,  # 데이터 마크 크기 조절 (원하는 크기로 설정)
                opacity=0.8,
                line=dict(width=2, color='DarkSlateGrey')
            )
        ))
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
        fig.update_xaxes(tickformat="Value", tickangle=-45, tickmode='auto', nticks=12) # 여기서 nticks를 조정하여 간격 변경
        
    elif graph_type == 'Bar':
        fig = px.bar(df_G4, x=df_G4.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = px.pie(df_G4, names=df_G4.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = px.line(df_G4, x=df_G4.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

    # 그래프의 배경을 투명하게 설정
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig



def update_graph_table5(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = go.Figure(data=go.Scatter(
            x=df_table5[df_table5.columns[0]].astype(str),
            y=df_table5[yaxis_column_name],
            mode='markers',  # 마커 모드 사용
            marker=dict(
                size=15,  # 데이터 마크 크기 조절 (원하는 크기로 설정)
                opacity=0.8,
                line=dict(width=2, color='DarkSlateGrey')
            )
        ))
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
        fig.update_xaxes(tickformat="Value", tickangle=-45, tickmode='auto', nticks=12) # 여기서 nticks를 조정하여 간격 변경
        
    elif graph_type == 'Bar':
        fig = px.bar(df_table5, x=df_table5.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = px.pie(df_table5, names=df_table5.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = px.line(df_table5, x=df_table5.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

    # 그래프의 배경을 투명하게 설정
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(domain=[0, 0.5]), #하단반쪽   cf)상단반쪽 [0.5,1]

    )

    return fig



def update_graph_table6(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = go.Figure(data=go.Scatter(
            x=df_table6[df_table6.columns[0]].astype(str),
            y=df_table6[yaxis_column_name],
            mode='markers',  # 마커 모드 사용
            marker=dict(
                size=15,  # 데이터 마크 크기 조절 (원하는 크기로 설정)
                opacity=0.8,
                line=dict(width=2, color='DarkSlateGrey')
            )
        ))
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
        fig.update_xaxes(tickformat="Value", tickangle=-45, tickmode='auto', nticks=12) # 여기서 nticks를 조정하여 간격 변경
        
    elif graph_type == 'Bar':
        fig = px.bar(df_table6, x=df_table6.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    elif graph_type == 'Pie':
        fig = px.pie(df_table6, names=df_table6.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = px.line(df_table6, x=df_table6.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

    # 그래프의 배경을 투명하게 설정
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
        height=700,
    )

    return fig

