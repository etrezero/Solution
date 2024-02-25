import dash
from dash import dcc, html
import pandas as pd
from dash.dash_table import DataTable

# 엑셀 파일 경로 및 시트 이름
path_Temp_TDF = r'C:\Users\서재영\Documents\Python Scripts\data\Temp.xlsx'
sheet_RANK = 'RANK'

# 대쉬 앱 설정
app = dash.Dash(__name__)

# 백분율로 표시할 열 딕셔너리
percentage_columns = {
    'table1': [4, 7, 10, 13, 17, 20, 23],
    'table2': [4, 7, 10, 13, 17, 20, 23],
    'table3': [4, 7, 10, 13, 17, 20, 23]
}

# 테이블 스타일 정의
table_style = {
    'width': '80%',  # 테이블 너비
    'margin': 'auto',  # 가운데 정렬
}

# 테이블 헤더 스타일 정의
header_style = {
    'backgroundColor': '#0074D9',  # 헤더 배경색
    'color': 'white',  # 헤더 텍스트 색상
    'textAlign': 'center'  # 헤더 텍스트 정렬
}

# 셀 스타일 자동 설정 함수
def generate_cell_style(df, table_number):
    cell_style = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # 숫자 데이터인 경우 오른쪽 정렬
            cell_style[col] = {'textAlign': 'right'}
            # 백분율로 변환하여 데이터 수정
            if int(col) in percentage_columns[table_number]:
                df[col] = df[col].apply(lambda x: f"{x:.1%}")
        else:
            # 문자 데이터인 경우 왼쪽 정렬
            cell_style[col] = {'textAlign': 'left'}
    return cell_style


# 데이터를 읽어와서 대쉬 테이블 생성하는 함수
def create_dash_table_from_excel(sheet_name, cell_range, table_number):
    # 엑셀 파일에서 데이터 읽기
    df = pd.read_excel(path_Temp_TDF, sheet_name=sheet_name, header=None, skiprows=2, usecols=range(2, 26), engine='openpyxl', nrows=30, dtype=str)
    # 숫자 데이터 형 변환
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    data = df.to_dict('records')  # 데이터프레임을 사전 형식으로 변환
    
    # 열 인덱스를 문자열로 변환
    df.columns = df.columns.astype(str)
    
    # 열 인덱스 이름이 빈 칸인 경우 공백으로 설정
    df.columns = df.columns.fillna(' ')
    
    # 테이블 헤더 설정
    header = [{'name': col, 'id': col} for col in df.columns]
    
    # 테이블 셀 스타일 설정
    cell_style = generate_cell_style(df, table_number)
    
    # 대쉬 테이블 생성
    return DataTable(
        id=f'table_{table_number}',
        columns=header,
        data=data,
        style_table=table_style,
        style_header=header_style,
        style_cell=cell_style,
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],  # 줄 간 간격 설정
        style_as_list_view=True,  # 리스트 뷰 스타일 적용
        fixed_rows={'headers': True},  # 헤더 고정
        virtualization=True,  # 가상화 설정
    )

# 대쉬 앱 레이아웃 설정
app.layout = html.Div([
    html.H1("대쉬 테이블"),
    
    # 테이블 1
    html.Div([
        html.H2("테이블 1"),
        create_dash_table_from_excel(sheet_RANK, 'C3:Z30', 'table1'),
    ]),
    
    # 테이블 2
    html.Div([
        html.H2("테이블 2"),
        create_dash_table_from_excel(sheet_RANK, 'C34:Z58', 'table2'),
    ]),
    
    # 테이블 3
    html.Div([
        html.H2("테이블 3"),
        create_dash_table_from_excel(sheet_RANK, 'C63:Z87', 'table3'),
    ]),
])

# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
