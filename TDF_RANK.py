# 필요한 패키지 임포트
from dash import Dash, dcc, html, dash_table, Input, Output
from dash.dash_table.Format import Format, Scheme
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import openpyxl
import warnings
import os
from openpyxl import Workbook

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning)

# 파일 경로 및 시트 이름 정의
path_TDF = r'C:/Covenant/data/0.TDF_모니터링.xlsx'
path_Temp_TDF = r'C:/Covenant/data/Temp_TDF.xlsx'
sheet_RANK = 'RANK'
sheet_Rank_History = 'Rank_History'

# 지정된 시트를 읽어와서 데이터프레임으로 저장
df_RANK = pd.read_excel(path_TDF, sheet_name=sheet_RANK)
df_Rank_History = pd.read_excel(path_TDF, sheet_name=sheet_Rank_History)

# 파일이 없는 경우 새 Workbook 생성
if not os.path.exists(path_Temp_TDF):
    wb = Workbook()
    wb.save(path_Temp_TDF)
    print(f"새 파일 '{path_Temp_TDF}' 생성됨.")

# DataFrame을 엑셀 시트로 저장할 때, 열의 순서로 열 이름을 지정
with pd.ExcelWriter(path_Temp_TDF, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_RANK.to_excel(writer, sheet_name=sheet_RANK, index=False)
    df_Rank_History.to_excel(writer, sheet_name=sheet_Rank_History, index=False)

print(f"'{sheet_RANK}' 시트를 '{sheet_RANK}' 시트로 저장했습니다.")
print(f"'{sheet_Rank_History}' 시트를 '{sheet_Rank_History}' 시트로 저장했습니다.")


# 앱 초기화
app = Dash(__name__)


# 데이터 테이블을 생성하는 함수
def create_data_table(path, sheet_name, cell_range):
    wb = openpyxl.load_workbook(path_Temp_TDF, data_only=True)
    sheet = wb[sheet_RANK]
    data = []
    for row in sheet[cell_range]:
        data.append([cell.value for cell in row])
    df = pd.DataFrame(data[0:])

    # 백분율 소수점 첫째자리로 가운데 정렬하기 위한 Format 객체 생성
    percentage_format = Format(precision=1, scheme=Scheme.percentage)

    columns = []
    for i, col in enumerate(df.columns):
        column_config = {'name': str(col), 'id': str(col), 'type': 'numeric'}

        # 2, 5, 8, 11, 13, 17, 20, 23번째 열에 대해 백분율 소수점 첫째자리로 가운데 정렬 설정
        if i+1 in [3, 6, 9, 12, 15, 18, 21, 24]:
            column_config['format'] = percentage_format

        columns.append(column_config)

    # 모든 열에 대해 스타일을 적용합니다.
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




    # 테이블 스타일 설정
    style_table = {
        'overflowX': 'auto', 
        'marginTop': '-2%', 
        'marginBottom': '1%'  # 상하 마진 추가
    }


    # 데이터 테이블 생성
    return dash_table.DataTable(
        data=df.iloc[0:].to_dict('records'),
        columns=columns,
        page_size=18,  #행 개수 넘어가면 페이지 넘어가
        style_cell={'textAlign': 'center'},  # 여기서 가운데 정렬을 설정합니다.
        style_table=style_table,  # 테이블 스타일 설정을 적용합니다.
        style_header={
            'color': 'white',
            'fontWeight': 'bold',
            'background-color': '#3762AF',
            'display': 'none',
        },  
        style_data_conditional=style_data_conditional  # 조건부 데이터 스타일을 적용합니다.
    )



# df_Rank_History에서 최초 3행을 제거한 새로운 데이터프레임 생성
df_new = df_Rank_History.iloc[2:].reset_index(drop=True)

# 데이터의 이름 가져오기
legend = df_new.iloc[0, 1:6].tolist()  # 첫 번째 행은 그래프의 범례로 사용될 이름들을 포함합니다.

# 실제 데이터 가져오기
df_data = df_new.iloc[1:, :]  # 데이터의 첫 번째 행은 범례이므로 제외합니다.

# 선 그래프 생성
trace = []
date_column = df_data.columns[0]  # 날짜 열은 실제 데이터에서 가져옵니다.
for column, name in zip(df_data.columns[1:3], legend):  # 실제 데이터의 열과 범례 이름을 순회합니다.
    trace.append(go.Scatter(x=df_data[date_column], y=df_data[column], mode='lines', name=name))  # 그래프의 데이터를 생성합니다.


# 앱 레이아웃 설정
app.layout = html.Div([
    
    # 그래프 레이아웃 설정
    html.Div([
        dcc.Graph(
            id='rank-history-graph',
            figure={
                'data': trace,
                'layout': {
                    'title': 'YTD Rank History',
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': 'Value', 'autorange': 'reversed'},  # y축을 역축으로 설정
                    'width': '70vh',  # 가로 크기를 70%로 설정
                    'height': 'auto',  # 세로 크기를 자동으로 조정
                }
            }
        )
    ], style={
            'margin': 'auto', 
            'display': 'flex',  # 요소를 가로로 나열하기 위한 스타일
            'justifyContent': 'center', 
            # 'width': '90%',  # 가로 크기를 70%로 설정
            }),  # 그래프를 가로로 가운데로 정렬


    # 테이블 1
    html.Div([
        html.H3('YTD RANK_수익률'),
        create_data_table(path_TDF, sheet_RANK, 'C3:Z30')
    ], className='table'),

    # 테이블 2
    html.Div([
        html.H3('YTD RANK_변동성'),
        create_data_table(path_TDF, sheet_RANK, 'C34:Z58')
    ], className='table'),

    # 테이블 3
    html.Div([
        html.H3('YTD 위험대비 수익률'),
        create_data_table(path_TDF, sheet_RANK, 'C63:Z87')
    ], className='table'),



])

# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
