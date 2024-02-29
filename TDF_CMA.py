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
sheet_CMA_KRW = 'CMA_KRW'
sheet_CMA_USD = 'CMA_USD'

# 지정된 시트를 읽어와서 데이터프레임으로 저장
df_CMA_KRW = pd.read_excel(path_TDF, sheet_name=sheet_CMA_KRW)
df_CMA_USD = pd.read_excel(path_TDF, sheet_name=sheet_CMA_USD)

# 파일이 없는 경우 새 Workbook 생성
if not os.path.exists(path_Temp_TDF):
    wb = Workbook()
    wb.save(path_Temp_TDF)
    print(f"새 파일 '{path_Temp_TDF}' 생성됨.")

# DataFrame을 엑셀 시트로 저장할 때, 열의 순서로 열 이름을 지정
with pd.ExcelWriter(path_Temp_TDF, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_CMA_KRW.to_excel(writer, sheet_name=sheet_CMA_KRW, index=False)
    df_CMA_USD.to_excel(writer, sheet_name=sheet_CMA_USD, index=False)

print(f"'{sheet_CMA_KRW}' 시트를 '{sheet_CMA_KRW}' 시트로 저장했습니다.")
print(f"'{sheet_CMA_USD}' 시트를 '{sheet_CMA_USD}' 시트로 저장했습니다.")


# 앱 초기화
app = Dash(__name__)


# 데이터 테이블을 생성하는 함수
def create_data_table(path, sheet_name, cell_range):
    wb = openpyxl.load_workbook(path_Temp_TDF, data_only=True)
    sheet = wb[sheet_CMA_KRW]
    data = []
    for row in sheet[cell_range]:
        data.append([cell.value for cell in row])
    df = pd.DataFrame(data[1:], columns=data[0])  # 첫 번째 행을 헤더로 사용합니다.

    # 백분율 소수점 첫째자리로 가운데 정렬하기 위한 Format 객체 생성
    percentage_format = Format(precision=1, scheme=Scheme.percentage)
    number_format = Format(precision=1, scheme=Scheme.fixed)

    columns = []
    for i, col in enumerate(df.columns):
        column_config = {'name': str(col), 'id': str(col), 'type': 'numeric'}

        # 2, 5, 8, 11, 13, 17, 20, 23번째 열에 대해 백분율 소수점 첫째자리로 가운데 정렬 설정
        if i+1 in [3, 4]:
            column_config['format'] = percentage_format
        
        elif i+1 in [5]:
            column_config['format'] = number_format

        columns.append(column_config)


    # 열의 너비 계산
        num_columns = len(df.columns)
        column_width = "{}%".format(100 / num_columns)
        
    cell_style = {
            'width': column_width,
            'minWidth': column_width,
            'maxWidth': column_width,
            'whiteSpace': 'normal',
            'textAlign': 'center',
            'verticalAlign': 'middle',
            'fontSize': '14px',
            # 'fontWeight': 'bold',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis'
        }


    # 모든 열에 대해 스타일을 적용합니다.
    style_data_conditional = [
        # {
        #     'if': {
        #         'filter_query': '{{{0}}} contains "포커스"'.format(col_name),
        #         'column_id': str(col_name)  # 문자열로 변경*중요함
        #     },
        #     'color': 'white',  # 폰트 컬러를 그레이 블루로 설정합니다.
        #     'background-color': '#3762AF',  # 폰트 컬러를 그레이 블루로 설정합니다.
        #     'fontWeight': 'bold'  # 볼드체로 설정합니다.
        # }
        # for col_name in df.columns
    ] 



    # 테이블 스타일 설정
    style_table = {
        'overflowX': 'auto', 
        'marginTop': '-2%', 
        'marginBottom': '1%', 
        'width' : '70%',
        'margin' : 'auto',
        'text-align' : 'center',

    }


    # 데이터 테이블 생성
    return dash_table.DataTable(
        data=df.iloc[0:].to_dict('records'),
        columns=columns,
        page_size=31,  #행 개수 넘어가면 페이지 넘어가
        style_cell=cell_style,  # 여기서 가운데 정렬을 설정합니다.
        style_table=style_table,  # 테이블 스타일 설정을 적용합니다.
        style_header={
            'color': 'white',
            'fontWeight': 'bold',
            'background-color': '#3762AF',
            # 'display': 'none',
        },  
        style_data_conditional=style_data_conditional  # 조건부 데이터 스타일을 적용합니다.
    )



# 그래프 데이터 생성

# df_CMA_KRW_History에서 최초 3행을 제거한 새로운 데이터프레임 생성
df_new_krw = df_CMA_KRW.iloc[2:].reset_index(drop=True)
df_new_usd = df_CMA_USD.iloc[2:].reset_index(drop=True)

# 데이터의 이름 가져오기
legend_krw = df_new_krw.iloc[0].tolist()  # 첫 번째 행은 그래프의 범례로 사용될 이름들을 포함합니다.
legend_usd = df_new_usd.iloc[0].tolist()

# 실제 데이터 가져오기
df_data_krw = df_new_krw.iloc[1:, :]  # 데이터의 첫 번째 행은 범례이므로 제외합니다.
df_data_usd = df_new_usd.iloc[1:, :]

# 데이터프레임의 해당 열을 숫자로 변환하고 NaN 값을 0으로 채웁니다.
df_data_krw[df_data_krw.columns[4]] = pd.to_numeric(df_data_krw[df_data_krw.columns[4]], errors='coerce').fillna(0)
df_data_usd[df_data_usd.columns[4]] = pd.to_numeric(df_data_usd[df_data_usd.columns[4]], errors='coerce').fillna(0)

# 데이터의 최소값과 최대값을 이용하여 표준화를 진행합니다.
min_value_krw = df_data_krw[df_data_krw.columns[4]].min()
max_value_krw = df_data_krw[df_data_krw.columns[4]].max()
min_value_usd = df_data_usd[df_data_usd.columns[4]].min()
max_value_usd = df_data_usd[df_data_usd.columns[4]].max()
df_data_krw[df_data_krw.columns[4]] = (df_data_krw[df_data_krw.columns[4]] - min_value_krw) / (max_value_krw - min_value_krw)
df_data_usd[df_data_usd.columns[4]] = (df_data_usd[df_data_usd.columns[4]] - min_value_usd) / (max_value_usd - min_value_usd)

# 버블의 크기를 설정합니다. 가장 큰 버블은 절반 크기로 설정합니다.
max_bubble_size_krw = df_data_krw[df_data_krw.columns[4]].max() * 300
max_bubble_size_usd = df_data_usd[df_data_usd.columns[4]].max() * 300
bubble_sizes_krw = df_data_krw[df_data_krw.columns[4]] * 300
bubble_sizes_usd = df_data_usd[df_data_usd.columns[4]] * 300
bubble_sizes_krw[bubble_sizes_krw == max_bubble_size_krw] = max_bubble_size_krw / 2
bubble_sizes_usd[bubble_sizes_usd == max_bubble_size_usd] = max_bubble_size_usd / 2

trace_krw = go.Scatter(
    x=df_data_krw[df_data_krw.columns[3]],  # 가로축 데이터: 데이터 테이블의 4번째 열
    y=df_data_krw[df_data_krw.columns[2]],  # 세로축 데이터: 데이터 테이블의 3번째 열
    mode='markers',
    marker=dict(
        size=bubble_sizes_krw,  # 버블의 크기: 데이터 테이블의 5번째 열의 표준화된 값으로 설정
        color='#3762AF',  # 1번 그래프 버블 색상  
    )
)

trace_usd = go.Scatter(
    x=df_data_usd[df_data_usd.columns[3]],  # 가로축 데이터: 데이터 테이블의 4번째 열
    y=df_data_usd[df_data_usd.columns[2]],  # 세로축 데이터: 데이터 테이블의 3번째 열
    mode='markers',
    marker=dict(
        size=bubble_sizes_usd,  # 버블의 크기: 데이터 테이블의 5번째 열의 표준화된 값으로 설정
        color='#630',  # 2번 그래프 버블 색상
    )
)



# 레이아웃 생성
layout = go.Layout(
    # title='자산군별 위험대비수익률',
    xaxis=dict(
        title='변동성',
        range=[None,0.35],
        tickformat='.1%',   # y축의 범위를 0부터 시작하도록 설정합니다.
    ),  # 가로축 레이블
    yaxis=dict(
        title='기대수익률',  # y축의 제목을 설정합니다.
        range=[0, None],
        tickformat='.1%',   # y축의 범위를 0부터 시작하도록 설정합니다.
    ),
    width=700,  # 그래프의 가로 크기
    height=500,  # 그래프의 세로 크기
    margin=dict(l=50, r=100, t=1, b=1),  # 마진 설정
)


# 그래프 생성
fig_krw = go.Figure(data=[trace_krw], layout=layout)
fig_usd = go.Figure(data=[trace_usd], layout=layout)

# 데이터프레임을 버블 크기 열을 기준으로 내림차순 정렬합니다.
df_sorted_krw = df_data_krw.sort_values(by=df_data_krw.columns[4], ascending=False)
df_sorted_usd = df_data_usd.sort_values(by=df_data_usd.columns[4], ascending=False)

# 상위 7개의 데이터를 추출합니다.
top_names_krw = df_sorted_krw[df_sorted_krw.columns[1]].head(7)
top_names_usd = df_sorted_usd[df_sorted_usd.columns[1]].head(7)

# 상위 7개의 이름을 그래프에 표시합니다.
text_annotations_krw = []
text_annotations_usd = []
for name in top_names_krw:
    # 텍스트 어노테이션을 생성합니다.
    annotation = go.Scatter(
        x=[df_sorted_krw[df_sorted_krw[df_sorted_krw.columns[1]] == name][df_sorted_krw.columns[3]].values[0]],
        y=[df_sorted_krw[df_sorted_krw[df_sorted_krw.columns[1]] == name][df_sorted_krw.columns[2]].values[0]],
        mode='text',
        text=name,
        showlegend=False,
        textposition='middle right',
        textfont=dict(size=10, color='black')
    )
    text_annotations_krw.append(annotation)

for name in top_names_usd:
    # 텍스트 어노테이션을 생성합니다.
    annotation = go.Scatter(
        x=[df_sorted_usd[df_sorted_usd[df_sorted_usd.columns[1]] == name][df_sorted_usd.columns[3]].values[0]],
        y=[df_sorted_usd[df_sorted_usd[df_sorted_usd.columns[1]] == name][df_sorted_usd.columns[2]].values[0]],
        mode='text',
        text=name,
        showlegend=False,
        textposition='middle right',
        textfont=dict(size=10, color='black')
    )
    text_annotations_usd.append(annotation)

# 그래프 데이터에 텍스트 어노테이션을 추가합니다.
fig_krw.add_traces(text_annotations_krw)
fig_usd.add_traces(text_annotations_usd)


# 앱 레이아웃 설정
app.layout = html.Div([
    

    # 그래프 레이아웃 설정
    html.Div([
        # 첫 번째 그래프
        html.Div([
            html.H3('2024 LTCMA(KRW))', style={'text-align': 'center'}),
            dcc.Graph(
                id='bubble-chart-krw',
                figure=fig_krw,
                style={'width': '70vh', 'height': 'auto'}  # 그래프에 스타일을 적용합니다.
            )
        ], style={'display': 'inline-block', 'margin-right': '20px'}),  # 그래프를 가로로 정렬합니다.

        # 두 번째 그래프
        html.Div([
            html.H3('2024 LTCMA(USD))', style={'text-align': 'center'}),
            dcc.Graph(
                id='bubble-chart-usd',
                figure=fig_usd,
                style={'width': '70vh', 'height': 'auto'}  # 그래프에 스타일을 적용합니다.
            )
        ], style={'display': 'inline-block'}),  # 그래프를 가로로 정렬합니다.
    ], style={
        'margin': 'auto',
        'justifyContent': 'center',
        'textAlign': 'center',
    }),

    # 테이블 1
    html.Div([
        html.H3('2024 CMA_KRW', style={'text-align': 'center'}),
        create_data_table(path_TDF, sheet_CMA_KRW, 'A10:E41')
    ], className='table'),

    # 테이블 2
    html.Div([
        html.H3('2024 CMA_USD', style={'text-align': 'center'}),
        create_data_table(path_TDF, sheet_CMA_USD, 'A10:E41')
    ], className='table'),

])


# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
