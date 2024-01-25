from dash import Dash, dcc, html, Input, Output
import dash
from TDF_Dash_240124 import app as tdf_app  # TDF_Dash_240124.py에서 app을 가져옵니다


# 앱 초기화
app = Dash(__name__)

# CSS 스타일 정의
app.css.append_css({
    'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'
})



# 탭 스타일 설정
tab_style = {
    'backgroundColor': 'lightgrey',  # 탭 버튼 배경색
    'padding': '10px',              # 탭 내부 여백
    'borderRadius': '5px',          # 탭 버튼 모서리 둥글게
    'cursor': 'pointer',            # 포인터 커서로 변경
    'fontSize': '16px',             # 글꼴 크기
    'marginRight': '10px',          # 탭 버튼 사이 간격
    'width' : '80',
    'height' : '10',
    'color': 'White',               # 탭 버튼 글꼴 색상
    'fontweight' : 'Bold',
    'bodercolor': 'grey',           # 탭 버튼 글꼴 색상
    'text-alignment' : 'center',    # 탭 버튼 텍스트 가운데정렬
    # 'display':'inline-block',       # 인라인 블록 요소로 배치
    # 'justifyContent': 'center',     # 수평 가운데 정렬
    # 'margin': '0 auto',             # 탭 버튼 수평 가운데 정렬
}

# 활성화된 탭 스타일 설정
active_tab_style = {
    'backgroundColor': 'dodgerblue',  # 활성화된 탭 버튼 배경색
    'color': 'white',                # 활성화된 탭 버튼 글꼴 색상
}

# 탭 버튼과 컨텐츠 정의
app.layout = html.Div([
    # 탭 버튼
    html.Div(className='tab', children=[
        html.Button('ETF 포커스', id='tab-1-button', style={**tab_style, 'backgroundColor': 'darkblue'}),
        html.Button('T.Rowe TDF', id='tab-2-button', style={**tab_style, 'backgroundColor': 'brown'}),
        html.Button('탭 3', id='tab-3-button', style=tab_style),
        html.Button('탭 4', id='tab-4-button', style=tab_style)
    ]),
    
    # 탭 컨텐츠
    html.Div(id='tab-content', className='content')
])

# 콜백 함수 정의
@app.callback(
    Output('tab-content', 'children'),
    [Input('tab-1-button', 'n_clicks'),
     Input('tab-2-button', 'n_clicks'),
     Input('tab-3-button', 'n_clicks'),
     Input('tab-4-button', 'n_clicks')]
)
def display_tab_content(tab1_clicks, tab2_clicks, tab3_clicks, tab4_clicks):
    if tab1_clicks is None and tab2_clicks is None and tab3_clicks is None and tab4_clicks is None:
        # 초기 페이지 설정 (탭 1)
        return html.Div(id='page1', children=[
            html.H3('   '),
            tdf_app.layout   # TDF_Dash_240124 app.layout
            
        ])
            
            
            
            
       
    
    # 탭 버튼 클릭 시 해당 탭의 컨텐츠 표시
    ctx = dash.callback_context
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'tab-1-button':
            return html.Div(id='page1', children=[
                html.H3('탭 1 컨텐츠'),
                tdf_app.layout
            ])
        elif button_id == 'tab-2-button':
            return html.Div(id='page2', children=[
                html.H3('탭 2 컨텐츠'),
                html.P('이곳에 탭 2의 내용을 작성하세요.')
            ])
        elif button_id == 'tab-3-button':
            return html.Div(id='page3', children=[
                html.H3('탭 3 컨텐츠'),
                html.P('이곳에 탭 3의 내용을 작성하세요.')
            ])
        elif button_id == 'tab-4-button':
            return html.Div(id='page4', children=[
                html.H3('탭 4 컨텐츠'),
                html.P('이곳에 탭 4의 내용을 작성하세요.')
            ])

# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=True)
