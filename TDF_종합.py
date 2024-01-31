#240201

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash
from TDF_Dash_240131 import app as tdf_app
from flask import Flask, render_template



# 앱 초기화
app = Dash(__name__, suppress_callback_exceptions=True)

# CSS 스타일 정의
app.css.append_css({
    'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'
})

# 그리드 레이아웃 스타일 설정
grid_layout_style = {
    'display': 'grid',
    'gridTemplateColumns': 'repeat(4, 1fr)',
    'gap': '10px',
    'padding': '10px',
    'position': 'sticky',  # 여기에 추가
    'top': '0',            # 상단에 고정
    'zIndex': '1000',      # 다른 요소 위에 오도록 z-인덱스 설정
    'backgroundColor': '#fff'  # 배경색 설정 (필요에 따라 변경 가능)
}

#그리드 공통스타일
common_grid_style = {
    'padding': '20px',
    'border': '1px solid lightgrey',
    'borderRadius': '5px',
    'cursor': 'pointer',
    'textAlign': 'center',
    'color': 'white',
    'fontweight': 'bold',
}

#개별 그리드 스타일
grid1_style = {**common_grid_style, 'background-color': '#3762AF'}  # 그레이블루
grid2_style = {**common_grid_style, 'background-color': '#630'}     # 브라운
grid3_style = {**common_grid_style, 'background-color': '#4BACC6'}  # 하늘색
grid4_style = {**common_grid_style, 'background-color': '#11B1AA'}  # 에메랄드/비취색



# FRED 그래프 코드 리스트
base_url = "https://fred.stlouisfed.org/graph/graph-landing.php?g="
width_param = "&width=100%"
codes = [ "1dSVc", "1dsRu", "1dVQ6","1dFFw", "1dsBt", "1dHWE","1dHW3", 
         "1dYnv", "1dYo0", "1dYo9", "1dYos", "1dYpa", "1dYpq", "1dYpB", 
         "1dYpL", "1dYpS", "1dYpX", "1dYqi", "1dYqn", "1dYqA", "1dYqL",
         "1dYqR", "1dYqX", "1dYqZ", "1dYr4", "1dYrD", "1dNYU", "1dYsN", 
         "1dYsT", "1dYsZ", "1dYt9", "1dYtl", "1dYts", "16n5n", "1dYz8", 
         "1dYzF", "1dYBL", "1dYzi", "1dYAa", "1dYAg" , 
         ]  # FRED 코드 리스트

# FRED 그래프들을 Dash Iframe 요소로 변환
embed_iframes = []
for code in codes:
    full_src = f"{base_url}{code}{width_param}"
    iframe = html.Iframe(
        src=full_src,
        style={
            'position': 'relative',
            'width': '100%',
            'height': '550px',
            'border': '0'
        }
    )
    embed_iframes.append(iframe)


# 앱 레이아웃 정의
app.layout = html.Div([
    # 그리드 레이아웃
    html.Div(style=grid_layout_style, children=[
        html.Div(id='grid-1', style=grid1_style, children='ETF 포커스'),
        html.Div(id='grid-2', style=grid2_style, children='T.Rowe TDF'),
        html.Div(id='grid-3', style=grid3_style, children='멀티스크린'),
        html.Div(id='grid-4', style=grid4_style, children='FRED')
    ]),
    
    # 탭 컨텐츠
    html.Div(id='tab-content', className='content')
])




# 탭 3 '모니터링 메인' 콜백 로직
# iframe의 초기 URL 목록
iframe_src = [
    "https://stlouisfed.shinyapps.io/macro-snapshot/#keyIndicators",
    "https://ecos.bok.or.kr/?ref=frism.io#/",
    "https://tradingeconomics.com/countries",
    "https://www.bloomberg.com/economics",
    "https://finance.naver.com/sise/etf.naver"
]

@app.callback(
    [Output(f'iframe-container-{i}', 'children') for i in range(1, 6)],
    Input('browse-button', 'n_clicks'),
    [State('gridNumberInput', 'value'), State('urlInput', 'value')]
)
def update_iframe(n_clicks, grid_number, url):
    if n_clicks:
        if 1 <= grid_number <= 5 and url:
            iframe = html.Iframe(src=url, style={'width': '100%', 'height': '300px', 'border': '0'})
            return [iframe if i == grid_number else None for i in range(1, 6)]
    return [None for _ in range(1, 6)]



# Tab클릭 콜백 함수 정의
@app.callback(
    Output('tab-content', 'children'),
    [Input('grid-1', 'n_clicks'),
     Input('grid-2', 'n_clicks'),
     Input('grid-3', 'n_clicks'),
     Input('grid-4', 'n_clicks')]
)


def display_tab_content(grid1_clicks, grid2_clicks, grid3_clicks, grid4_clicks):
    ctx = dash.callback_context

    if not ctx.triggered:
        return ""  # 초기 상태 또는 특정 그리드 클릭이 없을 경우 비어있는 문자열 반환

    grid_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if grid_id == 'grid-1':
        return tdf_app.layout  # TDF 앱의 레이아웃 반환

    elif grid_id == 'grid-2':
        return html.Div([
            html.H3('탭 2 컨텐츠'),
            html.P('이곳에 탭 2의 내용을 작성하세요.')
        ])

    elif grid_id == 'grid-3':
        return html.Div(
        style={
            'margin': '0',
            'padding': '0',
            'display': 'grid',
            'gridTemplateRows': 'repeat(2, 1fr)',
            'gridTemplateColumns': 'repeat(3, 1fr)',
            'gap': '10px',
            'height': '100vh',
        },
        children=[
            html.Div(
                className="grid-item",
                id=f"site{i}",
                children=[
                    html.Iframe(
                        src=iframe_src[i-1],
                        style={'width': '100%', 'height': '100%', 'border': 'none'}
                    )
                ]
            ) for i in range(1, 6)
        ] + [
            html.Div(
                className="grid-item grid-info",
                children=[
                    dcc.Input(
                        type='number',
                        id='gridNumberInput',
                        placeholder='Browse할 웹페이지를 선택하세요<1-5>',
                        style={'width': '100%', 'height': '30px', 'padding': '5px'}
                    ),
                    dcc.Input(
                        type='text',
                        id='urlInput',
                        placeholder='URL 입력',
                        style={'width': '100%', 'height': '30px', 'padding': '5px'}
                    ),
                    html.Button('Browse', id='browse-button', n_clicks=0),
                ]
            )
        ]
    )
                
                
                
    
    elif grid_id == 'grid-4':
        return html.Div([
            html.H3('FRED Macro'),
            html.Div(
                [html.Div(iframe) for iframe in embed_iframes],
                style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(2, 1fr)',
                    'gridTemplateRows': 'repeat(20, 540px)',
                    'gridColumnGap': '50px',
                    'gridRowGap': '20px',
                    'margin': '10px',
                    'padding': '10px'
                }
            )
        ])





# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=True)
