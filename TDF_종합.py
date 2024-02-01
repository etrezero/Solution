#240201

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash
from TDF_Dash_240131 import app as tdf_app
import TDF_Dash_240131 as tdf
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
    'zIndex': '0',      # 다른 요소 위에 오도록 z-인덱스 설정
    # 'backgroundColor': '#fff'  # 배경색 설정 (필요에 따라 변경 가능)
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


image_style = {
    'width': '70vw',  # 70% of the viewport width
    'height': 'auto',  # Auto height to maintain aspect ratio
    'position': 'fixed',  # Fixed position relative to the viewport
    'top': '50%',
    'left': '50%',
    'transform': 'translate(-50%, -50%)',
    'zIndex': '9999'  # High z-index to ensure it is on top of other elements
}




# 앱 레이아웃 정의
app.layout = html.Div([
    # 그리드 레이아웃
    html.Div(style=grid_layout_style, children=[
        
        html.Div(id='grid-1', style=grid1_style, children='ETF 포커스'),
        html.Div(id='grid-2', style=grid2_style, children='T.Rowe TDF'),
        html.Div(id='grid-3', style=grid3_style, children='멀티스크린'),
        html.Div(id='grid-4', style=grid4_style, children='FRED'),

                # Centered image on the initial screen
        html.Img(src=r'C:\Users\USER\Desktop\Excel Project\templates/DALL_Solution_Division.PNG', style=image_style),

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
    
    #초기 상태
    grid_id = ctx.triggered[0]['prop_id'].split('.')[0]
    

    #탭 1
    if grid_id == 'grid-1':
        return tdf_app.layout  # TDF 앱의 레이아웃 반환
    

    #탭 2
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
                dcc.Dropdown(
                    id='gridNumberInput',
                    options=[{'label': i, 'value': i} for i in range(1, 6)],
                    placeholder='Browse할 웹페이지를 선택하세요',
                    style={'width': '70%', 'padding': '5px'}
                ),
                dcc.Input(
                    type='text',
                    id='urlInput',
                    placeholder='URL 입력',
                    style={'width': '45%', 'height': '4%', 'padding': '10px', 'margin' : '1%'}
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










# 콜백 함수 정의
@app.callback(
    Output('line-graph-table1', 'figure'),
    [Input('yaxis-column-table1', 'value'),
     Input('graph-type-table1', 'value')]
)
def update_graph_table1(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = tdf.go.Figure(data=tdf.go.Scatter(
            x=tdf.df_table1[tdf.df_table1.columns[0]].astype(str),
            y=tdf.df_table1[yaxis_column_name],
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
        fig = tdf.px.bar(tdf.df_table1, x=tdf.df_table1.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = tdf.px.pie(tdf.df_table1, names=tdf.df_table1.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = tdf.px.line(tdf.df_table1, x=tdf.df_table1.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

            # 그래프의 배경을 투명하게 설정
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig





@app.callback(
    Output('line-graph-table2', 'figure'),
    [Input('yaxis-column-table2', 'value'),
     Input('graph-type-table2', 'value')]
)

def update_graph_table2(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = tdf.go.Figure(data=tdf.go.Scatter(
            x=tdf.df_table2[tdf.df_table2.columns[0]].astype(str),
            y=tdf.df_table2[yaxis_column_name],
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
        fig = tdf.px.bar(tdf.df_table2, x=tdf.df_table2.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = tdf.px.pie(tdf.df_table2, names=tdf.df_table2.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = tdf.px.line(tdf.df_table2, x=tdf.df_table2.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

    # 그래프의 배경을 투명하게 설정
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig



@app.callback(
    Output('line-graph-table3', 'figure'),
    [Input('yaxis-column-table3', 'value'),
     Input('graph-type-table3', 'value')]
)

def update_graph_table3(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = tdf.go.Figure(data=tdf.go.Scatter(
            x=tdf.df_G3[tdf.df_G3.columns[0]].astype(str),
            y=tdf.df_G3[yaxis_column_name],
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
        fig = tdf.px.bar(tdf.df_G3, x=tdf.df_G3.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = tdf.px.pie(tdf.df_G3, names=tdf.df_G3.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = tdf.px.line(tdf.df_G3, x=tdf.df_G3.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

    # 그래프의 배경을 투명하게 설정
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


@app.callback(
    Output('line-graph-table4', 'figure'),
    [Input('yaxis-column-table4', 'value'),
     Input('graph-type-table4', 'value')]
)

def update_graph_table4(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = tdf.go.Figure(data=tdf.go.Scatter(
            x=tdf.df_G4[tdf.df_G4.columns[0]].astype(str),
            y=tdf.df_G4[yaxis_column_name],
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
        fig = tdf.px.bar(tdf.df_G4, x=tdf.df_G4.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = tdf.px.pie(tdf.df_G4, names=tdf.df_G4.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = tdf.px.line(tdf.df_G4, x=tdf.df_G4.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

    # 그래프의 배경을 투명하게 설정
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


@app.callback(
    Output('line-graph-table5', 'figure'),
    [Input('yaxis-column-table5', 'value'),
     Input('graph-type-table5', 'value')]
)

def update_graph_table5(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = tdf.go.Figure(data=tdf.go.Scatter(
            x=tdf.df_table5[tdf.df_table5.columns[0]].astype(str),
            y=tdf.df_table5[yaxis_column_name],
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
        fig = tdf.px.bar(tdf.df_table5, x=tdf.df_table5.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = tdf.px.pie(tdf.df_table5, names=tdf.df_table5.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = tdf.px.line(tdf.df_table5, x=tdf.df_table5.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

    # 그래프의 배경을 투명하게 설정
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(domain=[0, 0.5]), #하단반쪽   cf)상단반쪽 [0.5,1]

    )

    return fig



@app.callback(
    Output('line-graph-table6', 'figure'),
    [Input('yaxis-column-table6', 'value'),
     Input('graph-type-table6', 'value')]
)
def update_graph_table6(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = tdf.go.Figure(data=tdf.go.Scatter(
            x=tdf.df_table6[tdf.df_table6.columns[0]].astype(str),
            y=tdf.df_table6[yaxis_column_name],
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
        fig = tdf.px.bar(tdf.df_table6, x=tdf.df_table6.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    elif graph_type == 'Pie':
        fig = tdf.px.pie(tdf.df_table6, names=tdf.df_table6.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = tdf.px.line(tdf.df_table6, x=tdf.df_table6.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

    # 그래프의 배경을 투명하게 설정
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
        height=700,
    )

    return fig




# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')

    #http://192.168.194.140:8050
