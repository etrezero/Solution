#240212

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash
from TDF_메인 import app as 메인_app
import TDF_메인 as 메인
from TDF_포커스 import app as 포커스_app
import TDF_포커스 as 포커스
from TDF_TRP import app as TRP_app
import TDF_TRP as TRP
from flask import Flask
from dash.exceptions import PreventUpdate



# Flask 서버 생성
server = Flask(__name__)


# 앱 초기화
app = Dash(__name__, suppress_callback_exceptions=True, server=server)
app.title = 'Covenant Seo'  # 브라우저 탭 제목 설정


# CSS 스타일 정의
app.css.append_css({
    'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'
})


# 상단 버튼 그리드 레이아웃 스타일 설정
grid_layout_style = {
    'position': 'sticky',  # 여기에 추가
    'display': 'grid',
    'gridTemplateColumns': 'repeat(7, 1fr)',   #상단 버튼 개수
    'gap': '30px',
    'padding': '0px',
    'top': '0',            # 상단에 고정
    'zIndex': '9999',      # 다른 요소 위에 오도록 z-인덱스 설정
}


#상단 버튼 뒤 흰색 배경
grid0_style = {
    'position': 'sticky',
    'top': '0',     
    'padding': '0%',
    'background-color': 'white',
    'display': 'grid',
    'width' : '100%', 
    'zIndex': '5',  
    }  
    

#그리드 공통스타일
common_grid_style = {
    'top': '0',     
    'width' : '70%', # 상단에 고정
    'padding': '20px',
    'border': '1px solid lightgrey',
    'borderRadius': '10px',
    'cursor': 'pointer',
    'textAlign': 'center',
    'color': 'white',
    'fontSize': '20px',
    'fontWeight': 'bold',
    'boxShadow': '12px 12px 10px rgba(0, 0, 0, 0.4)', #[가로 거리] [세로 거리] [흐림 정도] [색상];
    # 'background': 'linear-gradient(45deg, #FFD700, #FFA500)', #그라데이션 배경 추가 */
    'zIndex': '0',  
}

#개별 그리드 스타일 - 상단 버튼
grid00_style = {**common_grid_style, 'background-color': '#4BACC6', 'zIndex': '0'}  # 하늘색
grid1_style = {**common_grid_style, 'background-color': '#3762AF', 'zIndex': '0'}  # 그레이블루
grid2_style = {**common_grid_style, 'background-color': '#630', 'zIndex': '0'}     # 브라운
grid3_style = {**common_grid_style, 'background-color': '#4BACC6', 'zIndex': '0'}  # 하늘색
grid4_style = {**common_grid_style, 'background-color': '#4BACC6', 'zIndex': '0'}  # 하늘색
grid5_style = {**common_grid_style, 'background-color': '#11B1AA', 'zIndex': '0'}  # 에메랄드/비취색  #FFD700 노랑
grid6_style = {**common_grid_style, 'background-color': '#11B1AA', 'zIndex': '0'}  # 에메랄드/비취색  #FFD700 노랑


img_container_style = {
    'display': 'flex',
    'justifyContent': 'center',
    'alignItems': 'center',
    'height': '100vh',
    'width': '100vw',
    'position': 'fixed',
    'top': '0',
    'left': '0',
    'zIndex': '0',
}

image_style = {
    'maxWidth': '100%',
    'maxHeight': '100vh',
    'margin': 'auto',
    'zIndex': '1',
}



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



# 앱 레이아웃 정의---------------------------------------
app.layout = html.Div([
    # 그리드 레이아웃
    html.Div(id='grid-0', style=grid0_style, children=[
        html.Div(style=grid_layout_style, children=[
            html.Div(id='grid-00', style=grid00_style, children='TDF 메인'),
            html.Div(id='grid-1', style=grid1_style, children='ETF 포커스'),
            html.Div(id='grid-2', style=grid2_style, children='T.Rowe TDF'),
            html.Div(id='grid-3', style=grid3_style, children='멀티스크린'),
            html.Div(id='grid-4', style=grid4_style, children='FRED'),
            html.Div(id='grid-5', style=grid5_style, children='S자산배분'),
            html.Div(id='grid-6', style=grid6_style, children='DB자산배분'),
        ])
    ]),
    
    # 탭 컨텐츠
    html.Div(id='tab-content', className='content')
])






# Tab클릭 콜백 함수 정의---------------------------------------
@app.callback(
    Output('tab-content', 'children'),
    [Input('grid-00', 'n_clicks'),
     Input('grid-1', 'n_clicks'),
     Input('grid-2', 'n_clicks'),
     Input('grid-3', 'n_clicks'),
     Input('grid-4', 'n_clicks'),
     Input('grid-5', 'n_clicks'),
     Input('grid-6', 'n_clicks'),
    ]
)
def display_tab_content(
    grid00_clicks, 
    grid1_clicks, 
    grid2_clicks, 
    grid3_clicks, 
    grid4_clicks, 
    grid5_clicks,
    grid6_clicks,
    ):
    ctx = dash.callback_context

    if not ctx.triggered:
        # 초기 상태에서 중앙에 이미지를 표시
        return html.Div([
            # html.Img(src='/BGR.png', style=image_style),
        ], style=img_container_style)

    grid_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # 각 탭의 레이아웃을 반환하는 함수
    def get_tab_layout(grid_id):
        if grid_id == 'grid-00':
            return 메인_app.layout
        elif grid_id == 'grid-1':
            return 포커스_app.layout
        elif grid_id == 'grid-2':
            return TRP_app.layout
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
        elif grid_id == 'grid-5':
            return html.Div([
                html.H3('S자산배분'),
                html.Div(
                    [html.Div() ],
                    style={
                        'display': 'grid',
                        'gridTemplateColumns': 'repeat(2, 1fr)',
                        'gridColumnGap': '50px',
                        'gridRowGap': '20px',
                        'margin': '10px',
                        'padding': '10px',
                        'gridboderline': '1px',
                    }
                )
            ])
        elif grid_id == 'grid-6':
            return html.Div([
                html.H3('DB자산배분'),
                html.Div(
                    [html.Div() ],
                    style={
                        'display': 'grid',
                        'gridTemplateColumns': 'repeat(2, 1fr)',
                        'gridColumnGap': '50px',
                        'gridRowGap': '20px',
                        'margin': '10px',
                        'padding': '10px',
                        'gridboderline': '1px',
                    }
                )
            ])

    # 해당하는 탭의 레이아웃을 반환
    return get_tab_layout(grid_id)











# 콜백 함수 정의
@app.callback(
    Output('line-graph-table1', 'figure'),
    [Input('yaxis-column-table1', 'value'),
     Input('graph-type-table1', 'value')]
)
def update_graph_table1(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = 포커스.go.Figure(data=포커스.go.Scatter(
            x=포커스.df_table1[포커스.df_table1.columns[0]].astype(str),
            y=포커스.df_table1[yaxis_column_name],
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
        fig = 포커스.px.bar(포커스.df_table1, x=포커스.df_table1.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = 포커스.px.pie(포커스.df_table1, names=포커스.df_table1.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = 포커스.px.line(포커스.df_table1, x=포커스.df_table1.columns[0], y=yaxis_column_name)
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
        fig = 포커스.go.Figure(data=포커스.go.Scatter(
            x=포커스.df_table2[포커스.df_table2.columns[0]].astype(str),
            y=포커스.df_table2[yaxis_column_name],
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
        fig = 포커스.px.bar(포커스.df_table2, x=포커스.df_table2.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = 포커스.px.pie(포커스.df_table2, names=포커스.df_table2.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = 포커스.px.line(포커스.df_table2, x=포커스.df_table2.columns[0], y=yaxis_column_name)
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
        fig = 포커스.go.Figure(data=포커스.go.Scatter(
            x=포커스.df_G3[포커스.df_G3.columns[0]].astype(str),
            y=포커스.df_G3[yaxis_column_name],
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
        fig = 포커스.px.bar(포커스.df_G3, x=포커스.df_G3.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = 포커스.px.pie(포커스.df_G3, names=포커스.df_G3.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = 포커스.px.line(포커스.df_G3, x=포커스.df_G3.columns[0], y=yaxis_column_name)
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
        fig = 포커스.go.Figure(data=포커스.go.Scatter(
            x=포커스.df_G4[포커스.df_G4.columns[0]].astype(str),
            y=포커스.df_G4[yaxis_column_name],
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
        fig = 포커스.px.bar(포커스.df_G4, x=포커스.df_G4.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = 포커스.px.pie(포커스.df_G4, names=포커스.df_G4.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = 포커스.px.line(포커스.df_G4, x=포커스.df_G4.columns[0], y=yaxis_column_name)
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
        fig = 포커스.go.Figure(data=포커스.go.Scatter(
            x=포커스.df_table5[포커스.df_table5.columns[0]].astype(str),
            y=포커스.df_table5[yaxis_column_name],
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
        fig = 포커스.px.bar(포커스.df_table5, x=포커스.df_table5.columns[0], y=yaxis_column_name)
    elif graph_type == 'Pie':
        fig = 포커스.px.pie(포커스.df_table5, names=포커스.df_table5.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = 포커스.px.line(포커스.df_table5, x=포커스.df_table5.columns[0], y=yaxis_column_name)
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
        fig = 포커스.go.Figure(data=포커스.go.Scatter(
            x=포커스.df_table6[포커스.df_table6.columns[0]].astype(str),
            y=포커스.df_table6[yaxis_column_name],
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
        fig = 포커스.px.bar(포커스.df_table6, x=포커스.df_table6.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    elif graph_type == 'Pie':
        fig = 포커스.px.pie(포커스.df_table6, names=포커스.df_table6.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = 포커스.px.line(포커스.df_table6, x=포커스.df_table6.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

    # 그래프의 배경을 투명하게 설정
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
        height=700,
    )

    return fig

@app.callback(
    Output('line-graph-table7', 'figure'),
    [Input('yaxis-column-table7', 'value'),
     Input('graph-type-table7', 'value')]
)
def update_graph_table7(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = 포커스.go.Figure(data=포커스.go.Scatter(
            x=포커스.df_table7[포커스.df_table7.columns[0]].astype(str),
            y=포커스.df_table7[yaxis_column_name],
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
        fig = 포커스.px.bar(포커스.df_table7, x=포커스.df_table7.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    elif graph_type == 'Pie':
        fig = 포커스.px.pie(포커스.df_table7, names=포커스.df_table7.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = 포커스.px.line(포커스.df_table7, x=포커스.df_table7.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

    # 그래프의 배경을 투명하게 설정
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
        height=700,
    )

    return fig



@app.callback(
    Output('line-graph-table8', 'figure'),
    [Input('yaxis-column-table8', 'value'),
     Input('graph-type-table8', 'value')]
)
def update_graph_table8(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = 포커스.go.Figure(data=포커스.go.Scatter(
            x=포커스.df_table8[포커스.df_table8.columns[0]].astype(str),
            y=포커스.df_table8[yaxis_column_name],
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
        fig = 포커스.px.bar(포커스.df_table8, x=포커스.df_table8.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    elif graph_type == 'Pie':
        fig = 포커스.px.pie(포커스.df_table8, names=포커스.df_table8.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = 포커스.px.line(포커스.df_table8, x=포커스.df_table8.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

    # 그래프의 배경을 투명하게 설정
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
        height=700,
    )

    return fig



@app.callback(
    Output('line-graph-table9', 'figure'),
    [Input('yaxis-column-table9', 'value'),
     Input('graph-type-table9', 'value')]
)
def update_graph_table9(yaxis_column_name, graph_type):
    if graph_type == 'Dot':
        # Dot 그래프 생성
        fig = 포커스.go.Figure(data=포커스.go.Scatter(
            x=포커스.df_table9[포커스.df_table9.columns[0]].astype(str),
            y=포커스.df_table9[yaxis_column_name],
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
        fig = 포커스.px.bar(포커스.df_table9, x=포커스.df_table9.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    elif graph_type == 'Pie':
        fig = 포커스.px.pie(포커스.df_table9, names=포커스.df_table9.columns[0], values=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정
    else:
        fig = 포커스.px.line(포커스.df_table9, x=포커스.df_table9.columns[0], y=yaxis_column_name)
        fig.update_yaxes(tickformat=".1%")  # Y 축 형식을 백분율로 설정

    # 그래프의 배경을 투명하게 설정
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
        height=700,
    )

    return fig





# 탭 3 '모니터링 메인' 콜백 로직--------------------------------
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
    [Input('browse-button', 'n_clicks')],
    [State('gridNumberInput', 'value'), State('urlInput', 'value')]
)
def update_iframe(n_clicks, grid_number, url):
    if n_clicks:
        if 1 <= grid_number <= 5 and url:
            iframe = html.Iframe(src=url, style={'width': '100%', 'height': '300px', 'border': '0'})
            return [iframe if i == grid_number else None for i in range(1, 6)]
    return [dash.no_update for _ in range(1, 6)]

@app.callback(
    Output('site1', 'children'),
    Output('site2', 'children'),
    Output('site3', 'children'),
    Output('site4', 'children'),
    Output('site5', 'children'),
    Input('browse-button', 'n_clicks'),
    Input('urlInput', 'value'),
    Input('gridNumberInput', 'value')
)
def browseURL(n_clicks, url, gridNumber):
    if n_clicks and url.strip() != '' and 1 <= gridNumber <= 5:
        # 새로운 iframe 요소 생성
        iframe = html.Iframe(src=url, style={'width': '100%', 'height': '100%', 'border': 'none'})
        
        # 입력된 그리드 번호에 따라 해당 그리드 아이템을 갱신
        return (
            iframe if gridNumber == 1 else dash.no_update,
            iframe if gridNumber == 2 else dash.no_update,
            iframe if gridNumber == 3 else dash.no_update,
            iframe if gridNumber == 4 else dash.no_update,
            iframe if gridNumber == 5 else dash.no_update,
        )
    else:
        return [dash.no_update] * 5





# Dash 앱/서버 실행
if __name__ == '__main__':
    server.run(debug=True, host='0.0.0.0')


# Dash 앱/서버 실행
# if __name__ == '__main__':
#     app.run_server(debug=True)
#     # app.run_server(debug=True, host='0.0.0.0')

    #http://192.168.194.140:8050 : 회사 - Dash
    #http://192.168.196.158:5000 : 노트북 - Flask

    # Dash : 5000
    # Flask : 8050
