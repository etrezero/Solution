import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests
from dash import Dash, dcc, html, Input, Output, State
import dash
import plotly.graph_objects as go
from pathlib import Path
import yaml
import os
from openpyxl import Workbook

from dash import ctx


BASE_DIR = Path("C:/covenant")
CONFIG_PATH = BASE_DIR / "config.yaml"
SAVE_PATH = BASE_DIR / "data" / "ECOS_data.csv"
CACHE_FILE = BASE_DIR / "cache" / "cache_ECOS.json"
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

def save_excel(df, sheetname, index_option=None):
    path = rf'C:\Covenant\data\ECOS.xlsx'
    if not os.path.exists(path):
        wb = Workbook()
        wb.save(path)
        print(f"새 파일 '{path}' 생성됨.")

    try:
        if index_option is None or index_option:
            df.index = pd.to_datetime(df.index, errors='coerce')
            df.index = df.index.strftime('%Y-%m-%d')
            index = True
        else:
            index = False
    except Exception:
        print("Index를 날짜 형식으로 변환할 수 없습니다. 기본 인덱스를 사용합니다.")
        index = index_option if index_option is not None else True

    with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheetname, index=index)
        print(f"'{sheetname}' 저장 완료.")

    df.index = pd.to_datetime(df.index, errors='coerce')

def Load_API_KEY():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["ECOS_API_KEY"]

ECOS_API_KEY = Load_API_KEY()

app = Dash(__name__)
app.title = "ECOS Data Viewer"

app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <script src="https://unpkg.com/split.js/dist/split.min.js"></script>
    <style>
        .custom-gutter {
            background-color: #ccc;
            cursor: col-resize;
            width: 8px;
            height: 100%;
            z-index: 1000;
        }
        .custom-gutter:hover {
            background-color: #aaa;
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
    <script>
        function initializeSplit() {
            const left = document.querySelector('#left-pane');
            const right = document.querySelector('#right-pane');
            if (left && right) {
                Split(['#left-pane', '#right-pane'], {
                    sizes: [50, 50],
                    minSize: 200,
                    gutterSize: 8,
                    gutterAlign: 'center',
                    cursor: 'col-resize',
                    gutter: function (index, direction) {
                        const gutter = document.createElement('div');
                        gutter.className = 'custom-gutter';
                        return gutter;
                    }
                });
            } else {
                setTimeout(initializeSplit, 100);
            }
        }

        document.addEventListener('DOMContentLoaded', initializeSplit);
    </script>
</body>
</html>
'''

df_cache = pd.DataFrame()

app.layout = html.Div(
    style={'height': '100vh'},
    children=[
        html.Div(
            id='split-container',
            style={'display': 'flex', 'height': '100%'},
            children=[
                
                html.Div(
                    id='left-pane',
                    style={
                        'width': '50%', 'resize': 'horizontal', 'overflow': 'auto',
                        'minWidth': '300px', 'maxWidth': '90vw', 'font-size': '15px',
                        'padding': '20px', 'boxSizing': 'border-box', 'margin-top': '10px',
                        'borderRight': '2px solid #ccc'
                    },
                    children=[
                        html.H1("ECOS Data Viewer", style={'textAlign': 'center', 'marginBottom': '30px'}),
                        
                        html.Label("Stat Code:"),
                        dcc.Input(
                            id='stat-code', type='text', value='902Y015', placeholder='Enter stat code',
                            style={'width': '30%', 'marginBottom': '15px', 'fontSize': '15px'}
                        ),

                        html.Label("Term:"),
                        dcc.Dropdown(
                            id='term',
                            options=[{'label': t, 'value': t} for t in ['A', 'Q', 'M', 'D']],
                            value='Q',
                            style={'width': '30%', 'marginBottom': '15px'}
                        ),

                        html.Label("Start Date:"),
                        dcc.DatePickerSingle(
                            id='start-date',
                            date=(datetime.today() - relativedelta(years=5)).strftime('%Y-%m-%d'),
                            style={'width': '30%', 'marginBottom': '15px'}
                        ),

                        html.Label("End Date:"),
                        dcc.DatePickerSingle(
                            id='end-date',
                            date=datetime.today().strftime('%Y-%m-%d'),
                            style={'width': '30%', 'marginBottom': '30px'}
                        ),

                        html.Div(id='dropdown-container', style={'width': '30%', 'marginBottom': '30px'}),

                        html.Div([
                            html.Button("Stat Code 확정", id='fetch-button', n_clicks=0,
                                        style={'width': '30%', 'marginRight': '10px', 'fontSize': '14px'}),
                            html.Button("🔍 Filter Data", id='filter-button', n_clicks=0,
                                        style={'width': '30%', 'marginRight': '10px', 'fontSize': '14px'}),
                            html.Button("🔄 Reset", id='reset-dict', n_clicks=0,
                                        style={'width': '30%', 'backgroundColor': '#dc3545',
                                            'color': 'white', 'fontSize': '14px'})
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '30px'}),

                        dcc.Graph(id='data-graph')
                    ]
                ),




                html.Div(
                    id='right-pane',
                    style={
                        'width': '50%', 'padding': '20px', 'overflowY': 'auto', 'backgroundColor': '#fff'
                    },
                    children=[
                        html.H1("ECOS API Documentation", style={'textAlign': 'center'}),
                        html.Iframe(
                            src="https://ecos.bok.or.kr/api/#/DevGuide/StatisticalCodeSearch",
                            style={'width': '100%', 'height': '90vh', 'border': 'none'}
                        )
                    ]
                )
            ]
        )
    ]
)



from dash import ctx

@app.callback(
    Output('stat-code', 'value'),
    Output('term', 'value'),
    Output('start-date', 'date'),
    Output('end-date', 'date'),
    Output('dropdown-container', 'children'),
    Input('reset-dict', 'n_clicks'),
    Input('fetch-button', 'n_clicks'),
    State('stat-code', 'value'),
    State('term', 'value'),
    State('start-date', 'date'),
    State('end-date', 'date')
)
def update_inputs_or_fetch(reset_clicks, fetch_clicks, stat_code, term, start, end):
    global df_cache
    triggered_id = ctx.triggered_id

    if triggered_id == 'reset-dict':
        return (
            '902Y015',
            'Q',
            (datetime.today() - relativedelta(years=5)).strftime('%Y-%m-%d'),
            datetime.today().strftime('%Y-%m-%d'),
            []
        )

    elif triggered_id == 'fetch-button':
        if not stat_code:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, []

        start_dt = datetime.strptime(start, '%Y-%m-%d')
        end_dt = datetime.strptime(end, '%Y-%m-%d')

        valid_terms = ['A', 'Q', 'M', 'D']
        if term not in valid_terms:
            raise ValueError(f"잘못된 기간(term) 값입니다. 현재 term: {term}. 유효한 값: {valid_terms}")

        if term == 'A':
            start_date = start_dt.strftime('%Y')
            end_date = end_dt.strftime('%Y')
        elif term == 'Q':
            start_date = f"{start_dt.year}Q{(start_dt.month - 1) // 3 + 1}"
            end_date = f"{end_dt.year}Q{(end_dt.month - 1) // 3 + 1}"
        elif term == 'M':
            start_date = start_dt.strftime('%Y%m')
            end_date = end_dt.strftime('%Y%m')
        elif term == 'D':
            start_date = start_dt.strftime('%Y%m%d')
            end_date = end_dt.strftime('%Y%m%d')

        url = f"http://ecos.bok.or.kr/api/StatisticSearch/{ECOS_API_KEY}/json/kr/1/100000/{stat_code}/{term}/{start_date}/{end_date}"
        response = requests.get(url, verify=False)
        rows = response.json().get('StatisticSearch', {}).get('row', [])
        df = pd.DataFrame(rows)

        if df.empty:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, html.Div("No data fetched.")

        # ITEM_COMBO 열 생성 및 인덱스 설정
        df['ITEM_COMBO'] = df[['ITEM_NAME1', 'ITEM_NAME2', 'ITEM_NAME3', 'ITEM_NAME4']].fillna('').agg('-'.join, axis=1)
        df['DATA_VALUE'] = df['DATA_VALUE'].astype(float)
        df.set_index(['TIME', 'ITEM_COMBO'], inplace=True)
        df.sort_index(inplace=True)

        # 중복 제거
        df = df[~df.index.duplicated(keep='first')]

        df_cache = df.copy()

        save_excel(df.reset_index(), 'ECOS', index_option=False)

        dropdowns = []
        for col in ['ITEM_NAME1', 'ITEM_NAME2', 'ITEM_NAME3', 'ITEM_NAME4']:
            if col in df.reset_index().columns:
                values = sorted(df.reset_index()[col].dropna().unique())
                dropdowns.append(html.Div([
                    html.Label(col),
                    dcc.Dropdown(
                        id=f'dropdown-{col.lower()}',
                        options=[{'label': v, 'value': v} for v in values],
                        value=[],
                        multi=True
                    )
                ]))

        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dropdowns

    else:
        raise dash.exceptions.PreventUpdate







@app.callback(
    Output('data-graph', 'figure'),
    Input('filter-button', 'n_clicks'),
    State('dropdown-item_name1', 'value'),
    State('dropdown-item_name2', 'value'),
    State('dropdown-item_name3', 'value'),
    State('dropdown-item_name4', 'value')
)
def filter_and_plot(n, val1, val2, val3, val4):
    global df_cache
    if df_cache.empty:
        return go.Figure()

    df = df_cache.copy().reset_index()

    filters = []

    if 'ITEM_NAME1' in df.columns and val1:
        filters.append(df['ITEM_NAME1'].isin(val1))
    if 'ITEM_NAME2' in df.columns and val2:
        filters.append(df['ITEM_NAME2'].isin(val2))
    if 'ITEM_NAME3' in df.columns and val3:
        filters.append(df['ITEM_NAME3'].isin(val3))
    if 'ITEM_NAME4' in df.columns and val4:
        filters.append(df['ITEM_NAME4'].isin(val4))

    if filters:
        condition = filters[0]
        for f in filters[1:]:
            condition &= f
        df = df[condition]

    df['TIME'] = pd.to_datetime(df['TIME'])

    fig = go.Figure()
    for combo in df['ITEM_COMBO'].unique():
        df_line = df[df['ITEM_COMBO'] == combo]
        fig.add_trace(go.Scatter(
            x=df_line['TIME'],
            y=df_line['DATA_VALUE'],
            mode='lines+markers',
            name=combo
        ))

    fig.update_layout(
        title="Filtered ECOS Data (ITEM 조합별 라인)",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white"
    )
    return fig





if __name__ == "__main__":
    app.run_server(debug=False, host='0.0.0.0')
