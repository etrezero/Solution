import pymysql
import pandas as pd
import concurrent.futures
import dash
from dash import dcc, html, Input, Output, State, dash_table
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json

# Excel 파일 경로와 JSON 파일 경로
Path_DB = r'C:\Covenant\data\0.DB_Table.xlsx'
Path_DB_json = r'C:\Covenant\data\0.DB_Table.json'

# JSON 파일을 읽어 DataFrame으로 변환
with open(Path_DB_json, 'r', encoding='utf-8') as f:
    data = json.load(f)
    df_db = pd.DataFrame(data)

df_db = df_db[['테이블한글명', '테이블영문명', '칼럼명(한글)', '칼럼명(영문)']]

# 드롭다운 목록 생성
table_options = [{'label': row['테이블한글명'], 'value': row['테이블영문명']} for index, row in df_db.iterrows()]
table_options = list({v['value']:v for v in table_options}.values())
column_options = [{'label': row['칼럼명(한글)'], 'value': row['칼럼명(영문)']} for index, row in df_db.iterrows()]

app = dash.Dash(__name__)
app.title = 'Data Base'

app.layout = html.Div([
    dcc.Dropdown(
        id='table-dropdown',
        options=table_options,
        value=table_options[0]['value'],
        style={'width': '50%', 'margin': '20px'}
    ),
    dcc.Dropdown(
        id='column-dropdown',
        multi=True,
        style={'width': '50%', 'margin': '20px'}
    ),
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=(datetime.today() - relativedelta(years=1)).strftime('%Y-%m-%d'),
        
        display_format='YYYYMMDD',
        style={'width': '50%', 'margin': '20px'}
    ),
    dcc.Dropdown(
        id='string-filter-column-dropdown',
        placeholder='문자열 조건 적용할 컬럼을 선택하세요',
        style={'width': '50%', 'margin': '20px'}
    ),
    dcc.Input(
        id='filter-input',
        type='text',
        placeholder='포함할 문자열',
        style={'width': '30%', 'margin': '20px'}
    ),
    dcc.Input(
        id='exclude-input',
        type='text',
        placeholder='제외할 문자열',
        style={'width': '30%', 'margin': '20px'}
    ),
    html.Button('쿼리 실행', id='execute-query', n_clicks=0, style={'width': '20%', 'margin': 'auto'}),
    dash_table.DataTable(
        id='db-table',
        columns=[{"name": col, "id": col} for col in df_db.columns],
        data=df_db.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px', 'minWidth': '100px', 'maxWidth': '200px', 'whiteSpace': 'normal'},
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold'
        },
        style_as_list_view=True,
    ),
    html.Div(style={'height': '40px'}),
    dash_table.DataTable(
        id='query-result-table',
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px', 'minWidth': '100px', 'maxWidth': '200px', 'whiteSpace': 'normal'},
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold'
        },
        style_as_list_view=True,
    ),
], style={'margin': '20px'})

@app.callback(
    [Output('column-dropdown', 'options'),
     Output('db-table', 'data')],
    [Input('table-dropdown', 'value')]
)
def update_columns_and_table(selected_table):
    if selected_table is None:
        return [], []

    filtered_df = df_db[df_db['테이블영문명'] == selected_table]
    column_options = [{'label': row['칼럼명(한글)'], 'value': row['칼럼명(영문)']} for index, row in filtered_df.iterrows()]
    
    return column_options, filtered_df.to_dict('records')

@app.callback(
    [Output('query-result-table', 'columns'),
     Output('query-result-table', 'data')],
    Input('execute-query', 'n_clicks'),
    [State('table-dropdown', 'value'),
     State('column-dropdown', 'value'),
     State('date-picker-range', 'start_date'),
     State('string-filter-column-dropdown', 'value'),
     State('filter-input', 'value'),
     State('exclude-input', 'value')]
)

def update_table(n_clicks, selected_table, selected_columns, start_date, filter_column, include_str, exclude_str):
    if n_clicks == 0 or not selected_table or not selected_columns:
        return [], []

    connection = pymysql.connect(
        host='192.168.195.55',
        user='solution',
        password='Solution123!',
        database='dt',
        port=3306,
        cursorclass=pymysql.cursors.DictCursor
    )

    def execute_query(connection, query):
        try:
            with connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
                return pd.DataFrame(result)
        except Exception as e:
            print(f"Error executing query: {e}")
            return None

    try:
        query = f"""
        SELECT {', '.join(selected_columns)} 
        FROM {selected_table} 
        WHERE {selected_columns[0]} >= '{start_date.replace('-', '')}'
        """
        
        if include_str:
            query += f" AND {filter_column} LIKE '%{include_str}%'"
        if exclude_str:
            query += f" AND {filter_column} NOT LIKE '%{exclude_str}%'"

        df_select = execute_query(connection, query)
        connection.close()

        if df_select is not None:
            columns = [{"name": col, "id": col} for col in df_select.columns]
            data = df_select.to_dict('records')
            
            # 명령 프롬프트에 결과 출력
            print(f"Query executed:\n{query}\n")
            print(f"Result:\n{df_select.head()}\n")

            return columns, data
        
        else:
            return [], []
        
    except Exception as e:
        print(f"Error executing query: {e}")
        connection.close()
        return [], []
    
    



@app.callback(
    Output('string-filter-column-dropdown', 'options'),
    Input('column-dropdown', 'value')
)
def update_string_filter_column_options(selected_columns):
    if not selected_columns:
        return []
    
    return [{'label': col, 'value': col} for col in selected_columns]

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
