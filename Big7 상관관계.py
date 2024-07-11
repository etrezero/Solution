import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta
import dash
from dash import dcc, html, dash_table
import plotly.graph_objs as go
import concurrent.futures


# 오늘 날짜를 변수에 할당
today = datetime.now()
today_str = today.strftime('%Y-%m-%d')

# 시작 날짜 설정
Start = today - timedelta(days=365*5)
End = today - timedelta(days=1)

code = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META']

def fetch_data(code):
    try:
        df = fdr.DataReader(code, Start, End)['Close'].rename(code)
        return df
    except Exception as e:
        print(f"Error code '{code}': {e}")
        return None

# 병렬로 데이터를 가져오는 함수
def fetch_all_data(codes):
    data_frames = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_data, code): code for code in codes}
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                data_frames.append(result)
                
    # 데이터 프레임을 인덱스를 기준으로 병합
    df_price = pd.concat(data_frames, axis=1, join='outer')
    return df_price

df_price = fetch_all_data(code)
df_price = df_price.ffill()  # NaN 값을 이전에 있는 유효한 값으로 채우기

print(df_price.head())




df_R = df_price.pct_change(periods=1).fillna(0)
df_3M_R = df_price.pct_change(periods=60).fillna(0)


df_mean_R = df_R.mean(axis=1)



df_cum = (1+df_R).cumprod()-1
df_cum_mean = (1+df_mean_R).cumprod()-1


# 상관관계 계산
df_corr_matrix = df_3M_R.corr()

# 상관관계 시계열 계산 함수
def calculate_rolling_correlation(df, window=60):
    corr_dict = {}
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:
                corr_key = f"{col1} vs {col2}"
                corr_dict[corr_key] = df[col1].rolling(window).corr(df[col2])
    return pd.DataFrame(corr_dict)

df_corr = calculate_rolling_correlation(df_3M_R).dropna()

# 모든 쌍의 상관관계 평균 계산
df_corr['average'] = df_corr.mean(axis=1)
df_corr_average = df_corr['average']


#문자형식으로 소수점 4자리로 변환 후 테이블/그래프에 전달
df_corr_matrix = df_corr_matrix.applymap(lambda x: f"{x:.4f}")
df_corr = df_corr.applymap(lambda x: f"{x:.4f}")



print('df_corr_matrix========', df_corr_matrix)
print('df_corr===============', df_corr)
print('df_corr_average=======', df_corr_average)




# 대시 앱 생성
app = dash.Dash(__name__)
app.title = 'EMP 모니터링'


# 대시 앱 레이아웃 설정
app.layout = html.Div(
    style={'width': '65%', 'margin': 'auto'},
    children=[
        dcc.Graph(
            id='Big7 Correlation',
             figure={
                'data': [
                    go.Scatter(
                        x=df_corr.index, 
                        y=df_corr['average'].astype(float), 
                        mode='lines', 
                        name='Average Corr',
                        yaxis='y1',
                        line=dict(color='#3762AF', width=2),
                    ),
                    go.Scatter(
                        x=df_cum_mean.index, 
                        y=df_cum_mean, 
                        mode='lines', 
                        name='Cumulative Average Return',
                        yaxis='y2',
                        line=dict(color='#630', width=2),
                    ),
                ], 
                'layout': {
                    'title': 'Big7 Correlation and Return',
                    'xaxis': {
                        'title': 'Date', 
                        'tickformat': '%Y-%m-%d',
                        'tickmode': 'auto', 
                        'nticks': 10,
                        'textangle' : 10,
                    },
                    'yaxis': {
                        'title': 'Correlation', 
                        'tickformat': '.1f',
                        'side': 'left',
                    },
                    'yaxis2': {
                        'title': 'Average Return',
                        'overlaying': 'y',
                        'side': 'right',
                        'tickformat': '.0%',
                    },
                }
            }
        ),



        html.H3("Correlation Matrix 3M"),
        dash_table.DataTable(
            id='corr-matrix-table',
            columns=[{"name": i, "id": i} for i in df_corr_matrix.columns],
            data=df_corr_matrix.reset_index().to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'padding': '5px'},
            style_header={
                'backgroundColor': 'white',
                'fontWeight': 'bold'
            },
           
        ),


        html.H3("Correlation 3M"),
        dash_table.DataTable(
            id='corr-time-series-table',
            columns=[{"name": i, "id": i} for i in df_corr.columns],
            data=df_corr.reset_index().to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'padding': '5px'},
            style_header={
                'backgroundColor': 'white',
                'fontWeight': 'bold'
            },
          
        ),






    ]
)

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')
