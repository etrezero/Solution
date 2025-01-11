import pandas as pd
import yfinance as yf
import concurrent.futures
import requests
from pykrx import stock as pykrx
from tqdm import tqdm
from datetime import datetime, timedelta
import pickle
import os
import numpy as np
import dash
from dash import dcc, html, dash_table
from scipy.stats import skew
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from sklearn.metrics.pairwise import cosine_similarity

# KRX에서 다운로드한 CSV 파일 경로
krx_file_path = r'D:\data\KRX_list.csv'

# CSV 파일 읽기 (encoding='cp949'로 파일을 읽음)
df_KRX_list = pd.read_csv(krx_file_path, encoding='cp949')

# 종목 코드를 6자리 문자열로 변환
df_KRX_list['종목코드'] = df_KRX_list['종목코드'].astype(str)

# 시장구분에 따라 종목코드에 접미사 추가 (.KS 또는 .KQ)
df_KRX_list['종목코드'] = df_KRX_list.apply(
    lambda row: row['종목코드'] + (".KS" if row['시장구분'] == "KOSPI" else ".KQ"), axis=1
)

# 종목 코드와 이름 딕셔너리 생성
code_dict = df_KRX_list[['종목코드', '종목명', '시장구분']].set_index('종목코드').to_dict()['종목명']
code = list(code_dict.keys())

# yfinance로 데이터를 받아오는 함수
def fetch_data(code):
    try:
        if isinstance(code, int) or code.isdigit() or code.endswith(".KS") or code.endswith(".KQ"):
            if len(code) == 5:
                code = '0' + code
            krx_code = code.replace(".KS", "").replace(".KQ", "")  # pykrx는 .KS, .KQ 없이 코드만 필요
            df_price = pykrx.get_market_ohlcv_by_date(start, end, krx_code)['종가']
        else:
            session = requests.Session()
            session.verify = False
            yf_data = yf.Ticker(code, session=session)
            df_price = yf_data.history(start=start, end=end)['Adj Close']

        df_price = pd.DataFrame(df_price)
        df_price.columns = [code]
        df_price.index = pd.to_datetime(df_price.index).strftime('%Y-%m-%d')

        return df_price

    except Exception as e:
        print(f"Error fetching {code}: {e}")
        return None

# 여러 종목의 데이터를 병렬로 가져오는 함수
def Func(code_list, batch_size=10):
    all_data = []
    total_batches = len(code_list) // batch_size + (1 if len(code_list) % batch_size != 0 else 0)

    with tqdm(total=total_batches, desc="Processing batches", unit="batch") as pbar:
        for i in range(0, len(code_list), batch_size):
            batch_codes = code_list[i:i + batch_size]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(fetch_data, code): code for code in batch_codes}

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        all_data.append(result)

            pbar.update(1)  # 각 배치 처리 후 진행률 업데이트

    if all_data:
        return pd.concat(all_data, axis=1)  # 각 종목의 'Adj Close'를 열(column)로 병합
    else:
        return None

# 캐시에서 데이터를 로드하거나 새로 데이터를 받아오는 함수
def get_data_with_cache(file_path, expiry, fetch_func, *args, **kwargs):
    cache_path = r'D:\data\price_krx.pkl'
    cache_expiry = timedelta(days=1)

    if os.path.exists(file_path):
        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
        if datetime.now() - file_mtime < expiry:
            with open(file_path, 'rb') as file:
                print("Loading data from cache...")
                return pickle.load(file)

    print("Fetching new data...")
    data = fetch_func(*args, **kwargs)

    if data is not None:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Data saved to cache at {file_path}")
    else:
        print("No data fetched.")

    return data

# 데이터 가져오기 (캐시에서 로드하거나 새로운 데이터를 가져옴)
cache_path = r'D:\data\price_krx.pkl'
cache_expiry = timedelta(days=1)

start = datetime(datetime.today().year-1, 1, 1).strftime('%Y-%m-%d')
end = datetime.today().strftime('%Y-%m-%d')
df_price = get_data_with_cache(cache_path, cache_expiry, Func, code, batch_size=100)
df_price = df_price[df_price.index >= start]
df_price = df_price.bfill()


# 종목명을 데이터프레임의 열로 변환
df_price.columns = [code_dict[code] for code in df_price.columns]

df_R = df_price.pct_change().fillna(0)
df_R_percent = df_R.applymap(lambda x: "{:.2%}".format(x))
print(df_R_percent)

# 통계 데이터 계산 함수
def calculate_stats(df_R):
    stats = pd.DataFrame(index=df_R.columns)

    stats['3M RR Average'] = df_R.rolling(window=63).mean().mean()
    stats['변동성'] = df_R.std() * np.sqrt(252)
    stats['3M Max '] = df_R.rolling(window=63).mean().max()
    stats['3M min'] = df_R.rolling(window=63).mean().min()

    return stats

# 통계 데이터 계산 및 출력
df_stats = calculate_stats(df_R)
df_stats = df_stats[(df_stats != 0).any(axis=1)]
df_stats = df_stats.applymap(lambda x: "{:.2%}".format(x))
print(df_stats)

# Dash 테이블에서 열 이름을 문자열로 변환
df_stats.reset_index(inplace=True)
df_stats.rename(columns={'index': '종목명'}, inplace=True)

# 유사 종목 찾기 함수 (코사인 유사도 기반)
def find_similar_stocks(df_R, df_top_20):
    df_R_filled = df_R.fillna(0)
    df_top_20_filled = df_top_20.fillna(0)
    
    similarity_matrix = cosine_similarity(df_R_filled.T, df_top_20_filled.T)
    mean_similarity = np.mean(similarity_matrix, axis=1)
    similar_stocks = df_R_filled.columns[np.argsort(mean_similarity)[-20:]]  # 유사한 20개의 종목 선택
    return df_R_filled[similar_stocks], similar_stocks

# 누적 수익률 상위 20개 주식의 누적 수익률 그래프를 그리는 함수
def Top20(df_R, hidden_lines=[]):
    df_cumulative_returns = (1 + df_R).cumprod() - 1
    final_cumulative_returns = df_cumulative_returns.iloc[-1].sort_values(ascending=False)
    top_20_stocks = final_cumulative_returns.head(20).index
    df_top_20 = df_cumulative_returns[top_20_stocks]

    df_top_20_filtered = df_top_20.drop(columns=hidden_lines, errors='ignore')

    traces = []
    for stock in df_top_20_filtered.columns:
        trace = go.Scatter(
            x=df_top_20.index,
            y=df_top_20[stock],
            mode='lines',
            name=stock,
            opacity=0.6,
            customdata=[stock] * len(df_top_20),  # 주식명을 customdata로 전달
            hovertemplate='Stock: %{customdata}<br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        )
        traces.append(trace)

    if not df_top_20_filtered.empty:
        df_avg_returns = df_top_20_filtered.mean(axis=1)
        avg_trace = go.Scatter(
            x=df_avg_returns.index,
            y=df_avg_returns,
            mode='lines',
            name='Average',
            line=dict(color='black', width=3, dash='dash'),
            opacity=1
        )
        traces.append(avg_trace)

    return {
        'data': traces,
        'layout': go.Layout(
            title="Top 20 Stocks Cumulative Returns",
            xaxis={'title': 'Date'},
            yaxis={'title': 'Cumulative Returns', 'tickformat': '.0%'},
            legend={'x': 1, 'y': 1},
            margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
            hovermode='closest'
        )
    }


# 유사 종목들의 누적 수익률 그래프를 그리는 함수, 숨겨진 라인 고려하여 평균 계산
def SimilarStocksGraph(df_R, df_top_20, hidden_lines=[]):
    similar_stocks_df, similar_stocks_list = find_similar_stocks(df_R, df_top_20)  # 유사 종목 데이터와 리스트 분리
    df_cumulative_returns_similar = (1 + similar_stocks_df).cumprod() - 1  # 유사 종목들의 누적 수익률 계산

    # 숨겨진 종목을 제외하고 그래프 생성
    traces = []
    for stock in df_cumulative_returns_similar.columns:
        if stock not in hidden_lines:
            trace = go.Scatter(
                x=df_cumulative_returns_similar.index,
                y=df_cumulative_returns_similar[stock],
                mode='lines',
                name=stock,
                opacity=0.6,
                customdata=[stock] * len(df_cumulative_returns_similar),
                hovertemplate='Stock: %{customdata}<br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
            )
            traces.append(trace)

    # 숨겨진 종목을 제외하고 평균 수익률 추가
    if not df_cumulative_returns_similar.empty:
        df_cumulative_returns_similar_filtered = df_cumulative_returns_similar.drop(columns=hidden_lines, errors='ignore')
        if not df_cumulative_returns_similar_filtered.empty:
            df_avg_returns_similar = df_cumulative_returns_similar_filtered.mean(axis=1)
            avg_trace_similar = go.Scatter(
                x=df_avg_returns_similar.index,
                y=df_avg_returns_similar,
                mode='lines',
                name='Similar Stocks Average',
                line=dict(color='blue', width=3, dash='dot'),
                opacity=1
            )
            traces.append(avg_trace_similar)

    return {
        'data': traces,
        'layout': go.Layout(
            title="Similar Stocks Cumulative Returns",
            xaxis={'title': 'Date'},
            yaxis={'title': 'Cumulative Returns', 'tickformat': '.0%'},
            legend={'x': 1, 'y': 1},
            margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
            hovermode='closest'
        )
    }




# Dash 애플리케이션 초기화
app = dash.Dash(__name__)

# Dash 레이아웃 설정
app.layout = html.Div([
    
    
    
    
    
    html.Div(
        children=[
            
            
            
            dcc.Graph(
                id='cumulative-returns-graph',
                figure=Top20(df_R)
            ),
            dcc.Graph(
                id='similar-stocks-graph',
                figure=SimilarStocksGraph(df_R, df_price)
            ),
            dash_table.DataTable(
                id='similar-stocks-table',
                columns=[{"name": "종목명", "id": "종목명"}],
                data=[],
                page_size=10,
                style_table={'width': '20%', 'margin': 'auto'},
                style_cell={'textAlign': 'center'},
                style_header={'fontWeight': 'bold'},
                page_action='native',
            ),
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in df_stats.columns],
                data=df_stats.to_dict('records'),
                sort_action='native',
                style_table={'width': '20%', 'margin': 'auto'},
                style_cell={'textAlign': 'center'},
                style_header={'fontWeight': 'bold'},
                style_data_conditional=[{'if': {'row_index': 'odd'}}]
            )
        ],
        style={'width': '75%', 'margin': 'auto'}
    ),
    html.Div(id='hidden-lines', style={'display': 'none'}),
    html.Div(id='hidden-lines-similar', style={'display': 'none'}),
])

@app.callback(
    Output('hidden-lines', 'children'),
    [Input('cumulative-returns-graph', 'clickData')],
    [State('hidden-lines', 'children')]
)
def update_hidden_lines(clickData, hidden_lines):
    if hidden_lines is None:
        hidden_lines = []
    else:
        hidden_lines = hidden_lines.split(',')

    if clickData is not None:
        clicked_stock = clickData['points'][0]['customdata']
        if clicked_stock not in hidden_lines:
            hidden_lines.append(clicked_stock)

    return ','.join(hidden_lines)

@app.callback(
    Output('hidden-lines-similar', 'children'),
    [Input('similar-stocks-graph', 'clickData')],
    [State('hidden-lines-similar', 'children')]
)
def update_hidden_lines_similar(clickData, hidden_lines_similar):
    if hidden_lines_similar is None:
        hidden_lines_similar = []
    else:
        hidden_lines_similar = hidden_lines_similar.split(',')

    if clickData is not None:
        clicked_stock = clickData['points'][0]['customdata']
        if clicked_stock not in hidden_lines_similar:
            hidden_lines_similar.append(clicked_stock)

    return ','.join(hidden_lines_similar)

@app.callback(
    Output('cumulative-returns-graph', 'figure'),
    [Input('hidden-lines', 'children')]
)
def update_graph(hidden_lines):
    if hidden_lines is None:
        hidden_lines = []
    else:
        hidden_lines = hidden_lines.split(',')

    return Top20(df_R, hidden_lines)

@app.callback(
    Output('similar-stocks-graph', 'figure'),
    [Input('hidden-lines-similar', 'children')]
)
def update_similar_stocks_graph(hidden_lines_similar):
    if hidden_lines_similar is None:
        hidden_lines_similar = []
    else:
        hidden_lines_similar = hidden_lines_similar.split(',')

    return SimilarStocksGraph(df_R, df_price, hidden_lines_similar)

@app.callback(
    Output('similar-stocks-table', 'data'),
    [Input('cumulative-returns-graph', 'figure')]
)
def update_similar_stocks_table(figure):
    _, similar_stocks = find_similar_stocks(df_R, df_price)
    similar_stocks_df = pd.DataFrame(similar_stocks, columns=["종목명"])
    return similar_stocks_df.to_dict('records')

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0")
