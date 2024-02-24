import FinanceDataReader as fdr
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt

# 엑셀 파일 경로
save_path = 'C:/Users/서재영/Documents/Python Scripts/data/TDF_holdings.xlsx'

# 오늘 날짜를 변수에 할당
today = datetime.now()
today_str = today.strftime('%Y-%m-%d')
print(today_str)

# 시작 날짜 설정
Start = today - timedelta(days=1000)
End = today - timedelta(days=0)

def fetch_and_save_data(sheet_name, 종목코드_list):
    data_frames = []
    for 종목코드 in tqdm(종목코드_list, desc=f'Fetching data for {sheet_name}'):
        try:
            df = fdr.DataReader(종목코드, Start, End)['Close'].rename(종목코드)
            data_frames.append(df)
        except Exception as e:
            print(f"Error fetching data for 종목코드 '{종목코드}': {e}")
    combined_df = pd.concat(data_frames, axis=1)
    
    # 인덱스를 날짜 형식으로 변경
    combined_df.index = pd.to_datetime(combined_df.index)
    # 인덱스 날짜 형식을 YY-MM-DD로 포맷팅
    combined_df.index = combined_df.index.strftime('%y-%m-%d')
    
    with pd.ExcelWriter(save_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        combined_df.to_excel(writer, sheet_name=sheet_name, index=True)  # 인덱스 포함
    print(f'데이터가 {save_path}의 {sheet_name} 시트에 저장되었습니다.')

# List1 데이터 가져오기
df_list1 = pd.read_excel(save_path, sheet_name='List1', usecols=[0], names=['종목코드'])
종목코드_list1 = df_list1['종목코드'].astype(str).tolist()
fetch_and_save_data('Price1', 종목코드_list1)

# 표준화 합산 점수 계산 함수 정의
def calculate_score(df):
    # 인덱스를 날짜 형식으로 변경
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
    
    # 3개월 rolling return 계산
    rolling_return = df.pct_change(periods=63).fillna(0)

    # 3개월 변동성 계산
    df_W_R = df.pct_change(periods=7)
    df_W_interval = df_W_R.resample('W').last()
    volatility = df_W_interval.rolling(window=4 * 3).std() * (52 ** 0.5)

    # 3개월 DD (Drawdown) 계산
    cumulative_returns = (1 + df.pct_change(fill_method=None)).cumprod()
    max_cumulative_returns = cumulative_returns.rolling(window=63).max()    
    DD = (cumulative_returns / max_cumulative_returns) - 1

    # Winning mean gap 계산
    rolling_mean = rolling_return.mean(axis=0)
    winning_mean_gap = (rolling_return - rolling_mean)

    # 3개월 skewness의 일간 변화 계산
    skewness_change = df.rolling(window=63).apply(lambda x: skew(x.pct_change().fillna(0))).fillna(0).cumsum(axis=0)

    return rolling_return, volatility, DD, winning_mean_gap, skewness_change, cumulative_returns

# Price1 시트 데이터 가져오기
price1_df = pd.read_excel(save_path, sheet_name='Price1', index_col=0)

# 인덱스를 날짜 형식으로 변경하고 "YYYY-MM-DD"로 포맷팅
price1_df.index = pd.to_datetime(price1_df.index, format='%y-%m-%d').strftime('%Y-%m-%d')

# 각 열에 대해 점수 계산
rolling_return, volatility, DD, winning_mean_gap, skewness_change, cumulative_returns = calculate_score(price1_df)

# 시트 이름 설정
sheet_names = ['rolling_return', 'volatility', 'DD', 'winning_mean_gap', 'skewness_change', 'cumulative_returns']

# 결과를 각 시트에 저장
with pd.ExcelWriter(save_path, engine='openpyxl', mode='a') as writer:
    for i, data in enumerate([rolling_return, volatility, DD, winning_mean_gap, skewness_change, cumulative_returns]):
        sheet_name = sheet_names[i]
        if sheet_name in writer.book.sheetnames:
            writer.book.remove(writer.book[sheet_name])  # 이미 있는 시트 제거
        data.to_excel(writer, sheet_name=sheet_name, index=True)

# 각 시트의 데이터를 불러옵니다.
rolling_return_df = pd.read_excel(save_path, sheet_name='rolling_return', index_col=0)
volatility_df = pd.read_excel(save_path, sheet_name='volatility', index_col=0)
DD_df = pd.read_excel(save_path, sheet_name='DD', index_col=0)
winning_mean_gap_df = pd.read_excel(save_path, sheet_name='winning_mean_gap', index_col=0)
skewness_change_df = pd.read_excel(save_path, sheet_name='skewness_change', index_col=0)
skewness_change_df = pd.read_excel(save_path, sheet_name='skewness_change', index_col=0)


# 각 데이터프레임의 열 이름과 tail을 데이터프레임으로 만듭니다.
rolling_return_tail = rolling_return_df.tail(500)
volatility_tail = volatility_df.tail(500)
DD_tail = DD_df.tail(500)
cumulative_returns_tail = cumulative_returns.tail(500)
skewness_change_tail = skewness_change_df.tail(500)

# 그래프들의 행과 열의 개수 설정
num_rows = 3
num_cols = 2

# 그래프를 그립니다.
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

# 데이터프레임과 그래프 제목을 묶어서 순회합니다.
for idx, (df, title) in enumerate([(rolling_return_tail, 'rolling_return_tail'),
                                    (volatility_tail, 'volatility_tail'),
                                    (DD_tail, 'DD_tail'),
                                    (cumulative_returns_tail, 'cumulative_returns_tail'),
                                    (skewness_change_tail, 'skewness_change_tail')]):
    # 각 그래프의 행과 열 인덱스 계산
    row_idx = idx // num_cols
    col_idx = idx % num_cols

    # 그래프를 그립니다.
    df.plot(ax=axes[row_idx, col_idx], kind='line')  # 데이터프레임을 선 그래프로 플로팅합니다.
    axes[row_idx, col_idx].set_title(f'{title} Data')  # 그래프의 제목을 설정합니다.
    axes[row_idx, col_idx].set_xlabel('Date')  # x축 레이블을 설정합니다.
    axes[row_idx, col_idx].set_ylabel('Values')  # y축 레이블을 설정합니다.
    axes[row_idx, col_idx].tick_params(axis='x', rotation=45)  # x축 눈금 라벨을 45도로 회전합니다.

# 그래프 간의 간격을 조정합니다.
plt.tight_layout()
plt.show()  # 그래프를 표시합니다.

# DD 시트의 각 열의 합계를 일자별로 계산해서 df_Sum_DD로 정의하고 그래프로 추가해줍니다.
df_Sum_DD = DD_df.sum(axis=1)

# cumulative_returns 중 열 이름이 ACWI, BND인 열의 데이터를 선택하여 그래프에 추가합니다.
cumulative_returns_ACWI_BND = cumulative_returns[['ACWI', 'BND']]

# 그래프를 추가합니다.
plt.figure(figsize=(10, 6))
df_Sum_DD.plot(kind='line', label='Sum of DD')
cumulative_returns_ACWI_BND.plot(ax=plt.gca(), kind='line', secondary_y=True)  # 보조 축에 추가합니다.
plt.title('Sum of DD Data with ACWI and BND')
plt.xlabel('Date')
plt.ylabel('Sum of DD')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

print("저장이 완료되었습니다.")
