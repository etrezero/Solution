import yfinance as yf
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from math import sqrt

# 엑셀 파일 경로
save_path = 'C:/Users/서재영/Documents/Python Scripts/data/0.TDF_모니터링_FN스펙트럼.xlsx'

# 오늘 날짜 설정
today = datetime.now()
today_str = today.strftime('%Y-%m-%d')
print(today_str)

# 시작 및 종료 날짜 설정
Start = today - timedelta(days=368)
End = today - timedelta(days=5)
YTD = (today.replace(month=1, day=1) - timedelta(days=1)).strftime('%Y-%m-%d')




def fetch_and_save_data(sheet_name, 종목코드_list):
    
    # Price 저장
    data_frames = []
    for 종목코드 in tqdm(종목코드_list, desc=f'Fetching data for {sheet_name}'):
        try:
            df = fdr.DataReader(종목코드, Start, End)['Close'].rename(종목코드)
            data_frames.append(df)
        except Exception as e:
            print(f"Error fetching data for 종목코드 '{종목코드}': {e}")
    combined_df = pd.concat(data_frames, axis=1)
    with pd.ExcelWriter(save_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        combined_df.to_excel(writer, sheet_name=sheet_name, index=True)
    print(f'데이터가 {save_path}의 {sheet_name} 시트에 저장되었습니다.')


    # 수익률 계산 및 저장
    Rrtn_df = calculate_Rrtn(combined_df)
    with pd.ExcelWriter(save_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        Rrtn_df.to_excel(writer, sheet_name='수익률', index=True)
    print(f'수익률 데이터가 {save_path}의 수익률 시트에 저장되었습니다.')


    # 변동성 계산 및 저장
    VOL_df = calculate_Vol_for_all(combined_df)
    with pd.ExcelWriter(save_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        VOL_df.to_excel(writer, sheet_name='변동성', index=True)
    print(f'변동성 데이터가 {save_path}의 변동성 시트에 저장되었습니다.')

# MDD 계산 및 저장
    mdd_df = calculate_mdd_for_all(combined_df)
    with pd.ExcelWriter(save_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        mdd_df.to_excel(writer, sheet_name='MDD', index=True)
    print(f'MDD 데이터가 {save_path}의 MDD 시트에 저장되었습니다.')





# 수익률 계산 수식
def calculate_Rrtn(combined_df):
    Rrtn = {}
    for name, df in combined_df.items():  # iteritems() 대신 items() 사용
        try:
            # 올해 최초 날짜 계산
            first_day_of_year = datetime(today.year, 1, 1)
            # 올해 최초 날짜 이전의 데이터프레임 인덱스 중 가장 가까운 날 찾기
            closest_date = min(filter(lambda x: x < first_day_of_year, df.index), key=lambda x: abs((x - first_day_of_year).days))

            price_YTD = df.loc[closest_date]
            current_price = df.iloc[-1]
            
            # 1개월, 3개월, 6개월, 1년 수익률 계산
            one_month_ago = today - relativedelta(months=1)
            three_month_ago = today - relativedelta(months=3)
            six_month_ago = today - relativedelta(months=6)
            one_year_ago = today - relativedelta(years=1)

            
            YTD_return = (current_price / price_YTD) - 1
            one_month_return = (current_price / df[df.index <= one_month_ago].iloc[-1]) - 1
            three_month_return = (current_price / df[df.index <= three_month_ago].iloc[-1]) - 1
            six_month_return = (current_price / df[df.index <= six_month_ago].iloc[-1]) - 1
            one_year_return = (current_price / df[df.index <= one_year_ago].iloc[-1]) - 1

            # 결과 저장
            Rrtn[name] = {
                'YTD수익률': YTD_return,
                '1개월수익률': one_month_return,
                '3개월수익률': three_month_return,
                '6개월수익률': six_month_return,
                '1년수익률': one_year_return,
            }
        except Exception as e:
            print(f"Error calculating Rrtn for {name}: {e}")

    return pd.DataFrame.from_dict(Rrtn, orient='index')


#변동성 계산 수식
def calculate_weekly_returns(df):
    # 주간 수익률 계산
    weekly_returns = df.pct_change(7)  # 7일 주기의 수익률 계산
    return weekly_returns
def calculate_Vol_for_all(combined_df):
    VOL = {}
    for name in combined_df.columns:
        weekly_returns = calculate_weekly_returns(combined_df[name])
        VOL[name] = calculate_Vol(weekly_returns)

    VOL_df = pd.DataFrame.from_dict(VOL, orient='index')
    return VOL_df
def calculate_Vol(weekly_returns_df):
    # 각 기간의 시작 인덱스를 계산합니다.
    
    one_month_ago = today - relativedelta(months=1)
    three_months_ago = today - relativedelta(months=3)
    six_months_ago = today - relativedelta(months=6)
    one_year_ago = today - relativedelta(years=1)

    # 각 기간에 대한 주간 수익률의 표준편차를 계산하고 연환산합니다.
    VOL = {
        'YTD_Vol': weekly_returns_df[YTD:].std() * sqrt(52),
        '1M_Vol': weekly_returns_df[one_month_ago:].std() * sqrt(52),
        '3M_Vol': weekly_returns_df[three_months_ago:].std() * sqrt(52),
        '6M_Vol': weekly_returns_df[six_months_ago:].std() * sqrt(52),
        '1Y_Vol': weekly_returns_df[one_year_ago:].std() * sqrt(52),
    }
    return VOL


def calculate_mdd(df):
    """
    최대 낙폭(MDD) 계산 함수
    :param df: 주가 데이터가 포함된 DataFrame
    :return: MDD 값을 반환
    """
    # 누적 최대값 계산
    cum_max = df.cummax()
    # 누적 최대값 대비 현재 가격의 하락률 계산
    drawdown = (df - cum_max) / cum_max
    # 최대 낙폭(MDD) 계산
    mdd = drawdown.min()
    return mdd

# 각 종목별 MDD 계산
def calculate_mdd_for_all(combined_df):
    mdds = {}
    for name in combined_df.columns:
        df = combined_df[name]
        mdds[name] = calculate_mdd(df)

    mdd_df = pd.DataFrame.from_dict(mdds, orient='index', columns=['MDD'])
    return mdd_df


# List1 데이터 가져오기 및 처리
df_list1 = pd.read_excel(save_path, sheet_name='List1', usecols=[0], names=['종목코드'])
종목코드_list1 = df_list1['종목코드'].astype(str).tolist()

# 주가 데이터 가져오기 및 지표 계산
price_df = fetch_and_save_data('Price1', 종목코드_list1)

