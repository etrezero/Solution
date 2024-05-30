import FinanceDataReader as fdr
from pykrx import stock as pykrx
import pandas as pd
from datetime import datetime, timedelta
import dash
from dash import dcc, html
import plotly.graph_objs as go
import json
from dash import html, dcc, Input, Output
from dash import dash_table
import os
import pymysql
from dateutil.relativedelta import relativedelta
import numpy as np





j1 = r'C:\Covenant\data\j1.json'
j2 = r'C:\Covenant\data\j2.json'
j3 = r'C:\Covenant\data\j3.json'

path_모니터링 = r'C:\Covenant\data\0_모니터링_솔루션.xlsx'




# # ================================================


# def execute_query(connection, query):
#     try:
#         with connection.cursor() as cursor:
#             cursor.execute(query)
#             result = cursor.fetchall()
#             return result
#     except Exception as e:
#         print(f"Error executing query: {e}")
#         return None


# def main():
#     # MySQL 연결 정보 설정
#     connection = pymysql.connect(
#         host='',
#         user='solution',
#         password='!',
#         database='dt',
#         port=3306,  # 기본 포트는 3306입니다. 필요에 따라 수정하세요.
#         cursorclass=pymysql.cursors.DictCursor
#     )

#     # 펀드별 투자비중
#     query1 = """
#         SELECT 
#             STD_DT,               -- 기준일
#             FUND_CD,              -- 펀드코드
#             FUND_NM,              -- 펀드명
#             ITEM_CD,              -- 종목코드
#             ITEM_NM,              -- 종목명
#             NAST_TAMT_AGNST_WGH   -- 순자산비중
            
            
#         FROM 
#             DWPM10530 
#         WHERE 
#             FUND_CD IN (
#                 '07J66', '07J71', '07J76', '07J81', '07J86', '07J91', '07J96', '07M02', '07Q93',
#                 '05C16', '05K82', '05C30', '05C44', '05C58', '05C72', '05C86', '05W24', '07A77', '05W41', '07H33', '07H50','07H68', '07H85', '05C02','05N79', '05N43', '05N61', '05N25',
#                 '3JM08', '3JM09', '3JM10', '3JM11', '3JM12', '3JM13', '4JM03', '4JM04',
#                 '2JM57', '3JM97', '06K04',
#                 '03V94', '03V95', '05H80', '05H73', '05W58', 
#                 '2JM14', '05Q62', '06F77', '2JM08', '2JM66'
#                 )
#             AND STD_DT >= '20231231';
#     """
#     result1 = execute_query(connection, query1)
#     if result1 is not None:
#         # 결과를 데이터프레임으로 변환
#         df1 = pd.DataFrame(result1)

#         if os.path.exists(j1):
#             os.remove(j1)
       
#         df1.to_json(j1, orient='records', index=False)
#         print("Query1 : df1 ==========================", df1.head())


#     # 수정기준가 / 설정액 / 순자산
#     query2 = """
#         SELECT 
#             STD_DT,          -- 기준일
#             FUND_CD,         -- 펀드코드
#             MOD_STPR,        -- 수정기준가
#             OPNG_AMT,        -- 설정금액 
#             TDD_OPNG_AMT,    -- 당일설정
#             TDD_CLSR_AMT,    -- 당일해지
#             NAST_AMT         -- 순자산

#         FROM 
#             DWPM10510 
#         WHERE 
#             FUND_CD IN (
#                 '07J66', '07J71', '07J76', '07J81', '07J86', '07J91', '07J96', '07M02', '07Q93',
#                 '05C16', '05K82', '05C30', '05C44', '05C58', '05C72', '05C86', '05W24', '07A77', '05W41', '07H33', '07H50','07H68', '07H85', '05C02','05N79', '05N43', '05N61', '05N25',
#                 '3JM08', '3JM09', '3JM10', '3JM11', '3JM12', '3JM13', '4JM03', '4JM04',
#                 '2JM57', '3JM97', '06K04',
#                 '03V94', '03V95', '05H80', '05H73', '05W58', 
#                 '2JM14', '05Q62', '06F77', '2JM08', '2JM66'
#                 )
#             AND STD_DT >= '20231231';
#     """

#     result2 = execute_query(connection, query2)
#     if result2 is not None:
#         # 결과를 데이터프레임으로 변환
#         df2 = pd.DataFrame(result2)

#         if os.path.exists(j2):
#             os.remove(j2)

#         df2.to_json(j2, orient='records', index=False)
#         print("Query2 : df2  =======================", df2.head())





# # 판매회사별 / 설정액
#     query3 = """
#         SELECT 
#             TR_YMD,          -- 기준일자
#             UNYONG_NM,       -- 운용사명
#             FUND_NM,        -- 펀드명
#             PM_NM,          -- 판매사명
#             AMT             -- 설정금액 
           

#         FROM 
#             ST_KITCA_DS 
#         WHERE 
#             TR_YMD >= '20231231'
#             AND FUND_NM LIKE '%TDF%' OR FUND_NM LIKE '%TIF%';
#     """

#     result3 = execute_query(connection, query3)
#     if result3 is not None:
#         # 결과를 데이터프레임으로 변환
#         df3 = pd.DataFrame(result3)

#         if os.path.exists(j3):
#             os.remove(j3)

#         df3.to_json(j3, orient='records', index=False)
#         print("Query3 : df3  =======================", df3.head())




#     # 연결 종료
#     connection.close()

# if __name__ == "__main__":
#     main()


# # ===================================================================



df1 = pd.read_json(j1)     #투자비중
df2 = pd.read_json(j2)     #수정기준가
df3 = pd.read_json(j3)     #판매사별


# 첫 번째 열을 datetime으로 변환하고, 날짜 형식으로 포맷팅
df1['STD_DT'] = pd.to_datetime(df1['STD_DT'].astype(str), format='%Y%m%d')
df2['STD_DT'] = pd.to_datetime(df2['STD_DT'].astype(str), format='%Y%m%d')
df3['TR_YMD'] = pd.to_datetime(df3['TR_YMD'].astype(str), format='%Y%m%d')



start = df1['STD_DT'].iloc[0]
T0 = df1['STD_DT'].iloc[-1]
T0 = pd.to_datetime(T0, format='%Y%m%d')


print("start", start)
print("T0", T0)




# ITEM_CD와 ITEM_NM을 딕셔너리로 만듭니다.
item_dict = pd.Series(df1['ITEM_NM'].values, index=df1['ITEM_CD']).to_dict()

# 딕셔너리를 데이터프레임으로 변환
item_df = pd.DataFrame(list(item_dict.items()), columns=['ITEM_CD', 'ITEM_NM'])


# 변환 함수 정의
def change_code(item_code):
    if item_code.startswith('KR7'):
        return item_code[3:9] + '.KS' # KR7 다음 6자리 지정
    elif item_code.startswith('US') and not item_code.startswith('USM'):
        return 'US ETF'  # US로 시작하고 USM으로 시작하지 않으면 'US ETF'로 변환
    
    elif item_code.startswith('LU'):
        return None  # LU로 시작하면 'SICAV'로 변환
    return None  # 그 외의 경우 None 반환하여 삭제 대상임을 표시

# ITEM_CD를 변환하여 새로운 열에 저장
item_df['Trans_CODE'] = item_df['ITEM_CD'].apply(change_code)

# 변환되지 않은 나머지 행을 삭제
item_df = item_df.dropna(subset=['Trans_CODE'])



df_유니버스 = pd.read_excel(path_모니터링, sheet_name='유니버스')

# item_df에서 Trans_CODE가 'US ETF'인 행에 대해 처리
for index, row in item_df.iterrows():
    if row['Trans_CODE'] == 'US ETF':
        # df_유니버스에서 일치하는 name을 찾아 Ticker로 대체
        matching_ticker = df_유니버스.loc[df_유니버스['Name'] == row['ITEM_NM'], 'Ticker']
        item_df['Trans_CODE'] = item_df['Trans_CODE'].apply(lambda x: x.replace(" US Equity", "") if isinstance(x, str) else "")
        if not matching_ticker.empty:
            item_df.at[index, 'Trans_CODE'] = matching_ticker.values[0]








# 호출 예시: 시작과 종료 날짜 정의
code2 = ['USD/KRW']

code = list(item_df['Trans_CODE']) + code2
start = start.strftime('%Y%m%d')
end = T0.strftime('%Y%m%d')


def FDR(codes, start, end):
    data_frames = []
    for code in codes:
        try:
            # 한국 ETF 데이터 불러오기 시도
            if ".KS" in code:
                ETF_price = pykrx.get_etf_ohlcv_by_date(start, end, code.replace(".KS", ""))
                if '종가' in ETF_price.columns:
                    ETF_price = ETF_price['종가'].rename(code)
                else:
                    raise ValueError(f"{code}: '종가' column not found in pykrx data.")
            else:
                # FinanceDataReader를 사용하여 다른 국가의 ETF 또는 주식 데이터 불러오기
                ETF_price = fdr.DataReader(code, start, end)['Close'].rename(code)

            data_frames.append(ETF_price)
        except Exception as e:
            print(f"Error fetching data for {code}: {e}")

    return pd.concat(data_frames, axis=1) if data_frames else pd.DataFrame()

#======================================================
ETF_price = FDR(code, start, end)
#======================================================



# 데이터가 없는 날짜를 처리하기 위해 날짜를 조정하는 함수 
def get_last_date(df, target_date):
    while target_date not in df.index:
        target_date  -= timedelta(days=1)
    return target_date 

# T0 날짜 조정
last = get_last_date(ETF_price, T0)
print("last", last)



ETF_price = ETF_price.ffill()  # NaN 값을 이전에 있는 유효한 값으로 채우기
ETF_R = ETF_price.pct_change(periods=1).fillna(0)  # 일별 수익률 계산

# print('ETF_price====================',  ETF_price.head())



# df1에 Trans_CODE 열 추가(ITEM_NM과 Trans_CODE 매핑 생성)
df1['Trans_CODE'] = df1['ITEM_NM'].map(item_df.set_index('ITEM_NM')['Trans_CODE'].to_dict())

PV_W = df1.pivot_table(index='STD_DT', columns=['FUND_CD', 'Trans_CODE'] , values='NAST_TAMT_AGNST_WGH', aggfunc='sum', fill_value=0, margins=False)
PV_W['합계'] = PV_W.sum(axis=1)   #각 행의 합계 구하기
PV_W = (PV_W/100).round(2)
PV_W = PV_W.rename_axis('구분')     #축 이름을 구분으로 지정하기

W_3JM13 = PV_W.loc[:,'3JM13']
W_2JM57 = PV_W.loc[:,'2JM57']
W_3JM97 = PV_W.loc[:,'3JM97']
W_07J76 = PV_W.loc[:,'07J76']





PV_수정기준가 = df2.pivot_table(index='STD_DT', columns=['FUND_CD',] , values='MOD_STPR', aggfunc='sum', fill_value=0, margins=False)
PV_수정기준가 = PV_수정기준가.rename_axis('구분')     #축 컬럼 이름을 구분으로 지정하기
# 기준가_07J96 = PV_수정기준가.loc[:,'07J96']



PV_설정액 = df2.pivot_table(index='STD_DT', columns=['FUND_CD',] , values='OPNG_AMT', aggfunc='sum', fill_value=0, margins=False).rename_axis('구분')
PV_순자산 = df2.pivot_table(index='STD_DT', columns=['FUND_CD',] , values='NAST_AMT', aggfunc='sum', fill_value=0, margins=False).rename_axis('구분')





T_W = T0 - timedelta(weeks=1)
T_M = T0 - relativedelta(months=1)
T_3M = T0 - relativedelta(months=3)
T_6M = T0 - relativedelta(months=6)
T_1Y = T0 - relativedelta(years=1)
T_YTD = datetime(T0.year-1, 12, 31)

# print("기간별 날짜", T0, T_W, T_M, T_3M, T_6M, T_1Y, T_YTD)


# 각 시점별 펀드 수익률 계산=========================================
펀드_R = PV_수정기준가.pct_change(periods=1).fillna(0)
R_W = PV_수정기준가.loc[T0]/PV_수정기준가.loc[T_W]-1 if T_W in PV_수정기준가.index else 0
R_M = PV_수정기준가.loc[T0]/PV_수정기준가.loc[T_M]-1 if T_M in PV_수정기준가.index else 0
R_3M = PV_수정기준가.loc[T0]/PV_수정기준가.loc[T_3M]-1 if T_3M in PV_수정기준가.index else 0
R_6M = PV_수정기준가.loc[T0]/PV_수정기준가.loc[T_6M]-1 if T_6M in PV_수정기준가.index else 0
R_1Y = PV_수정기준가.loc[T0]/PV_수정기준가.loc[T_1Y]-1 if T_1Y in PV_수정기준가.index else 0
R_YTD = PV_수정기준가.loc[T0]/PV_수정기준가.loc[T_YTD]-1 if T_YTD in PV_수정기준가.index else 0


주간수익률 = PV_수정기준가.pct_change(periods=7)
interval_7 = 주간수익률.iloc[::7] # 7일 간격으로 데이터 추출

Vol_1M = interval_7.loc[T_M:].std() * np.sqrt(52)  # T_M 이후 데이터
Vol_3M = interval_7.loc[T_3M:].std() * np.sqrt(52)  # T_3M 이후 데이터
Vol_6M = interval_7.loc[T_6M:].std() * np.sqrt(52)  # T_6M 이후 데이터
Vol_1Y = interval_7.loc[T_1Y:].std() * np.sqrt(52)  # T_1Y 이후 데이터
Vol_YTD = interval_7.loc[T_YTD:].std() * np.sqrt(52)  # T_YTD 이후 데이터

cum = (1+펀드_R).cumprod()-1

# =======================================================================
# print(R_YTD.head)




# 각 시점별 ETF 수익률 계산=========================================

ETF_R = ETF_R
ETF_R_W = ETF_price.loc[last]/ETF_price.loc[T_W]-1 if T_W in ETF_price.index else 0
ETF_R_M = ETF_price.loc[last]/ETF_price.loc[T_M]-1 if T_M in ETF_price.index else 0
ETF_R_3M = ETF_price.loc[last]/ETF_price.loc[T_3M]-1 if T_3M in ETF_price.index else 0
ETF_R_6M = ETF_price.loc[last]/ETF_price.loc[T_6M]-1 if T_6M in ETF_price.index else 0
ETF_R_1Y = ETF_price.loc[last]/ETF_price.loc[T_1Y]-1 if T_1Y in ETF_price.index else 0
ETF_R_YTD = ETF_price.loc[last]/ETF_price.loc[T_YTD]-1 if T_YTD in ETF_price.index else 0


ETF_주간수익률 = ETF_price.pct_change(periods=7)
ETF_interval_7 = ETF_주간수익률.iloc[::7] # 7일 간격으로 데이터 추출

ETF_Vol_1M = interval_7.loc[T_M:].std() * np.sqrt(52)  # T_M 이후 데이터
ETF_Vol_3M = interval_7.loc[T_3M:].std() * np.sqrt(52)  # T_3M 이후 데이터
ETF_Vol_6M = interval_7.loc[T_6M:].std() * np.sqrt(52)  # T_6M 이후 데이터
ETF_Vol_1Y = interval_7.loc[T_1Y:].std() * np.sqrt(52)  # T_1Y 이후 데이터
ETF_Vol_YTD = interval_7.loc[T_YTD:].std() * np.sqrt(52)  # T_YTD 이후 데이터

ETF_cum = (1+ETF_R).cumprod()-1

# 중복되지 않은 열만 유지
ETF_cum = ETF_cum.loc[:, ~ETF_cum.columns.duplicated()]

# print(ETF_cum.head())
# =======================================================================











ctr = {}
# 모든 FUND_CD에 대해 교집합 컬럼 계산
for fund_cd in PV_W.columns.get_level_values('FUND_CD').unique():
    fund_columns = PV_W.xs(fund_cd, level='FUND_CD', axis=1).columns
    common_columns = fund_columns.intersection(ETF_R.columns)
    # print(f'Common columns for FUND_CD {fund_cd}:', common_columns)
    
    # 이제 common_columns를 사용하여 특정 연산을 수행할 수 있습니다.
    # 예를 들어 PV_W와 ETF_R 사이의 연산:
    if common_columns.empty:
        continue  # 공통 컬럼이 없으면 다음 펀드 코드로 넘어감

    weight_data = PV_W.xs(fund_cd, level='FUND_CD', axis=1)[common_columns]
    etf_data = ETF_R[common_columns]
    ctr_ETF = weight_data * etf_data

    # 연산 결과는 df 딕셔너리에 ctr_{FUND_CD} 형태의 키로 저장됩니다.
    ctr[fund_cd] = ctr_ETF


# # ctr 딕셔너리에 저장된 각 FUND_CD의 데이터프레임 상태 확인
# for key in ctr.keys():
#     print(f"{key}===================",ctr[key].head())



# # '3JM13'이 ctr에 존재하는지 확인하고 그 결과를 Excel로 저장
if '03V95' or '3JM13' or '07J76' or '2JM57' in ctr:
    ctr_03V95 = ctr['03V95']
    cum_03V95 = ctr_03V95.cumsum().ffill()  # 각 컬럼의 누적 합계 계산

    ctr_3JM13 = ctr['3JM13']
    cum_3JM13 = ctr_3JM13.cumsum().ffill()  # 각 컬럼의 누적 합계 계산

    ctr_07J76 = ctr['07J76']
    cum_07J76 = ctr_07J76.cumsum().ffill()  # 각 컬럼의 누적 합계 계산

    ctr_2JM57 = ctr['2JM57']
    cum_2JM57 = ctr_2JM57.cumsum().ffill()  # 각 컬럼의 누적 합계 계산

    # # Excel 파일로 저장
    # with pd.ExcelWriter(path_모니터링, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    #     ctr_03V95.to_excel(writer, sheet_name='ctr_03V95', index=True, merge_cells=False)
    #     cum_03V95.to_excel(writer, sheet_name='cum_03V95', index=True, merge_cells=False)
else:
    print("'03V95'이 ctr 딕셔너리에 없습니다.")

print('저장완료')




BM_R = ETF_R['ACWI']*0.5 + ETF_R['356540.KS']*0.5
BM_cum = (1+BM_R).cumprod()-1
BM_cum = BM_cum.reindex(cum.index).ffill()




# ==================================================================


Focus = [
        "07M02", "07J66", "07J71", "07J76", 
        "07J81", "07J86", "07J91", "07J96",
         ]

TRP = [
        "05C16", "05C30", "05C44", "05C58", 
        "05C72", "05C86", "05W24", "05W41", 
        "07H33", "07H50", "07H68", "07H85", 
        "05C02", "05N61", "05N25",
        ]

TRP_빈티지 = [
        '05N25', '05C44', '05C58', '05C72', 
        '05C86', '05W41', '07H50', '07H85', 
        ]


S자산배분 = [
        "3JM08", "3JM09", "3JM10", "3JM11", 
        "3JM12", "3JM13", "4JM03", "4JM04",
    ]


# 리스트에서 중복 제거
Best_10 = list(set(ETF_cum.loc[ETF_cum.last_valid_index()].nlargest(10).index))
Worst_10 = list(set(ETF_cum.loc[ETF_cum.last_valid_index()].nsmallest(10).index))



#각 그룹별로 설정액 합계 데이터프레임 만들기

def sum_df(df, fund_codes):
    # 주어진 코드에 대한 열만 선택
    select_columns = df.columns.intersection(fund_codes)
    # 선택된 열들의 합계를 계산
    sum_df = df[select_columns].sum(axis=1)
    return sum_df

# 각 그룹별로 데이터프레임 생성
설정액_Focus = sum_df(PV_설정액, Focus)
설정액_TRP = sum_df(PV_설정액, TRP)
설정액_S자산배분 = sum_df(PV_설정액, S자산배분)




#TDF 빈티지별 YTD 수익률
R_YTD_Focus = R_YTD.loc[R_YTD.index.isin(Focus) ]
R_YTD_TRP = R_YTD.loc[R_YTD.index.isin(TRP_빈티지) ]

vintage_map = {
    "TIF": ["07M02", "05N25"],
    "2030": ["07J66", "05C44"],
    "2035": ["07J71", "05C58"],
    "2040": ["07J76", "05C72"],
    "2045": ["07J81", "05C86"],
    "2050": ["07J86", "05W41"],
    "2055": ["07J91", "07H50"],
    "2060": ["07J96", "07H85"]
}


# vintage_mapping의 키와 값을 바꿉니다.
invert = {index: key for key, values in vintage_map.items() for index in values}



# 새로운 인덱스를 적용하기 위해 DataFrame을 재생성
R_YTD_Focus = R_YTD_Focus.rename(index=invert)
R_YTD_TRP = R_YTD_TRP.rename(index=invert)

# print(R_YTD_Focus, R_YTD_TRP)






#TDF 판매사별 설정액 ================================================


#빈티지 열을 추가하여 빈티지 표시
df3.loc[df3['FUND_NM'].str.contains('TIF'), '빈티지'] = 'TIF'
df3.loc[df3['FUND_NM'].str.contains('2030'), '빈티지'] = '2030'
df3.loc[df3['FUND_NM'].str.contains('2035'), '빈티지'] = '2035'
df3.loc[df3['FUND_NM'].str.contains('2040'), '빈티지'] = '2040'
df3.loc[df3['FUND_NM'].str.contains('2045'), '빈티지'] = '2045'
df3.loc[df3['FUND_NM'].str.contains('2050'), '빈티지'] = '2050'
df3.loc[df3['FUND_NM'].str.contains('2055'), '빈티지'] = '2055'
df3.loc[df3['FUND_NM'].str.contains('2060'), '빈티지'] = '2060'
df3.loc[~df3['FUND_NM'].str.contains('2030|2035|2040|2045|2050|2055|2060|TIF'), '빈티지'] = '기타'

#디폴트 열을 추가하여 O표시
df3.loc[df3['FUND_NM'].str.contains('O'), '디폴트'] = 'O'



start = datetime.strptime(start, '%Y%m%d')

설정액추이_운용사별 = df3.pivot_table(index='TR_YMD', columns='UNYONG_NM', values='AMT', aggfunc='sum', fill_value=0, margins=True)
설정액추이_운용사별 = 설정액추이_운용사별[설정액추이_운용사별.index != 'All']
설정액추이_운용사별 = 설정액추이_운용사별.loc[설정액추이_운용사별.index >= start]

설정액추이_판매사별 = df3.pivot_table(index='TR_YMD', columns='PM_NM', values='AMT', aggfunc='sum', fill_value=0, margins=True)
설정액추이_판매사별 = 설정액추이_판매사별[설정액추이_판매사별.index != 'All']
설정액추이_판매사별 = 설정액추이_판매사별.loc[설정액추이_판매사별.index >= start]


설정액추이_빈티지별 = df3.pivot_table(index='TR_YMD', columns='빈티지', values='AMT', aggfunc='sum', fill_value=0, margins=True)
설정액추이_빈티지별 = 설정액추이_빈티지별[설정액추이_빈티지별.index != 'All']
설정액추이_빈티지별 = 설정액추이_빈티지별.loc[설정액추이_빈티지별.index >= start]


디폴트 = df3.loc[df3['디폴트'] == 'O']
디폴트추이_운용사별 = 디폴트.pivot_table(index='TR_YMD', columns='UNYONG_NM', values='AMT', aggfunc='sum', fill_value=0, margins=True)
디폴트추이_운용사별 = 디폴트추이_운용사별[디폴트추이_운용사별.index != 'All']
디폴트추이_운용사별 = 디폴트추이_운용사별.loc[디폴트추이_운용사별.index >= start]

디폴트 = df3.loc[df3['디폴트'] == 'O']
디폴트추이_판매사별 = 디폴트.pivot_table(index='TR_YMD', columns='PM_NM', values='AMT', aggfunc='sum', fill_value=0, margins=True)
디폴트추이_판매사별 = 디폴트추이_판매사별[디폴트추이_판매사별.index != 'All']
디폴트추이_판매사별 = 디폴트추이_판매사별.loc[디폴트추이_판매사별.index >= start]


# # 엑셀저장 ========================================================

# print(df3.head)

# # with pd.ExcelWriter(path_모니터링, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
#     # df1.to_excel(writer, sheet_name='df1', index=False, merge_cells=False)
#     # df2.to_excel(writer, sheet_name='df2', index=False, merge_cells=False)
#     # df3.to_excel(writer, sheet_name='df3', index=False, merge_cells=False)
# # #     item_df.to_excel(writer, sheet_name='item_df', index=False, merge_cells=False)

# #     PV_W.to_excel(writer, sheet_name='PV_W', index=True, merge_cells=False)
# # #     W_3JM13.to_excel(writer, sheet_name='W_3JM13', index=True, merge_cells=False)
    
# #     PV_수정기준가.to_excel(writer, sheet_name='수정기준가', index=True, merge_cells=False)
# # #     펀드_R.to_excel(writer, sheet_name='펀드_R', index=True, merge_cells=False)
# # #     interval_7.to_excel(writer, sheet_name='interval_7', index=True, merge_cells=False)
# #     R_M.to_excel(writer, sheet_name='R_M', index=True, merge_cells=False)
# #     R_3M.to_excel(writer, sheet_name='R_3M', index=True, merge_cells=False)
# #     # R_6M.to_excel(writer, sheet_name='R_6M', index=True, merge_cells=False)
# #     # R_1Y.to_excel(writer, sheet_name='R_1Y', index=True, merge_cells=False)
# #     R_YTD.to_excel(writer, sheet_name='R_YTD', index=True, merge_cells=False)
# # #     Vol_3M.to_excel(writer, sheet_name='Vol_3M', index=True, merge_cells=False)
# # #     cum.to_excel(writer, sheet_name='cum', index=True, merge_cells=False)

# #     PV_설정액.to_excel(writer, sheet_name='설정액', index=True, merge_cells=False)
# #     PV_순자산.to_excel(writer, sheet_name='순자산', index=True, merge_cells=False)

# #     ETF_price.to_excel(writer, sheet_name='ETF_price', index=True, merge_cells=False)
# # #     ETF_R.to_excel(writer, sheet_name='ETF_R', index=True, merge_cells=False)
# # #     ETF_cum.to_excel(writer, sheet_name='ETF_cum', index=True, merge_cells=False)
# print("엑셀저장 완료")













# #==================================================================


# 대시 앱 생성
app = dash.Dash(__name__)


# 대시 앱 레이아웃 설정
app.layout = html.Div(
    
    style={'width': '50%', 'margin': 'auto'},
    children=[

        html.H3("모니터링_솔루션", style={'textAlign': 'center'}),
        # dash_table.DataTable(
        #     id='ytd-table',
        #     columns=[{'name': 'Index', 'id': 'index'}] + [{'name': col, 'id': col} for col in df_YTD_R.columns],
        #     data=df_YTD_R.reset_index().to_dict('records'),
        #     style_data={'whiteSpace': 'normal', 'height': 'auto', 'width': '100px'},  # 셀 가로 크기 설정
        #     style_cell={'textAlign': 'center', 'minWidth': '100px', 'width': '100px', 'maxWidth': '100px', 'background color' : ''},  # 셀 가로 크기 같게 설정
        #     style_data_conditional=[ ],
        # ),


        dcc.Graph(
                    id='line1',
                    figure={
                        'data': [
                            go.Scatter(
                                x=ETF_cum.index,  # Assuming your DataFrame has a suitable index
                                y=ETF_cum[column],
                                mode='lines',
                                name=column
                            ) for column in ETF_cum[Best_10].columns 
                        ],

                        'layout': {
                            'title': 'Best 10 ETF',
                            'xaxis': {'title': 'Date'},
                            'yaxis': {'title': 'Return','tickformat': '.1%'},
                        }

                    },
                    style={'width': '100%', 'margin': 'auto'},
                ), 




        dcc.Graph(
                    id='line2',
                    figure={
                        'data': [
                            go.Scatter(
                                x=ETF_cum.index,  # Assuming your DataFrame has a suitable index
                                y=ETF_cum[column],
                                mode='lines',
                                name=column
                            ) for column in ETF_cum[Worst_10].columns 
                        ],

                        'layout': {
                            'title': 'Worst 10 ETF',
                            'xaxis': {'title': 'Date'},
                            'yaxis': {'title': 'Return','tickformat': '.1%'},
                        }

                    },
                    style={'width': '100%', 'margin': 'auto'},
                ), 





        dcc.Graph(
                    id='line3',
                    figure={
                        'data': [
                            go.Scatter(
                                x=W_3JM13.index,  # Assuming your DataFrame has a suitable index
                                y=W_3JM13[column],
                                mode='lines',
                                name=column
                            ) for column in W_3JM13.columns if ".KS" not in column

                        ],
                        'layout': {
                            'title': 'W_3JM13(S자산배분50)',
                            'xaxis': {'title': 'Date'},
                            'yaxis': {'title': 'Return','tickformat': '.1%'},
                            'annotations': [
                                {
                                    'x': W_3JM13.index[-1],  # 마지막 데이터의 x 좌표
                                    'y': W_3JM13.iloc[-1][column],  # 마지막 데이터의 y 좌표
                                    'xref': 'x',
                                    'yref': 'y',
                                    'text': f'{column}: {W_3JM13.iloc[-1][column]*100:.1f}%',  # 마지막 데이터 값
                                    # 'showarrow': False,
                                    # 'arrowhead': 3,
                                    'ax': 40,
                                    'ay': -20
                                } for column in W_3JM13.columns if ".KS" not in column
                            ],
                            'legend': {'title': 'Legend'},
                        }
                    },
                    style={'width': '100%', 'margin': 'auto'},
                ), 



# 포커스 2040 투자비중
        dcc.Graph(
                    id='line4',
                    figure={
                        'data': [
                            go.Scatter(
                                x=W_07J76.index,  # Assuming your DataFrame has a suitable index
                                y=W_07J76[column],
                                mode='lines',
                                name=column
                            ) for column in W_07J76.columns if ".KS" not in column

                        ],
                        'layout': {
                            'title': 'W_07J76(포커스 2040)',
                            'xaxis': {'title': 'Date'},
                            'yaxis': {'title': 'Return','tickformat': '.1%'},
                            'annotations': [
                                {
                                    'x': W_07J76.index[-1],  # 마지막 데이터의 x 좌표
                                    'y': W_07J76.iloc[-1][column],  # 마지막 데이터의 y 좌표
                                    'xref': 'x',
                                    'yref': 'y',
                                    'text': f'{column}: {W_07J76.iloc[-1][column]*100:.1f}%',  # 마지막 데이터 값
                                    # 'showarrow': False,
                                    # 'arrowhead': 3,
                                    'ax': 40,
                                    'ay': -20
                                } for column in W_07J76.columns if ".KS" not in column
                            ],
                            'legend': {'title': 'Legend'},
                        }
                    },
                    style={'width': '100%', 'margin': 'auto'},
                ), 






# DB 70 투자비중
        dcc.Graph(
                    id='line5',
                    figure={
                        'data': [
                            go.Scatter(
                                x=W_2JM57.index,  # Assuming your DataFrame has a suitable index
                                y=W_2JM57[column],
                                mode='lines',
                                name=column
                            ) for column in W_2JM57.columns if ".KS" not in column

                        ],
                        'layout': {
                            'title': 'W_2JM57(DB70)',
                            'xaxis': {'title': 'Date'},
                            'yaxis': {'title': 'Return','tickformat': '.1%'},
                            'annotations': [
                                {
                                    'x': W_2JM57.index[-1],  # 마지막 데이터의 x 좌표
                                    'y': W_2JM57.iloc[-1][column],  # 마지막 데이터의 y 좌표
                                    'xref': 'x',
                                    'yref': 'y',
                                    'text': f'{column}: {W_2JM57.iloc[-1][column]*100:.1f}%',  # 마지막 데이터 값
                                    # 'showarrow': False,
                                    # 'arrowhead': 3,
                                    'ax': 40,
                                    'ay': -20
                                } for column in W_2JM57.columns if ".KS" not in column
                            ],
                            'legend': {'title': 'Legend'},
                        }
                    },
                    style={'width': '100%', 'margin': 'auto'},
                ), 





# DB 30 투자비중
        dcc.Graph(
                    id='line6',
                    figure={
                        'data': [
                            go.Scatter(
                                x=W_3JM97.index,  # Assuming your DataFrame has a suitable index
                                y=W_3JM97[column],
                                mode='lines',
                                name=column
                            ) for column in W_3JM97.columns if ".KS" not in column

                        ],
                        'layout': {
                            'title': 'W_3JM97(DB30)',
                            'xaxis': {'title': 'Date'},
                            'yaxis': {'title': 'Return','tickformat': '.1%'},
                            'annotations': [
                                {
                                    'x': W_3JM97.index[-1],  # 마지막 데이터의 x 좌표
                                    'y': W_3JM97.iloc[-1][column],  # 마지막 데이터의 y 좌표
                                    'xref': 'x',
                                    'yref': 'y',
                                    'text': f'{column}: {W_3JM97.iloc[-1][column]*100:.1f}%',  # 마지막 데이터 값
                                    # 'showarrow': False,
                                    # 'arrowhead': 3,
                                    'ax': 40,
                                    'ay': -20
                                } for column in W_3JM97.columns if ".KS" not in column
                            ],
                            'legend': {'title': 'Legend'},
                        }
                    },
                    style={'width': '100%', 'margin': 'auto'},
                ), 



#펀드 수익률 그래프


        dcc.Graph(
                    id='line7',
                    figure={
                        'data': [
                            go.Scatter(
                                x=cum.index,  # Assuming your DataFrame has a suitable index
                                y=cum[column],
                                mode='lines',
                                name=column
                            ) for column in ('05W41', '07J86', '2JM57', '3JM97', '3JM13', '03V95')

                        ]+ [
                            go.Scatter(
                                x=BM_cum.index,  
                                y=BM_cum,
                                mode='lines',
                                name='BM(ACWI*50%+KIS*50%)',
                                line=dict(color='firebrick', width=4, dash='dash')  
                            ),
                        ],
                        'layout': {
                            'title': '자산배분펀드별 수익률',
                            'xaxis': {'title': 'Date'},
                            'yaxis': {'title': 'Return','tickformat': '.1%'},
                            'annotations': [
                                {
                                    'x': cum.index[-1],  # 마지막 데이터의 x 좌표
                                    'y': cum.iloc[-1][column],  # 마지막 데이터의 y 좌표
                                    'xref': 'x',
                                    'yref': 'y',
                                    'text': f'{column}: {cum.iloc[-1][column]*100:.1f}%',  # 마지막 데이터 값
                                    'showarrow': False,
                                    # 'arrowhead': 3,
                                    'ax': 40,
                                    'ay': -0
                                } for column in ('05W41', '07J86', '2JM57', '3JM97', '3JM13', '03V95')
                            ],
                            'legend': {'title': 'Legend'},
                        }
                    },
                    style={'width': '100%', 'margin': 'auto'},
                ), 






        dcc.Graph(
                    id='line8',
                    figure={
                        'data': [
                            go.Scatter(
                                x=cum.index,  # Assuming your DataFrame has a suitable index
                                y=cum[column],
                                mode='lines',
                                name=column
                            ) for column in Focus

                        ],
                        'layout': {
                            'title': 'ETF 포커스 수익률',
                            'xaxis': {'title': 'Date'},
                            'yaxis': {'title': 'Return','tickformat': '.1%'},
                            'annotations': [
                                {
                                    'x': cum.index[-1],  # 마지막 데이터의 x 좌표
                                    'y': cum.iloc[-1][column],  # 마지막 데이터의 y 좌표
                                    'xref': 'x',
                                    'yref': 'y',
                                    'text': f'{column}: {cum.iloc[-1][column]*100:.1f}%',  # 마지막 데이터 값
                                    'showarrow': False,
                                    # 'arrowhead': 3,
                                    'ax': 40,
                                    'ay': -0
                                } for column in Focus
                            ],
                            'legend': {'title': 'Legend'},
                        }
                    },
                    style={'width': '100%', 'margin': 'auto'},
                ), 




#TRP 펀드 수익률 그래프
        dcc.Graph(
                    id='line9',
                    figure={
                        'data': [
                            go.Scatter(
                                x=cum.index,  # Assuming your DataFrame has a suitable index
                                y=cum[column],
                                mode='lines',
                                name=column
                            ) for column in TRP_빈티지

                        ],
                        'layout': {
                            'title': 'TRP 수익률',
                            'xaxis': {'title': 'Date'},
                            'yaxis': {'title': 'Return','tickformat': '.1%'},
                            'annotations': [
                                {
                                    'x': cum.index[-1],  # 마지막 데이터의 x 좌표
                                    'y': cum.iloc[-1][column],  # 마지막 데이터의 y 좌표
                                    'xref': 'x',
                                    'yref': 'y',
                                    'text': f'{column}: {cum.iloc[-1][column]*100:.2f}%',  # 마지막 데이터 값
                                    'showarrow': False,
                                    # 'arrowhead': 3,
                                    'ax': 40,
                                    'ay': -0
                                } for column in TRP
                            ],
                            'legend': {'title': 'Legend'},
                        }
                    },
                    style={'width': '100%', 'margin': 'auto'},
                ), 




#S자산배분 펀드 수익률 그래프
        dcc.Graph(
                    id='line10',
                    figure={
                        'data': [
                            go.Scatter(
                                x=cum.index,  # Assuming your DataFrame has a suitable index
                                y=cum[column],
                                mode='lines',
                                name=column
                            ) for column in S자산배분

                        ],
                        'layout': {
                            'title': 'S자산배분 수익률',
                            'xaxis': {'title': 'Date'},
                            'yaxis': {'title': 'Return','tickformat': '.1%'},
                            'annotations': [
                                {
                                    'x': cum.index[-1],  # 마지막 데이터의 x 좌표
                                    'y': cum.iloc[-1][column],  # 마지막 데이터의 y 좌표
                                    'xref': 'x',
                                    'yref': 'y',
                                    'text': f'{column}: {cum.iloc[-1][column]*100:.2f}%',  # 마지막 데이터 값
                                    'showarrow': False,
                                    # 'arrowhead': 3,
                                    'ax': 40,
                                    'ay': -0
                                } for column in S자산배분
                            ],
                            'legend': {'title': 'Legend'},
                        }
                    },
                    style={'width': '100%', 'margin': 'auto'},
                ), 



#cum 수익률 그래프

        dcc.Graph(
                    id='line11',
                    figure={
                        'data': [
                            go.Scatter(
                                x=cum_03V95.index,  # Assuming your DataFrame has a suitable index
                                y=cum_03V95[column],
                                mode='lines',
                                name=column
                            ) for column in cum_03V95.columns if ".KS" not in column
                        ],

                        'layout': go.Layout(
                                title='ctr_03V95',
                                xaxis={'title': 'Date'},
                                yaxis={'title': 'Return', 'tickformat': '.1%'},
                                hovermode='closest',
                            )
                        },
                        style={'width': '100%', 'margin': 'auto'},
                            
                    ),


        dcc.Graph(
                    id='line17',
                    figure={
                        'data': [
                            go.Scatter(
                                x=cum_3JM13.index,  # Assuming your DataFrame has a suitable index
                                y=cum_3JM13[column],
                                mode='lines',
                                name=column
                            ) for column in cum_3JM13.columns if ".KS" not in column
                        ],

                        'layout': go.Layout(
                                title='ctr_3JM13(S자산배분50)',
                                xaxis={'title': 'Date'},
                                yaxis={'title': 'Return', 'tickformat': '.1%'},
                                hovermode='closest',
                            )
                        },
                        style={'width': '100%', 'margin': 'auto'},
                            
                    ),



        dcc.Graph(
                    id='line20',
                    figure={
                        'data': [
                            go.Scatter(
                                x=cum_2JM57.index,  # Assuming your DataFrame has a suitable index
                                y=cum_2JM57[column],
                                mode='lines',
                                name=column
                            ) for column in cum_2JM57.columns if ".KS" not in column
                        ],

                        'layout': go.Layout(
                                title='ctr_2JM57(DB70)',
                                xaxis={'title': 'Date'},
                                yaxis={'title': 'Return', 'tickformat': '.1%'},
                                hovermode='closest',
                            )
                        },
                        style={'width': '100%', 'margin': 'auto'},
                            
                    ),


        dcc.Graph(
                    id='line18',
                    figure={
                        'data': [
                            go.Scatter(
                                x=cum_07J76.index,  # Assuming your DataFrame has a suitable index
                                y=cum_07J76[column],
                                mode='lines',
                                name=column
                            ) for column in cum_07J76.columns if ".KS" not in column
                        ],

                        'layout': go.Layout(
                                title='ctr_07J76(포커스 2040)',
                                xaxis={'title': 'Date'},
                                yaxis={'title': 'Return', 'tickformat': '.1%'},
                                hovermode='closest',
                            )
                        },
                        style={'width': '100%', 'margin': 'auto'},
                            
                    ),
                        







#설정액 그래프

        dcc.Graph(
                    id='graph12',
                    figure={
                        'data': [
                            go.Scatter(x=설정액_Focus.index, y=설정액_Focus.values/100000000, mode='lines', name='Focus', line=dict(color='blue')),
                            go.Scatter(x=설정액_TRP.index, y=설정액_TRP.values/100000000, mode='lines', name='TRP', line=dict(color='red')),
                            go.Scatter(x=설정액_S자산배분.index, y=설정액_S자산배분.values/100000000, mode='lines', name='S자산배분', line=dict(color='green'))
                        ],
                        'layout': go.Layout(
                            title='설정액',
                            xaxis={'title': 'Date'},
                            yaxis={'title': '설정액', 'tickformat': ',.0f'},
                            # legend={'title': 'Groups'},
                            hovermode='closest'
                        )
                    },
                    style={'width': '100%', 'height': '500px'}
                ),



        dcc.Graph(
                    id='line13',
                    figure={
                        'data': [
                            go.Scatter(
                                x=PV_설정액.index,  # Assuming your DataFrame has a suitable index
                                y=PV_설정액[column]/100000000,
                                mode='lines',
                                name=column
                            ) for column in Focus
                        ],

                        'layout': go.Layout(
                                title='Focus 빈티지별 설정액',
                                xaxis={'title': 'Date'},
                                yaxis={'title': 'Return', 'tickformat': '.0f'},
                                hovermode='closest',
                            )
                        },
                        style={'width': '100%', 'margin': 'auto'},
                            
                    ),




        dcc.Graph(
                    id='line14',
                    figure={
                        'data': [
                            go.Scatter(
                                x=PV_설정액.index,  # Assuming your DataFrame has a suitable index
                                y=PV_설정액[column]/100000000,
                                mode='lines',
                                name=column
                            ) for column in TRP
                        ],

                        'layout': go.Layout(
                                title='TRP 빈티지별 설정액',
                                xaxis={'title': 'Date'},
                                yaxis={'title': 'Return', 'tickformat': '.0f'},
                                hovermode='closest',
                            )
                        },
                        style={'width': '100%', 'margin': 'auto'},
                    ),


        dcc.Graph(
                    id='dot15',
                    figure={
                        'data': [
                            go.Scatter(
                                x=R_YTD_Focus.index, 
                                y=R_YTD_Focus,
                                mode='markers',
                                name='ETF 포커스',
                                marker=dict(color='#3762AF', size=14)
                            ),
                            go.Scatter(
                                x=R_YTD_TRP.index, 
                                y=R_YTD_TRP,
                                mode='markers',
                                name='TRP',
                                marker=dict(color='#630', size=14)
                            )
                        ],

                        'layout': go.Layout(
                                title='TDIF 수익률(YTD)',
                                xaxis={'title': '빈티지'},
                                yaxis={'title': 'YTD Return', 'tickformat': '.1%'},
                                hovermode='closest',
                            )
                        },
                        style={'width': '100%', 'margin': 'auto'},
                            
                    ),



# [설정액추이_운용사별.index != 'All']


        dcc.Graph(
                    id='line23',
                    figure={
                        'data': [
                            go.Scatter(
                                x=설정액추이_운용사별.index,  # Assuming your DataFrame has a suitable index
                                y=설정액추이_운용사별[column]/100000000,
                                mode='lines',
                                name=column
                            ) for column in 설정액추이_운용사별.columns if "All" not in column
                        ],

                        'layout': go.Layout(
                                title='TDIF 운용사별 설정액추이',
                                xaxis={'title': 'Date'},
                                yaxis={'title': '설정액', 'tickformat': ',.0f'},
                                hovermode='closest',
                            )
                        },
                        style={'width': '100%', 'margin': 'auto'},
                            
                    ),



        dcc.Graph(
                    id='line24',
                    figure={
                        'data': [
                            go.Scatter(
                                x=설정액추이_판매사별.index,  # Assuming your DataFrame has a suitable index
                                y=설정액추이_판매사별[column]/100000000,
                                mode='lines',
                                name=column
                            ) for column in 설정액추이_판매사별.columns if "All" not in column
                        ],

                        'layout': go.Layout(
                                title='TDIF 판매사별 설정액추이',
                                xaxis={'title': 'Date'},
                                yaxis={'title': '설정액', 'tickformat': ',.0f'},
                                hovermode='closest',
                            )
                        },
                        style={'width': '100%', 'margin': 'auto'},
                            
                    ),




        dcc.Graph(
                    id='line25',
                    figure={
                        'data': [
                            go.Scatter(
                                x=디폴트추이_운용사별.index,  # Assuming your DataFrame has a suitable index
                                y=디폴트추이_운용사별[column]/100000000,
                                mode='lines',
                                name=column
                            ) for column in 디폴트추이_운용사별.columns if "All" not in column
                        ],

                        'layout': go.Layout(
                                title='TDIF 디폴트옵션 설정액_운용사별',
                                xaxis={'title': 'Date'},
                                yaxis={'title': '설정액', 'tickformat': ',.0f'},
                                hovermode='closest',
                            )
                        },
                        style={'width': '100%', 'margin': 'auto'},
                            
                    ),


        dcc.Graph(
                    id='line26',
                    figure={
                        'data': [
                            go.Scatter(
                                x=디폴트추이_판매사별.index,  # Assuming your DataFrame has a suitable index
                                y=디폴트추이_판매사별[column]/100000000,
                                mode='lines',
                                name=column
                            ) for column in 디폴트추이_판매사별.columns if "All" not in column
                        ],

                        'layout': go.Layout(
                                title='TDIF 디폴트옵션 설정액_판매사별',
                                xaxis={'title': 'Date'},
                                yaxis={'title': '설정액', 'tickformat': ',.0f'},
                                hovermode='closest',
                            )
                        },
                        style={'width': '100%', 'margin': 'auto'},
                            
                    ),



        dcc.Graph(
                    id='line27',
                    figure={
                        'data': [
                            go.Scatter(
                                x=설정액추이_빈티지별.index,  # Assuming your DataFrame has a suitable index
                                y=설정액추이_빈티지별[column]/100000000,
                                mode='lines',
                                name=column
                            ) for column in 설정액추이_빈티지별.columns if "All" not in column
                        ],

                        'layout': go.Layout(
                                title='TDIF 설정액_빈티지별',
                                xaxis={'title': 'Date'},
                                yaxis={'title': '설정액', 'tickformat': ',.0f'},
                                hovermode='closest',
                            )
                        },
                        style={'width': '100%', 'margin': 'auto'},
                            
                    ),


    ]
)


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')



    #http://192.168.194.140:8050 : 회사 - Dash



