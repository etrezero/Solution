import os
import pandas as pd
import pymysql
from openpyxl import Workbook
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
import concurrent.futures
import re

import requests
import yfinance as yf
import concurrent.futures
import pickle



cache_price = r'C:\Covenant\data\S자산배분_월간보고서.pkl'
cache_expiry = timedelta(days=1)


# 데이터 가져오기 함수에서 시간대 정보 제거
def fetch_data(code, start, end):
    try:
        session = requests.Session()
        session.verify = False  # SSL 인증서 검증 비활성화
        yf_data = yf.Ticker(code, session=session)
        df_price = yf_data.history(start=start, end=end)['Close'].rename(code)

        # 데이터프레임으로 변환 및 인덱스 포맷 설정
        df_price = pd.DataFrame(df_price)
        df_price = df_price.ffill().bfill()
        df_price.columns = [code]

        # 열 이름을 BM_code_dict 키로 리네임
        renamed_columns = {
            col: key 
            for key in dict_BM.items() 
            for col in df_price.columns }
        
        
        df_price = df_price.rename(columns=renamed_columns)

        df_price.index = pd.to_datetime(df_price.index).strftime('%Y-%m-%d')  # 인덱스를 %Y-%m-%d 형식으로 변환
        df_price.index = pd.to_datetime(df_price.index).tz_localize(None)  # 시간대 정보 제거

        return df_price
    
    except Exception as e:
        print(f"Error fetch data {code}: {e}")
        return None



# 캐시를 통한 데이터 불러오기
def Func(code, start, end, batch_size=10):
    
    # 날짜 형식을 '%Y-%m-%d'로 변환
    start = pd.to_datetime(start).strftime('%Y-%m-%d')
    end = pd.to_datetime(end).strftime('%Y-%m-%d')
    
    if os.path.exists(cache_price):
        cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_price))
        if datetime.now() - cache_mtime < cache_expiry:
            with open(cache_price, 'rb') as f:
                print("Loading cache========================")
                return pickle.load(f)

    data_frames = []
    for i in range(0, len(code), batch_size):
        code_batch = code[i:i + batch_size]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(fetch_data, c, start, end): c for c in code_batch}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    data_frames.append(result)

    price_data = pd.concat(data_frames, axis=1) if data_frames else pd.DataFrame()

    with open(cache_price, 'wb') as f:
        pickle.dump(price_data, f)
        print("Data cached================================")

    return price_data




path = rf'C:\Covenant\data\0. 삼성생명_변액_관리파일.xlsx'
#엑셀 저장=======================================================

def save_excel(df, sheetname, index_option=None):

    # path = rf'C:\Covenant\data\S자산배분_월간보고서_202503.xlsx'
    # 파일이 없는 경우 새 Workbook 생성
    if not os.path.exists(path):
        wb = Workbook()
        wb.save(path)
        print(f"새 파일 '{path}' 생성됨.")
    
    # 인덱스를 날짜로 변환 시도
    try:
        # index_option이 None일 경우 인덱스를 포함하고 날짜 형식으로 저장
        if index_option is None or index_option:  # 인덱스를 포함하는 경우
            df.index = pd.to_datetime(df.index, errors='coerce')
            df.index = df.index.strftime('%Y-%m-%d')  # 벡터화된 방식으로 날짜 포맷 변경
            index = True  # 인덱스를 포함해서 저장
        else:
            index = False  # 인덱스를 제외하고 저장
    except Exception:
        print("Index를 날짜 형식으로 변환할 수 없습니다. 기본 인덱스를 사용합니다.")
        index = index_option if index_option is not None else True  # 변환 실패 시에도 인덱스를 포함하도록 설정

    # DataFrame을 엑셀 시트로 저장
    with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheetname, index=index)  # index 여부 설정
        print(f"'{sheetname}' 저장 완료.")



path = rf'C:\Covenant\data\0. 삼성생명_변액_관리파일.xlsx'

def read_excel_file(path, sheet_name):

    try:
        if not os.path.exists(path):
            print(f"Error: The file does not exist: {path}")
            return None
        df = pd.read_excel(path, sheet_name=sheet_name, engine='openpyxl')
        print(f"Successfully read the '{sheet_name}' sheet into a DataFrame.")
        return df
    except Exception as e:
        print(f"error occurred : reading the Excel file: {e}")
        return None



# # # ================================================
def execute_query(connection, query):
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
            return result
    except Exception as e:
        print(f"Error executing query: {e}")
        return None


def fetch_data(query):
    # MySQL 연결 정보 설정
    connection = pymysql.connect(
        host='192.168.195.55',
        user='solution',
        password='Solution123!',
        database='dt',
        port=3306,
        cursorclass=pymysql.cursors.DictCursor
    )
    try:
        result = execute_query(connection, query)
    finally:
        connection.close()
    
    if result:
        return pd.DataFrame(result)
    else:
        return None


def mainquery(start, end):
    
    cache_Q = rf'C:\Covenant\cache\S자산배분_쿼리_{start}.pkl'
    cache_Q_expiry = timedelta(days=1)

    
    # 캐시 파일이 존재하면 만료 여부 확인
    if os.path.exists(cache_Q):
        cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_Q))
        if datetime.now() - cache_mtime < cache_Q_expiry:
            # 캐시 데이터 로드
            with open(cache_Q, 'rb') as f:
                cached_data = pickle.load(f)
                print("🔹 캐시에서 데이터 로드 완료")
                return cached_data['query1'], cached_data['query2'], cached_data['query3']
    
    
    queries = {
        # 펀드별 투자비중
        'query1': f"""
            SELECT 
                A.STD_DT,               -- 기준일
                A.FUND_CD,              -- 펀드코드
                A.FUND_NM,              -- 펀드명
                A.ITEM_CD,              -- 종목코드
                A.ITEM_NM,              -- 종목명
                A.NAST_TAMT_AGNST_WGH,  -- 순자산비중
                A.APLD_UPR,             -- 적용단가
                B.TKR_CD                -- 티커코드 (TKR_CD)
            FROM 
                DWPM10530 A
            LEFT JOIN 
                DWPI10021 B
            ON 
                A.ITEM_CD = B.ITEM_CD
            WHERE 
                A.FUND_CD IN (
                    '3JM08', '3JM09', '3JM10', '3JM11', 
                    '3JM12', '3JM13', '4JM03', '4JM04'
                )
                AND A.STD_DT >= '{start}' 
                AND A.STD_DT <= '{end}';
        """,

        # 수정기준가 / 설정액 / 순자산
        'query2': f"""
            SELECT 
                STD_DT,          -- 기준일
                FUND_CD,         -- 펀드코드
                MOD_STPR,        -- 수정기준가
                NAST_AMT        -- 순자산

            FROM
                DWPM10510
            WHERE
                FUND_CD IN (
                    '3JM08', '3JM09', '3JM10', '3JM11', 
                    '3JM12', '3JM13', '4JM03', '4JM04'
                )
                AND STD_DT >= '{Y_ago}' AND STD_DT <= '{end}';
        """,


        # 분배금
        'query3': f"""
            SELECT 
                PCS_DT, 
                FUND_CD, 
                ACSB_NM,    -- 계정과목명
                CR_AMT
                
            FROM 
                DWPM11030
            
            WHERE 
                FUND_CD IN ('3JM08', '3JM09', '3JM10', '3JM11', '3JM12', '3JM13', '4JM03', '4JM04')
                AND PCS_DT >= '{start}' AND PCS_DT <= '{end}';"""
    }

                # OPNG_AMT,        -- 설정금액
                # NAST_AMT,        -- 순자산
                # TDD_OPNG_AMT,    -- 당일설정
                # TDD_CLSR_AMT     -- 당일해지


    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_query = {executor.submit(fetch_data, query): name for name, query in queries.items()}
        for future in concurrent.futures.as_completed(future_to_query):
            query_name = future_to_query[future]
            try:
                result = future.result()
                if result is not None:
                    results[query_name] = result
                    print(f"{query_name} Result:\n", result.head())
                else:
                    print(f"{query_name} returned no results.")
            except Exception as e:
                print(f"Error fetching data for {query_name}: {e}")



    # 캐시에 데이터 저장
    with open(cache_Q, 'wb') as f:
        pickle.dump(results, f)
        print("📌 데이터 캐시 저장 완료")

    return results.get('query1'), results.get('query2'), results.get('query3')



if __name__ == "__main__":
    # 오늘 날짜 기준으로 전전월 마지막 날과 지난달 마지막 날 계산
    today = datetime.today()
    # today = datetime.today() - relativedelta(months=1)

    Y_ago = (datetime.today() - relativedelta(years=1)).strftime('%Y%m%d')


    이달초 = today.replace(day=1)
    지난달말 = 이달초 - timedelta(days=1)
    지난달초 = 지난달말.replace(day=1) 
    

    #월간 보고서를 위한 날짜 설정   
    start_1M = 지난달초.strftime('%Y%m%d')
    end = 지난달말.strftime('%Y%m%d')

    #YTD 보고서를 위한 시작 날짜 설정
    연초 = (today.replace(year=today.year, month=1, day=1))
    YTD_start = 연초.strftime('%Y%m%d')




    #Start를 YTD와 1M을 바꿔서 종합 

    # ********************************************************
    start = start_1M
    # start = YTD_start
    # ********************************************************
    
    sheet = "YTD" if start == YTD_start else "M"

    # main 함수 실행 :  건들지 말것
    df_weight, df_기준가, df_미수분배금 = mainquery(start, end)  



    # # 1M 데이터 처리
    # # df_weight, df_기준가, df_미수분배금 = mainquery('20240101', '20240131')
    # # df_weight, df_기준가, df_미수분배금 = mainquery(YTD_start, end)


#? <구간표시>===========================================================================

    # Load data
    BM_R = read_excel_file(path, 'BM_Data')


    # data sheet KOSPI200~유동성
    BM_R.columns = BM_R.iloc[6]
    BM_R = BM_R.iloc[8:, 1:34]
    BM_R = BM_R.drop(BM_R.columns[1:10], axis=1)
    

    BM_R.reset_index(drop=True, inplace=True)
    BM_R.columns.values[0] = 'Date'
    BM_R.reset_index(drop=True, inplace=True)

    # Set 'Date' as index
    BM_R['Date'] = pd.to_datetime(BM_R['Date'], format='%Y-%m-%d', errors='coerce')
    BM_R.dropna(subset=['Date'], inplace=True)
    BM_R.set_index('Date', inplace=True)

    # Remove duplicate indices
    BM_R = BM_R[~BM_R.index.duplicated(keep='first')]

    print("BM_R=============================", BM_R.head(10))





    # Generate continuous date range
    date_range = pd.date_range(start=BM_R.index.min(), end=BM_R.index.max())
    BM_R = BM_R.reindex(date_range)

    BM_R = BM_R.infer_objects()

    BM_R = BM_R.astype(float)
    BM_R.fillna(0, inplace=True)
    
    BM_R = BM_R.fillna(0)

    # # Print results====================================================

    print("BM_R**************************", BM_R.loc['2025-04-30']*100)



    # BMW =====================================================================
    df_BMW = read_excel_file(path, 'BM_Weight')
    df_BMW = df_BMW.iloc[2:]
    df_BMW.columns = df_BMW.iloc[0]
    df_BMW = df_BMW.drop(index=2)
    df_BMW.reset_index(drop=True, inplace=True)



    # Set 'Date' as index
    df_BMW['Date'] = pd.to_datetime(df_BMW['운용일'], format='%Y-%m-%d', errors='coerce')
    df_BMW.dropna(subset=['운용일'], inplace=True)
    df_BMW.set_index('Date', inplace=True)
    # print("df_BMW:\n", df_BMW)

    # ******처음은 그대로 : 마지막은 +1 *******************
    BMW_50 = df_BMW.iloc[:,1:24].fillna(0).astype(float)
    BMW_30 = df_BMW.iloc[:,24:47].fillna(0).astype(float)
    BMW_70 = df_BMW.iloc[:,47:70].fillna(0).astype(float)
    BMW_퇴직70 = df_BMW.iloc[:,70:93].fillna(0).astype(float)

    BMW_해외채권 = df_BMW.iloc[:,93:99].fillna(0).astype(float)
    BMW_국내채권 = df_BMW.iloc[:,99:103].fillna(0).astype(float)

    print("BMW_50.columns====================\n", BMW_50.columns)
    print("BMW_30.columns====================\n", BMW_30.columns)
    print("BMW_70.columns====================\n", BMW_70.columns)
    print("BMW_퇴직70.columns====================\n", BMW_퇴직70.columns)
    print("BMW_해외채권.columns====================\n", BMW_해외채권.columns)
    print("BMW_국내채권.columns====================\n", BMW_국내채권.columns)






    # 각 BMW 데이터프레임을 start부터 end까지 필터링========================================
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    BMW_50 = BMW_50.loc[start_date:end_date]
    BMW_30 = BMW_30.loc[start_date:end_date]
    BMW_70 = BMW_70.loc[start_date:end_date]
    BMW_퇴직70 = BMW_퇴직70.loc[start_date:end_date]
    BMW_해외채권 = BMW_해외채권.loc[start_date:end_date]
    BMW_국내채권 = BMW_국내채권.loc[start_date:end_date]


    # 필터링된 결과 출력
    # print("BMW_50 (Filtered)*****************\n", BMW_50)
    # save_excel(BMW_50, "BMW_50", index_option=None)



    # print("BMW_30 (Filtered)====================\n", BMW_30)
    # print("BMW_70 (Filtered)====================\n", BMW_70)
    # print("BMW_퇴직70 (Filtered)====================\n", BMW_퇴직70)
    # print("BMW_해외채권 (Filtered)====================\n", BMW_해외채권)
    # print("BMW_국내채권 (Filtered)====================\n", BMW_국내채권)
    # # # ===========================================================================================

#? <구간표시>========================================================================


    # # *  자산차 = ∑∑(AP비중(i, t-1)-(BM비중(i, t-1))*r(i, t))), 수익차 = ∑∑((BM비중(i, t-1))*(r(k, t) - r(i, t))), 기타차 = 초과수익 - 자산차 - 수익차 (i = 자산군, t = 시간, k = 종목, r = 수익률)



    # BM_Match =====================================================================
    BM_match = read_excel_file(path, 'BM_Match')
    BM_match = BM_match.iloc[0:,5:9]


    # 1. 첫 번째 행을 컬럼명으로 지정
    BM_match.columns = BM_match.iloc[0, :]

    # 2. 첫 번째 행 제거
    BM_match = BM_match.drop(index=0).reset_index(drop=True)

    # 3. Ticker 컬럼 문자열 정리
    BM_match["Ticker"] = (
        BM_match["Ticker"]
        .str.replace(" US Equity", "", regex=False)
        .str.replace(" KS Equity", "", regex=False)
        .str.strip()
    )

    # 4. 전부 NaN인 열 제거
    BM_match.dropna(how='all', axis=0, inplace=True)  # ❗ 변수 할당 안 함
    BM_match.rename(columns={"(Name)(BBG)": "Full_name"}, inplace=True)


    print("BM_match.columns******************",  BM_match.columns)

    
    # 열 이름 정리 (공백 제거)
    BM_match.columns = BM_match.columns.str.strip()

    # "종목정보" 기준으로 중복 제거 후 처리
    BM_match = BM_match.drop_duplicates(subset="종목정보")
    print("BM_match=============================", BM_match)


    dict_BM = BM_match.groupby('종목정보')['Ticker'].apply(list).to_dict()    
    print("dict_BM=============================", dict_BM)

    #? 딕셔너리 프린트 해서 딕트 BM 만들고 보정해서 써라!

    # 보정한 딕셔너리 덮으씌우고 쓸때
    dict_BM = {
        'Aerospace&Defense': ['PPA'],
        'Agriculture': ['VEGI'],
        'Consumer Staples': ['XLP'],
        'Financials': ['IYF'],
        'KOSDAQ150': ['229200'],
        'KOSPI200': ['069500'],
        'LBMA Silver': ['SLV'],
        'MSCI AC Asia Ex. Japan': ['AAXJ'],
        'Russell 1000 Growth': ['VUG'],
        'MSCI China': ['MCHI'],
        'MSCI EM': ['IEMG'],
        'MSCI India': ['INDA'],
        'MSCI World': ['ACWI','VTI','VEA'],
        'NASDAQ 100': ['QQQ'],
        'Russell 1000 Value': ['VTV'],
        'S&P500': ['SPY'],
        'S&P500 Energy': ['XLE'],
        'Semiconductor': ['SOXX'],
        'US Dividend': ['SCHD'],
        'US Infrastructure': ['PAVE'],
        '국내채권': ['273130', '356540', '114460', '365780', '148070', '439870', '190620', '385540'],
        '해외채권': ['BND','IAGG','EMB','LQD','VCIT'],
    }

#     dict_BM = {
        
#         "KOSPI200": {"KSP2NTR Index", ".*069500.*|.*229200.*",},
#         "S&P500": {"SPTR500N Index",".*SPY.*|.*DIA.*|.*MAGS.*",},

#         "MSCI World": {"NDDUWI Index", ".*URTH.*|.*VEA.*|.*VTI.*",},
#         "MSCI AC Asia Ex. Japan": {"NDUECAXJ Index", ".*AAXJ.*|.*VPL.*",},
#         "MSCI China": {"M1CN Index","MCHI",},
#         "MSCI EM": {"NDUEEGF Index", "IEMG",},
#         "MSCI ACWI Growth": {"M1WD000G Index", ".*ACWI.*",},
#         "Russell 1000 Growth": {"RU1GN30U Index",".*IWF.*|.*VUG.*"},
#         "Russell 1000 Value": {"RU10VATR Index",".*.*VTV.*"},

#         "NASDAQ 100": {"XNDX Index", ".*QQQ.*|.*QQQM.*",},
#         "Semiconductor": {"XSOX Index", "SOXX",},
#         "US Dividend": {"TGPVAN Index", "SCHD",},
#         "Consumer Staples": {"SP5NCONS Index", ".*XLP.*|.*IYK.*|.*XLY.*",},
#         "US Infrastructure": {"NYFSINFT Index", "PAVE",},
#         "S&P500 Energy": {"SPTRENRS Index", "XLE",},
#         "LBMA Silver": {"SLVRLND Index", "SLV",},

#         "Financials": {"SP5NFINL Index", "IYF",},
#         "Aerospace&Defense": {"S5AEROTR Index", "PPA",},
#         "MSCI India": {"NDEUSIA Index", "INDA",},



#         "국내채권": {".*356540.*|.*114460.*|.*365780.*|.*148070.*|.*439870.*|.*273130.*|.*385540.*|.*190620.*"},
#         "해외채권": {".*BND.*|.*BNDX|.*LQD.*|.*VCLT.*|.*VCIT.*|.*EMB.*|.*IAGG.*"},




#         # "KAP종합": {"KBPMABIN Index", "356540.KS",},
#         # "KAP국고채10년": {"KAP KTB 10y TR Index", "356540.KS",},
#         # "KAP국고채30년": {"KBPM30TR Index", "356540.KS",},
#         # "MMF": {"MMF", "356540.KS",},
#         # "기준금리": {"KOCRD Index", "356540.KS",},


#         # "Barclays Global Aggregate TR Hedged": {"LEGATRUH Index", "BND"},
#         # "DM Sovereign": {"LT02TRUU Index", "BND"},
#         # "EM Sovereign (USD)": {"BURCTRUU Index", "EMB"},
#         # "US short-term IG": {"BUC1TRUU Index", "VCIT"},
#         # "US Long-term IG": {"BCR5TRUU Index", "VCLT"},
#         # "US High Yield": {"LF98TRUU Index", "HYG"},

    


#         # "국내채권": ".*채권.*|.*국고.*|.*국채.*|.*통안.*|.*중기.*",
#         # "해외채권": ".*BND.*|.*LQD.*|.*VCLT.*|.*VCIT.*|.*EMB.*",
        
#         # "ACE 종합":".*356540.*",
#         # "ACE 국고채3년": ".*114460.*",
#         # "ACE 국고채10년": ".*365780.*",
#         # 'KOSEF 국고채10년': '.*148070.*',
#         # "KODEX 국고채30년" : ".*439870.*",

#         # "KODEX 종합": ".*273130.*",
#         # "RISE 종합": ".*385540.*",
#         # "KBSTAR 종합": ".*385540.*",

#     }




    #? 열 이름을 dict_BM에 있는 값으로 매핑========================
    
    # 옵션 입력 안하면 그룹합계 False
    def rename_col_to_dict_value(df, dict, groupby_sum: bool = False):
        
        rename_map = {
            col: next(
                (key for key, value in dict.items()
                    if isinstance(value, list) and any(item in col for item in value)
                    or (isinstance(value, str) and value in col)), col
            ) for col in df.columns
        }
        df = df.rename(columns=rename_map)

        if groupby_sum:
            df = df.groupby(df.columns, axis=1).sum()

        return df

    #? df의 열 이름을 dict_BM에 있는 값으로 매핑========================
    












#  #? ======================================================================================



    df_미수분배금 = df_미수분배금.loc[
        (df_미수분배금['ACSB_NM'] == 'ETF분배금수익') | 
        (df_미수분배금['ACSB_NM'] == '집합투자증권분배금수익') |
        (df_미수분배금['ACSB_NM'] == '주식배당금수익') 
    ]


    PV_미수분배금 = df_미수분배금.pivot_table(
                index='PCS_DT',
                columns=['FUND_CD', 'ACSB_NM'],
                values='CR_AMT',
                aggfunc='sum',
    )

    # print("PV_미수분배금==============", PV_미수분배금)


    PV_미수분배금 = PV_미수분배금.groupby(level=0, axis=1).sum()
    # print("PV_미수분배금==============", PV_미수분배금)



    PV_순자산 = df_기준가.pivot_table(
                index='STD_DT',
                columns=['FUND_CD'],
                values='NAST_AMT',
                aggfunc='sum',
    )

    # print("PV_순자산==============", PV_순자산)
    


    # 두 데이터프레임의 인덱스 이름 통일
    PV_미수분배금.index.name = 'STD_DT'
    PV_순자산.index.name = 'STD_DT'

    # 동일한 FUND_CD를 기준으로 나누기
    PV_분배금비율 = PV_미수분배금.div(PV_순자산, level=0, axis=1).fillna(0)

    # print("PV_분배금비율==============", PV_분배금비율)



    # 딕셔너리 생성
    dict_FUND_CD = {
        # '3JM08': '30',
        # '3JM09': '30',
        # '3JM10': '30',
        # '3JM11': '50',
        # '3JM12': '30',
        '3JM13': '50',
        # '4JM03': '70',
        # '4JM04': '퇴직70'
    }

    # FUND_CD 목록
    List_FUND_CD = list(dict_FUND_CD.keys())


    # 제외할 문자열 리스트
    exclude = ['KRW/USD', 'FX', '미수금', '기타자산', 'DEPOSIT', '예금', 'REPO', '원천세', '분배금', '미지급금', 'CALL']
    
    

    # 제외 조건에 맞는 데이터 필터링
    df_weight = df_weight[~df_weight['ITEM_NM'].str.contains('|'.join(exclude), na=False)]

    df_weight = df_weight[~df_weight['ITEM_CD'].str.contains('FXW', na=False)]

    print("df_weight=======================", df_weight)


    # TKR_CD 키로, ITEM_NM을 값으로 하는 딕셔너리 생성
    dict_ITEM = (
        df_weight[['ITEM_NM', 'TKR_CD', 'ITEM_CD']]
        .dropna(subset=['ITEM_NM'])  # ITEM_NM이 NaN인 행 제거
        .drop_duplicates(subset=['ITEM_NM'])  # ITEM_NM 중복 제거
        .set_index('ITEM_NM')
        .to_dict(orient='index')
    )
    

    # TKR_CD에서 " US" 제거
    dict_ITEM = {
        key: {
            **value, 
            'TKR_CD': value['TKR_CD'].replace(" US", "") if 'TKR_CD' in value and value['TKR_CD'] else value['TKR_CD']
        } for key, value in dict_ITEM.items()
    }

    # print("dict_ITEM=======================", dict_ITEM)

#  #? ======================================================================================







    def generate_PV_기준가(List_FUND_CD, df_기준가):
        PV_기준가 = {}
        펀드_R = {}
        펀드_cum = {}


        for code in List_FUND_CD:
            # 데이터 필터링
            df_filtered = df_기준가.loc[df_기준가['FUND_CD'] == code]

            # 기준가 피벗 테이블 생성
            PV = df_filtered.pivot_table(
                index='STD_DT',
                values='MOD_STPR',
                aggfunc='mean',
            )

            # 결측값 처리
            PV = PV.fillna(0).ffill().bfill()

            # 수익률 및 누적 수익률 계산
            R = PV.pct_change().fillna(0)
            cum = (1 + R).cumprod() - 1

            # 사전에 저장
            PV_기준가[code] = PV
            펀드_R[code] = R
            펀드_cum[code] = cum

            # print(f"PV_기준가 {code}====================================\n", PV.head())
            # print(f"펀드_R {code}====================================\n", R.head())
            # print(f"펀드_cum {code}====================================\n", cum.head())


        # ? 기여도J 업데이트에 필요 : Group_기준가
        Group_기준가 = pd.concat(PV_기준가.values(), axis=1)
        Group_기준가.columns = List_FUND_CD  # 컬럼명: 펀드코드


        # with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        #     for code in PV_기준가.keys():
        #         PV_기준가[code].to_excel(writer, sheet_name=f"PV_기준가_{code}_{sheet}")
        #         펀드_R[code].to_excel(writer, sheet_name=f"R_{code}_{sheet}")
        #         펀드_cum[code].to_excel(writer, sheet_name=f"cum_{code}_{sheet}")
        #         print(f"'{code}' 관련 데이터 저장 완료.")

        return PV_기준가, 펀드_R, 펀드_cum, Group_기준가


    PV_기준가, 펀드_R, 펀드_cum, Group_기준가 = generate_PV_기준가(List_FUND_CD, df_기준가)
    
    print(f"Group_기준가====================================\n", PV_기준가)
    save_excel(Group_기준가, "Group_기준가", index_option=None)



    def generate_PV_단가(List_FUND_CD, df_weight, dict_ITEM):
        PV_단가 = {}
        for code in List_FUND_CD:
            df_filtered = df_weight.loc[df_weight['FUND_CD'] == code]

            PV = df_filtered.pivot_table(
                index='STD_DT',
                columns='ITEM_NM',
                values='APLD_UPR',  #적용단가
                aggfunc='mean',
            )

            
            PV = PV.replace(0, pd.NA).ffill().bfill()

            PV.rename(columns=lambda col: next(
                (value['TKR_CD'] if value['TKR_CD'] else value['ITEM_CD'][-9:-3])
                for key, value in dict_ITEM.items() if col == key
            ), inplace=True)

            PV_단가[code] = PV
            # print(f"PV_단가 {code} ====================================\n", PV.head())

        # with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        #     for code, PV in PV_단가.items():
        #         PV.to_excel(writer, sheet_name=f"PV_단가_{code}_{sheet}", index=True)
        #         print(f"'PV_단가_{code}_{sheet}' 엑셀 저장 완료.")

        return PV_단가

    PV_단가 = generate_PV_단가(List_FUND_CD, df_weight, dict_ITEM)




    def generate_PV_W(List_FUND_CD, df_weight, dict_ITEM):
        PV_W = {}

        for code in List_FUND_CD:
            df_filtered = df_weight.loc[df_weight['FUND_CD'] == code]

            PV = df_filtered.pivot_table(
                index='STD_DT',
                columns='ITEM_NM',
                values='NAST_TAMT_AGNST_WGH',
                aggfunc='mean',
            )

            PV = PV.fillna(0) / 100

            PV.rename(columns=lambda col: next(
                (value['TKR_CD'] if value['TKR_CD'] else value['ITEM_CD'][-9:-3])
                for key, value in dict_ITEM.items() if col == key
            ), inplace=True)
            
            PV.columns = PV.columns.astype(str)
            PV_W[code] = PV
            
            print(f"PV_W_{code}_{sheet}====================================\n", PV.head())

        # with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        #     for code, pv_table in PV_W.items():
        #         pv_table.to_excel(writer, sheet_name=f"PV_W_{code}_{sheet}", index=True)
        #         print(f"'PV_W_{code}_{sheet}' 저장 완료.")

        return PV_W

    PV_W = generate_PV_W(List_FUND_CD, df_weight, dict_ITEM)



    def generate_ctr(PV_W, PV_단가, List_FUND_CD):
        ctr = {}
        단가_R = {}

        for code in List_FUND_CD:
            common_columns = PV_W[code].columns.astype(str).intersection(PV_단가[code].columns.astype(str))

            if not common_columns.empty:
                단가_R = PV_단가[code].pct_change().fillna(0)
                단가_R.replace([float('inf'), float('-inf')], 0, inplace=True)

                단가_R_filtered = 단가_R[common_columns]
                PV_W_filtered = PV_W[code][common_columns]

                try:
                    ctr_code = PV_W_filtered * 단가_R_filtered
                    ctr_code

                    ctr[code] = ctr_code

                    # print(f"ctr_{code}_{sheet} ====================================\n", ctr_code.head())
                except ValueError as e:
                    print(f"ctr_{code}_{sheet} 생성 중 에러 발생: {e}")
            else:
                print(f"공통 열이 없어 ctr_{code}_{sheet} 생성할 수 없습니다.")

        
        # with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        #     for code, ctr_table in ctr.items():
        #         단가_R.to_excel(writer, sheet_name=f"단가_R{code}", index=True)
        #         print(f"'단가_R{code}' 저장 완료.")

        #     for code, ctr_table in ctr.items():
        #         ctr_table.to_excel(writer, sheet_name=f"ctr_{code}_{sheet}", index=True)
        #         print(f"'ctr_{code}_{sheet}' 개별종목 저장 완료.")


        return ctr, 단가_R

    PV_ctr, 단가_R = generate_ctr(PV_W, PV_단가, List_FUND_CD)
    
    








# ?====================================================================================


    def generate_AP_W(PV_W, dict_BM):
        AP_W = {}

        for code, pv_w_df in PV_W.items():
            # 각 펀드별 DataFrame에 대해 열 이름 리네임 및 groupby sum
            ap_w_df = rename_col_to_dict_value(pv_w_df, dict_BM, groupby_sum=True)

            # AP_W에 저장
            AP_W[code] = ap_w_df

            # 엑셀 저장
            # save_excel(ap_w_df, f"AP_W_{code}", index_option=None)
            print(f"AP_W_{code} ====================================\n", ap_w_df.head())

        return AP_W

    # AP_W 생성 호출
    AP_W = generate_AP_W(PV_W, dict_BM)







    def generate_EX_W(AP_W, dict_FUND_CD, BMW_30, BMW_50, BMW_70, BMW_퇴직70):
        EX_W = {}

        for code, ap_w_df in AP_W.items():
            # dict_FUND_CD에서 해당 펀드 코드에 해당하는 값을 가져옴
            fund_name = dict_FUND_CD.get(code, "")
            print(f"fund_name============ {fund_name}")

            # BMW 선택 조건
            if "30" in fund_name:
                bmw_df = BMW_30.copy()
            elif "50" in fund_name:
                bmw_df = BMW_50.copy()
            elif "70" in fund_name and "퇴직" not in fund_name:
                bmw_df = BMW_70.copy()
            elif "퇴직70" in fund_name:
                bmw_df = BMW_퇴직70.copy()
            else:
                print(f"{code}에 대한 BMW 매칭 조건을 찾을 수 없습니다.")
                continue

            # 인덱스를 datetime 형식으로 변환 (필요 시)
            ap_w_df.index = pd.to_datetime(ap_w_df.index, errors="coerce")
            bmw_df.index = pd.to_datetime(bmw_df.index, errors="coerce")

            # 인덱스 맞춤
            ap_w_df = ap_w_df.reindex(bmw_df.index).fillna(0)
            bmw_df = bmw_df.reindex(ap_w_df.index).fillna(0)

            # 공통 열 확인
            common_columns = list(set(ap_w_df.columns) & set(bmw_df.columns))

            if not common_columns:
                print(f"공통 열이 없습니다: {code}")
                continue

            # AP_W와 BMW 차이 계산
            ex_w_df = ap_w_df[common_columns] - bmw_df[common_columns]

            # 결과 저장
            EX_W[code] = ex_w_df
            # print(f"EX_W_{code} ====================================\n", ex_w_df.head())
        
        # # 엑셀 저장
        # with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        #     for code, ex_w_df in EX_W.items():
        #         ex_w_df.to_excel(writer, sheet_name=f"EX_W_{code}_{sheet}", index=True)
        #         print(f"'EX_W_{code}_{sheet}' 저장 완료.")

        return EX_W

    # EX_W 생성
    EX_W = generate_EX_W(AP_W, dict_FUND_CD, BMW_30, BMW_50, BMW_70, BMW_퇴직70)











    def generate_자산차(PV_ctr, EX_W, dict_FUND_CD, BMW_30, BMW_50, BMW_70, BMW_퇴직70, BM_R, dict_BM):
        자산차 = {}

        for code, ex_w_df in EX_W.items():
            print(f"Processing 자산차 for {code}")

            # dict_FUND_CD에서 해당 펀드 코드에 맞는 BMW 선택
            fund_name = dict_FUND_CD.get(code, "")
            if "30" in fund_name:
                bm_w_df = BMW_30.copy()
            elif "50" in fund_name:
                bm_w_df = BMW_50.copy()
            elif "70" in fund_name and "퇴직" not in fund_name:
                bm_w_df = BMW_70.copy()
            elif "퇴직70" in fund_name:
                bm_w_df = BMW_퇴직70.copy()
            else:
                print(f"{code}에 대한 BMW 매칭 조건을 찾을 수 없습니다.")
                continue

            # PV_ctr에서 인덱스 참조
            if code in PV_ctr:
                target_index = PV_ctr[code].index
            else:
                print(f"{code}에 대한 PV_ctr 데이터가 없습니다.")
                continue

            # 인덱스 및 열 이름 정렬
            bm_w_df = bm_w_df.reindex(target_index).fillna(0)
            bm_r_df = BM_R.reindex(target_index).fillna(0)
            ex_w_df = ex_w_df.reindex(target_index).fillna(0)
            

            ex_w_df = rename_col_to_dict_value(ex_w_df, dict_BM)
            bm_w_df = rename_col_to_dict_value(bm_w_df, dict_BM)
            bm_r_df = rename_col_to_dict_value(bm_w_df, dict_BM)

            # 공통 열 확인
            common_columns = list(
                set(ex_w_df.columns) & set(bm_w_df.columns) & set(bm_r_df.columns)
            )
            if not common_columns:
                print(f"공통 열이 없습니다: {code}")
                continue

            # 공통 열로 데이터 필터링
            ex_w_filtered = ex_w_df[common_columns]
            bm_r_filtered = bm_r_df[common_columns]

            # 자산 배분 효과 계산
            try:
                자산차_df = ex_w_filtered * bm_r_filtered

                # 첫 번째 값을 0으로 리셋하고 누적 수익률 계산
                자산차_df.iloc[0] = 0
                자산차_df = 자산차_df.cumsum()

                # 결과 저장
                자산차[code] = 자산차_df.fillna(0)
                # print(f"자산차 for {code} ====================================\n", 자산차_df.head())

                # !엑셀 저장  == 주석 풀지 말것 아래와 겹침
                # with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                #     자산차_df.to_excel(writer, sheet_name=f"자산차_{code}_{sheet}", index=True)
                #     print(f"자산차_{code}_{sheet} 저장 완료.")

            except Exception as e:
                print(f"Error calculating 자산차 for {code}: {e}")

        return 자산차

    # Example call
    자산차 = generate_자산차(PV_ctr, EX_W, dict_FUND_CD, BMW_30, BMW_50, BMW_70, BMW_퇴직70, BM_R, dict_BM)



    def generate_수익차(PV_ctr, AP_W, dict_FUND_CD, BMW_30, BMW_50, BMW_70, BMW_퇴직70, BM_R, 단가_R, dict_BM):
        수익차 = {}

        for code, ctr_df in PV_ctr.items():
            print(f"Processing 수익차 for {code}")

            # dict_FUND_CD에서 해당 펀드 코드에 맞는 BMW 선택
            fund_name = dict_FUND_CD.get(code, "")
            if "30" in fund_name:
                bm_w_df = BMW_30.copy()
            elif "50" in fund_name:
                bm_w_df = BMW_50.copy()
            elif "70" in fund_name and "퇴직" not in fund_name:
                bm_w_df = BMW_70.copy()
            elif "퇴직70" in fund_name:
                bm_w_df = BMW_퇴직70.copy()
            else:
                print(f"{code}에 대한 BMW 매칭 조건을 찾을 수 없습니다.")
                continue

            # AP_W 가져오기
            if code not in AP_W:
                print(f"{code}에 대한 AP_W가 없습니다.")
                continue

            ap_w_df = AP_W[code]

            # 인덱스 및 열 이름 정렬
            ap_w_df = ap_w_df.reindex(ctr_df.index).fillna(0)
            bm_w_df = bm_w_df.reindex(ctr_df.index).fillna(0)
            bm_r_df = BM_R.reindex(ctr_df.index).fillna(0)
            단가_r_df = 단가_R.reindex(ctr_df.index).fillna(0)


            ap_w_df = rename_col_to_dict_value(ap_w_df, dict_BM)
            bm_w_df = rename_col_to_dict_value(bm_w_df, dict_BM)
            bm_r_df = rename_col_to_dict_value(bm_r_df, dict_BM)
            단가_r_df = rename_col_to_dict_value(단가_r_df, dict_BM)
            ctr_df = rename_col_to_dict_value(ctr_df, dict_BM)

            # print("ctr_df###################", ctr_df)

            # 공통 열 확인
            common_columns = list(
                set(ctr_df.columns.astype(str)) & set(ap_w_df.columns.astype(str)) & set(bm_w_df.columns.astype(str)) & set(bm_r_df.columns.astype(str)) & set(단가_r_df.columns.astype(str))
            )
            if not common_columns:
                print(f"공통 열이 없습니다: {code}")
                continue

            # 공통 열로 데이터 필터링
            ap_w_filtered = ap_w_df[common_columns]
            bm_w_filtered = bm_w_df[common_columns]
            bm_r_filtered = bm_r_df[common_columns]
            bm_ctr_filtered = bm_w_filtered * bm_r_filtered
            단가_r_filtered = 단가_r_df[common_columns]
            ctr_df_filtered = ctr_df[common_columns]

            # 동일한 열 이름으로 그룹화하여 합계 계산
            ctr_df_grouped = ctr_df_filtered.groupby(ctr_df_filtered.columns, axis=1).sum()
            ap_w_filtered_grouped = ap_w_filtered.groupby(ap_w_filtered.columns, axis=1).sum()
            bm_w_filtered_grouped = bm_w_filtered.groupby(bm_w_filtered.columns, axis=1).sum()
            bm_r_filtered_grouped = bm_r_filtered.groupby(bm_r_filtered.columns, axis=1).sum()
            단가_r_filtered_grouped = 단가_r_filtered.groupby(단가_r_filtered.columns, axis=1).sum()


            # 확인 출력
            print("**************bm_w_filtered_grouped**************", bm_w_filtered_grouped)
            print("**************ctr_df_grouped**************", ctr_df_grouped)

            # 동일한 열 이름끼리 나누기 위해 교집합을 찾고 나누기
            common_columns_for_div = list(set(ctr_df_grouped.columns) & set(ap_w_filtered_grouped.columns))

            # 가중평균수익률 계산 (동일한 열 이름끼리 나누기)
            가중평균수익률 = ctr_df_grouped[common_columns_for_div].div(ap_w_filtered_grouped[common_columns_for_div])
            가중평균수익률 = 가중평균수익률.fillna(0)

            # 종목 선택 효과 계산
            try:
                수익차_df = bm_w_filtered_grouped * (가중평균수익률 - bm_r_filtered_grouped)
                수익차_df.iloc[0] = 0
                수익차_df = 수익차_df.cumsum()
                수익차[code] = 수익차_df.fillna(0)
                # print(f"수익차 for {code} ====================================\n", 수익차_df.head())

                자산차_df = bm_r_filtered_grouped * (ap_w_filtered_grouped - bm_w_filtered_grouped)
                자산차_df.iloc[0] = 0
                자산차_df = 자산차_df.cumsum()
                자산차[code] = 자산차_df.fillna(0)
                # print(f"자산차 for {code} ====================================\n", 자산차_df.head())


                # # 엑셀 저장 (전체 비활성화 할것 - 테스트용임)
                # with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                #     수익차_df.to_excel(writer, sheet_name=f"수익차_{code}_{sheet}", index=True)
                #     자산차_df.to_excel(writer, sheet_name=f"자산차_{code}_{sheet}", index=True)
                #     bm_w_filtered_grouped.to_excel(writer, sheet_name=f"bm_w_filtered_grouped{code}", index=True)
                #     bm_r_filtered_grouped.to_excel(writer, sheet_name=f"bm_r_filtered_grouped{code}", index=True)
                #     bm_ctr_filtered.to_excel(writer, sheet_name=f"bm_ctr_filtered{code}", index=True)
                #     ap_w_filtered_grouped.to_excel(writer, sheet_name=f"ap_w_filtered_grouped{code}", index=True)
                #     단가_r_filtered_grouped.to_excel(writer, sheet_name=f"단가_r_filtered_grouped{code}", index=True)
                #     ctr_df_grouped.to_excel(writer, sheet_name=f"ctr_df_grouped{code}", index=True)
                #     가중평균수익률.to_excel(writer, sheet_name=f"가중평균수익률{code}", index=True)

            except Exception as e:
                print(f"Error calculating 수익차 for {code}: {e}")

        return 수익차

    # 함수 호출
    수익차 = generate_수익차(PV_ctr, AP_W, dict_FUND_CD, BMW_30, BMW_50, BMW_70, BMW_퇴직70, BM_R, 단가_R, dict_BM)



    def generate_Total_BM(BMW_dict, BM_R):
        Total_BM = {}
        BM_ctr = {}

        for code, bmw_df in BMW_dict.items():
            if not isinstance(bmw_df, pd.DataFrame):
                print(f"{code}의 데이터가 DataFrame 형식이 아닙니다. 전달된 데이터: {type(bmw_df)}")
                continue

            # 인덱스를 datetime 형식으로 변환
            bmw_df.index = pd.to_datetime(bmw_df.index, errors="coerce")
            BM_R.index = pd.to_datetime(BM_R.index, errors="coerce")

            # 공통 열 확인
            common_columns = list(set(bmw_df.columns.astype(str)) & set(BM_R.columns.astype(str)))

            if not common_columns:
                print(f"공통 열이 없습니다: {code}")
                continue

            # 공통 열로 데이터 필터링
            filtered_bmw = bmw_df[common_columns]
            filtered_bm_r = BM_R[common_columns]

            # BMW와 BM_R 곱하여 BM_ctr 계산
            bm_ctr_df = filtered_bmw * (filtered_bm_r)
            BM_ctr[code] = bm_ctr_df

            # 총 BM 계산
            sum_BM_ctr = bm_ctr_df.sum(axis=1)

            # 누적 수익률 계산
            Total_BM_cum = (1 + sum_BM_ctr).cumprod() - 1

            # 결과 저장
            Total_BM[code] = Total_BM_cum

            # 엑셀 저장
            # with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            #     # BM_ctr 저장
            #     BM_ctr_df = BM_ctr[code]  # dict에서 code에 해당하는 값을 데이터프레임으로 가져옴
            #     BM_ctr_df.to_excel(writer, sheet_name=f"BM_ctr_{code}_{sheet}", index=True)

            #     # Total_BM 저장
            #     Total_BM_cum = Total_BM[code].to_frame(name=f"Total_BM_{code}_{sheet}")
            #     Total_BM_cum.to_excel(writer, sheet_name=f"Total_BM_{code}_{sheet}", index=True)

            #     # print(f"Total_BM_{code}_{sheet} 및 BM_ctr_{code}_{sheet} 저장 완료.")

        return Total_BM, BM_ctr



    # Example function call
    Total_BM, BM_ctr = generate_Total_BM(
        {"30": BMW_30, "50": BMW_50, "70": BMW_70, "퇴직70": BMW_퇴직70},
        BM_R
    )



    def generate_초과수익(
        PV_ctr, BM_ctr, dict_FUND_CD, 
        BMW_30, ctBMW_50, BMW_70, BMW_퇴직70, 
        BM_R, dict_BM, PV_분배금비율
    ):
        
        EX_R = {}

        for code, pv_data in PV_ctr.items():
            print(f"Processing 초과수익 for {code}")

            # BMW 선택
            fund_name = dict_FUND_CD.get(code, "")
            bmw_df = (
                BMW_30 if "30" in fund_name else
                BMW_50 if "50" in fund_name else
                BMW_70 if "70" in fund_name and "퇴직" not in fund_name else
                BMW_퇴직70 if "퇴직70" in fund_name else None
            )

            if bmw_df is None:
                print(f"{code}에 대한 BMW 매칭 조건을 찾을 수 없습니다.")
                continue

            # 공통 열 정의
            common_columns = list(set(bmw_df.columns.astype(str)) & set(BM_R.columns.astype(str)))
            if not common_columns:
                print(f"{code}에 공통 열이 없습니다.")
                continue

            # BM_ctr 계산
            BM_ctr[code] = bmw_df[common_columns] * BM_R[common_columns]

            # 열 이름 매핑



            pv_data = rename_col_to_dict_value(pv_data, dict_BM)

            # 그룹화 및 공통 열 계산
            grouped_PV_ctr = pv_data.groupby(pv_data.columns, axis=1).sum()
            common_columns = list(set(grouped_PV_ctr.columns.astype(str)) & set(BM_ctr[code].columns.astype(str)))
            if not common_columns:
                print(f"초과 수익 계산 공통 열이 없습니다: {code}")
                continue

            grouped_PV_ctr = grouped_PV_ctr[common_columns].reindex(grouped_PV_ctr.index).fillna(0)
            
            # 분배금 열 추가
            if '분배금' not in grouped_PV_ctr.columns:
                grouped_PV_ctr['분배금'] = PV_분배금비율.get(code, pd.Series(index=grouped_PV_ctr.index, dtype=float))
            
            
            BM_ctr_filtered = BM_ctr[code].reindex(grouped_PV_ctr.index).fillna(0)
            BM_ctr_filtered['분배금'] = 0


            # print("grouped_PV_ctr********************", grouped_PV_ctr)
            # print("BM_ctr_filtered********************", BM_ctr_filtered)

            # 초과 수익률 계산
            try:
                ex_r_df = grouped_PV_ctr - BM_ctr_filtered
                EX_R[code] = ex_r_df
                cum_EX_R = (1 + ex_r_df).cumprod() - 1



            except Exception as e:
                print(f"Error calculating EX_R for {code}: {e}")
                continue



        # # Save the results to Excel
        # with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        #     for code, ex_r_df in EX_R.items():
        #         ex_r_df.to_excel(writer, sheet_name=f"EX_R_{code}_{sheet}", index=True)
        #         cum_EX_R.to_excel(writer, sheet_name=f"cum_EX_R{code}_{sheet}", index=True)

        #         print(f"'{code}' 초과 수익 저장 완료.")
            
                    
        #         grouped_PV_ctr.to_excel(writer, sheet_name=f"grouped_PV_ctr_{code}_{sheet}", index=True)
        #         print(f"'{code}' grouped_PV_ctr 저장 완료.")

        return EX_R, BM_ctr_filtered, grouped_PV_ctr



    EX_R, BM_ctr_filtered, grouped_PV_ctr = generate_초과수익(
        PV_ctr, BM_ctr, dict_FUND_CD, 
        BMW_30, BMW_50, BMW_70, BMW_퇴직70, 
        BM_R, dict_BM, PV_분배금비율
    )



    def merge_종합(EX_R, 자산차, 수익차, AP_W, dict_FUND_CD, BMW_30, BMW_50, BMW_70, BMW_퇴직70):
        # 최종 결과 저장용 데이터프레임 리스트
        results = []

        for code in 자산차.keys():
            # 자산차 마지막 행 추출
            자산차_last_row = 자산차[code].iloc[-1]
            자산차_last_row.name = f"{code}_자산차"

            # 수익차 마지막 행 추출
            if code in 수익차:
                수익차_last_row = 수익차[code].iloc[-1]
                수익차_last_row.name = f"{code}_수익차"
            else:
                수익차_last_row = pd.Series(0, index=자산차_last_row.index, name=f"{code}_수익차")

            # EX_R 마지막 행 추출
            if code in EX_R:
                ex_r_last_row = ((1+EX_R[code]).cumprod()-1).iloc[-1]
                ex_r_last_row.name = f"{code}_초과수익"
            else:
                ex_r_last_row = pd.Series(0, index=자산차_last_row.index, name=f"{code}_초과수익")

            # AP_W 마지막 행 추출
            if code in AP_W:
                ap_w_last_row = AP_W[code].mean()
                ap_w_last_row.name = f"{code}_AP_W"
            else:
                ap_w_last_row = pd.Series(0, index=자산차_last_row.index, name=f"{code}_AP_W")

            # BMW 데이터 가져오기
            fund_name = dict_FUND_CD.get(code, "")
            if "30" in fund_name:
                bmw_df = BMW_30
            elif "50" in fund_name:
                bmw_df = BMW_50
            elif "70" in fund_name and "퇴직" not in fund_name:
                bmw_df = BMW_70
            elif "퇴직70" in fund_name:
                bmw_df = BMW_퇴직70
            else:
                bmw_df = None

            # BMW 마지막 행 추출
            if bmw_df is not None:
                bm_w_last_row = bmw_df.mean()
                bm_w_last_row.name = f"{code}_BMW"
            else:
                bm_w_last_row = pd.Series(0, index=자산차_last_row.index, name=f"{code}_BMW")

            # 기타차 계산 (초과수익 - (자산차 + 수익차))
            기타차_last_row = ex_r_last_row - (자산차_last_row + 수익차_last_row)
            기타차_last_row.name = f"{code}_기타차"

            # 병합할 시리즈를 리스트에 추가
            results.append(ap_w_last_row)
            results.append(bm_w_last_row)
            results.append(ex_r_last_row)
            results.append(자산차_last_row)
            results.append(수익차_last_row)
            results.append(기타차_last_row)

        # 중복된 인덱스 확인 및 제거
        for i, series in enumerate(results):
            if series.index.duplicated().any():
                series = series[~series.index.duplicated(keep="first")]
                results[i] = series

        # 병합된 결과를 데이터프레임으로 변환
        try:
            종합 = pd.concat(results, axis=1).T.fillna(0)
        except ValueError as e:
            print(f"병합 중 오류 발생: {e}")
            # 인덱스를 고유하게 변경
            unique_results = [series.reset_index(drop=True) for series in results]
            종합 = pd.concat(unique_results, axis=1).T.fillna(0)

        # 합계 열 추가
        종합["합계"] = 종합.sum(axis=1)

        # 엑셀 저장
        with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            종합.to_excel(writer, sheet_name=f"종합_{sheet}", index=True)
            print(f"{path}에 종합 저장 완료.")

        # 결과 반환
        return 종합

    # 함수 호출
    종합 = merge_종합(EX_R, 자산차, 수익차, AP_W, dict_FUND_CD, BMW_30, BMW_50, BMW_70, BMW_퇴직70)


    # 결과 출력
    print("자산차, 수익차, 초과수익 병합 결과 ====================================", 종합)
    

    print("*********************작업완료*******************")
