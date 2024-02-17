import pandas as pd
from tqdm import tqdm

# 엑셀 파일 경로
path_요약 = 'C:/Covenant/data/요약.xlsx'
sheet_from = 'Price1'
path_stat = 'C:/Covenant/data/calcul.xlsx'

try:
    # 엑셀 파일에서 데이터프레임 읽어오기
    df = pd.read_excel(path_요약, sheet_name=sheet_from, index_col='Date')

    # 숫자로 인식되어야 하는 열에 문자열 데이터가 있는지 확인하고, 있다면 숫자 형식으로 변환
    df = df.apply(pd.to_numeric, errors='coerce')

    # NA 또는 NaN 값이 있는지 확인하고, 있다면 0으로 대체
    df.fillna(0, inplace=True)

    # tqdm을 사용하여 데이터프레임의 처리과정을 모니터링

    df_1D_R = df.pct_change(periods=1)
    df_W_R = df.pct_change(periods=7)
    df_1M_R = df.pct_change(periods=30)
    df_3M_R = df.pct_change(periods=90)
    df_6M_R = df.pct_change(periods=180)
    df_1Y_R = df.pct_change(periods=365)
    df_2Y_R = df.pct_change(periods=730)
    df_3Y_R = df.pct_change(periods=1095)

    df_W_interval = df_W_R.resample('W').last()  # 7일수익률데이터를 한 주의 마지막 날짜로 데이터를 샘플링합니다.
    df_1M_Vol = df_W_interval.rolling(window=4).std() * (52 ** 0.5)
    df_3M_Vol = df_W_interval.rolling(window=4 * 3).std() * (52 ** 0.5)
    df_6M_Vol = df_W_interval.rolling(window=4 * 6).std() * (52 ** 0.5)
    df_1Y_Vol = df_W_interval.rolling(window=4 * 12).std() * (52 ** 0.5)
    df_2Y_Vol = df_W_interval.rolling(window=4 * 24).std() * (52 ** 0.5)
    df_3Y_Vol = df_W_interval.rolling(window=4 * 36).std() * (52 ** 0.5)


    # 날짜 인덱스를 포함하여 데이터프레임을 저장
    with pd.ExcelWriter(path_stat) as writer:
        df_1D_R.to_excel(writer, sheet_name='df_1D_R', index=True)
        df_W_R.to_excel(writer, sheet_name='df_W_R', index=True)
        df_1M_R.to_excel(writer, sheet_name='df_1M_R', index=True)
        df_3M_R.to_excel(writer, sheet_name='df_3M_R', index=True)
        df_6M_R.to_excel(writer, sheet_name='df_6M_R', index=True)
        df_1Y_R.to_excel(writer, sheet_name='df_1Y_R', index=True)
        df_2Y_R.to_excel(writer, sheet_name='df_2Y_R', index=True)
        df_3Y_R.to_excel(writer, sheet_name='df_3Y_R', index=True)
      
        df_1M_Vol.to_excel(writer, sheet_name='df_1M_V', index=True)
        df_3M_Vol.to_excel(writer, sheet_name='df_3M_V', index=True)
        df_6M_Vol.to_excel(writer, sheet_name='df_6M_V', index=True)
        df_1Y_Vol.to_excel(writer, sheet_name='df_1Y_V', index=True)
        df_2Y_Vol.to_excel(writer, sheet_name='df_2Y_V', index=True)
        df_3Y_Vol.to_excel(writer, sheet_name='df_3Y_V', index=True)


except FileNotFoundError:
    print("파일을 찾을 수 없습니다.")
except Exception as e:
    print("오류 발생:", e)






print(f"{path_stat}에 저장되었습니다.")

print(df.head)

