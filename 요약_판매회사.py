import pandas as pd

def update_fund_category(fund_name):
    if '포커스' in fund_name:
        return '포커스'
    elif 'TDF' in fund_name or 'TIF' in fund_name:
        return 'TRP'
    return '기타'

def include_o(fund_name):
    return '(O) 포함' if '(O)' in fund_name else '(O) 미포함'

def convert_table_values(table):
    """피벗 테이블의 값에 대해 변환을 적용하는 함수"""
    for col in table.columns:
        table[col] = table[col].apply(lambda x: '' if x == 0 else round(x / 100000000, 0))
    return table

def main():
    path_판매회사 = r'C:/Covenant/data/요약.xlsx'
    df = pd.read_excel(path_판매회사, sheet_name='BOS3426', usecols=['일자', '펀드', '펀드명', '펀드구분', '종류현구분', '판매사명', '설정액'])

    df_filtered = df[(df['판매사명'] != '공통') & (df['종류현구분'] == '클래스펀드')].copy()
    df_filtered['펀드구분'] = df_filtered['펀드명'].apply(update_fund_category)
    df_filtered['(O) 포함 여부'] = df_filtered['펀드명'].apply(include_o)

    df_final = df_filtered[df_filtered['펀드구분'].isin(['TRP', '포커스'])]

    pivot_table = df_final.pivot_table(index='판매사명', columns=['펀드구분', '(O) 포함 여부'], values='설정액', aggfunc='sum', fill_value=0, margins=True)
    pivot_table_converted = convert_table_values(pivot_table)


    # 첫 번째 열을 기준으로 내림차순 정렬
    first_column = pivot_table_converted.columns[0]
    pivot_table_sorted = pivot_table_converted.sort_values(by=first_column, ascending=False)


    # df_final의 첫 번째 행의 '일자' 값 가져오기
    # df_final의 첫 번째 행의 '일자' 값을 YYYY-MM-DD 형식으로 가져오기
    date = pd.to_datetime(df_final.iloc[0]['일자']).strftime('%Y-%m-%d')


    # 피벗 테이블 헤더 상단에 날짜 출력
    pivot_table_sorted.columns = pd.MultiIndex.from_tuples([(date, '')] 
                                + list(pivot_table_sorted.columns)[1:])


    with pd.ExcelWriter(path_판매회사, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        pivot_table_sorted.to_excel(writer, sheet_name=f'합계{date}')
        df_final.to_excel(writer, sheet_name='DF')

    print(df_final.head)
    print(f"요약파일_판매회사 시트에 저장 완료")


    # print(df_final.tail)
    # 헤더 출력
    # print("피벗 테이블 헤더:")
    # print(pivot_table_sorted.head)

if __name__ == "__main__":
    main()
