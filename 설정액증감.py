import pandas as pd

path_판매회사 = r'C:/Covenant/data/BOS3426.xlsx'

#펀드명 필터링
# def update_fund_category(fund_name):
#     if '포커스' in fund_name:
#         return '포커스'
#     elif 'TDF' in fund_name or 'TIF' in fund_name:
#         return 'TRP'
#     return '기타'

# def include_o(fund_name):
#     return '(O) 포함' if '(O)' in fund_name else '(O) 미포함'

def convert_table_values(table):
    """피벗 테이블의 값을 억 단위로"""
    for col in table.columns:
        table[col] = table[col].apply(lambda x: '' if x == 0 else round(x / 100000000, 0))
    return table

def main():
    
    df = pd.read_excel(path_판매회사, sheet_name='펀드판매사정보', 
                       usecols=['일자', '펀드', '펀드명', 
                                '펀드구분', '펀드유형', '모자구분', 
                                '종류현구분', '판매사명', '설정액'
                                ])

    df_final = df[(df['모자구분'] != '모펀드') & ((df['종류현구분'] == '클래스펀드') | (df['종류현구분'] == '일반펀드'))].copy()
    # df_final = df_filtered[df_filtered['펀드구분'].isin(['TRP', '포커스'])]

    pivot_table = df_final.pivot_table(index='판매사명',  values='설정액', aggfunc='sum', fill_value=0, margins=True)   #columns=[''],
    pivot_table_converted = convert_table_values(pivot_table)


    # 첫 번째 열을 기준으로 내림차순 정렬
    first_column = pivot_table_converted.columns[0]
    pivot_table_sorted = pivot_table_converted.sort_values(by=first_column, ascending=False, )


    
    # df_final의 첫 번째 행의 '일자' 값을 YYYY-MM-DD 형식으로 가져오기
    last_column_first_row = pd.to_datetime(df_final.iloc[0]['일자']).strftime('%Y-%m-%d')


    with pd.ExcelWriter(path_판매회사, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        pivot_table_sorted.to_excel(writer, sheet_name=f'합계{last_column_first_row}')
        df_final.to_excel(writer, sheet_name=last_column_first_row)

    print(df_final.head)
    print(f"저장 완료")


    # print(df_final.tail)
    # 헤더 출력
    # print("피벗 테이블 헤더:")
    # print(pivot_table_sorted.head)

if __name__ == "__main__":
    main()
