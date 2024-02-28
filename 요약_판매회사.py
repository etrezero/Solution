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
    path_요약 = r'C:/Covenant/data/요약.xlsx'
    df = pd.read_excel(path_요약, sheet_name='BOS3426', usecols=['일자', '펀드', '펀드명', '펀드구분', '종류현구분', '판매사명', '설정액'])

    df_filtered = df[(df['판매사명'] != '공통') & (df['종류현구분'] == '클래스펀드')].copy()
    df_filtered['펀드구분'] = df_filtered['펀드명'].apply(update_fund_category)
    df_filtered['(O) 포함 여부'] = df_filtered['펀드명'].apply(include_o)

    df_final = df_filtered[df_filtered['펀드구분'].isin(['TRP', '포커스'])]

    pivot_table = df_final.pivot_table(index='판매사명', columns=['펀드구분', '(O) 포함 여부'], values='설정액', aggfunc='sum', fill_value=0, margins=True)
    pivot_table_converted = convert_table_values(pivot_table)


     # 'All' 행 분리
    all_row = pivot_table_converted.loc['All'] if 'All' in pivot_table_converted.index else None
    pivot_table_converted = pivot_table_converted.drop('All', errors='ignore')

    # 첫 번째 열을 정의
    first_column = pivot_table_converted.columns[0]

    # first_column이 문자열과 숫자를 모두 포함하고 있다고 가정할 때,
    # 모든 값을 문자열로 변환
    pivot_table_converted = pivot_table_converted.astype(str)

    # 이제 정렬을 시도
    pivot_table_sorted = pivot_table_converted.sort_values(by=first_column, ascending=False)

    # 'All' 행을 2번째 위치에 추가
    if all_row is not None:
        pivot_table_sorted = pd.concat([pd.DataFrame(all_row).T, pivot_table_sorted])
    
    # df_final의 첫 번째 행의 '일자' 값 가져오기
    date = pd.to_datetime(df_final.iloc[0]['일자']).strftime('%Y-%m-%d') if not df_final.empty else '데이터 없음'

    # 피벗 테이블 헤더 상단에 날짜 출력
    pivot_table_sorted.columns = pd.MultiIndex.from_tuples([(date, '')] + list(pivot_table_sorted.columns)[1:])

    with pd.ExcelWriter(path_요약, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        pivot_table_sorted.to_excel(writer, sheet_name=f'판매회사{date}')
        df_final.to_excel(writer, sheet_name='DF', index=False)

    print(df_final.head())
    print(f'데이터가 {path_요약}의 판매회사{date}와 DF 시트에 저장되었습니다.')

if __name__ == "__main__":
    main()
