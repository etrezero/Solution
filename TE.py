import tradingeconomics as te
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yaml
import os


# 로그인 필수---------------------------------------------
# YAML 파일 경로 설정
yaml_path = 'C:\\Users\\서재영\\Documents\\Python Scripts\\koreainvestment-autotrade-main\\config.yaml'

# YAML 파일 로드
with open(yaml_path, encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)
TE_key = _cfg['TE_key']

te.login(TE_key)
# 로그인 필수---------------------------------------------




# te.getCalendarData()
# te.getIndicatorData(country=['mexico', 'sweden'], output_type='df')
# te.getMarketsData(marketsField = 'commodities')
# te.getMarketsBySymbol(symbols='aapl:us')
# te.getFinancialsData(symbol = 'aapl:us', output_type = 'df')






today     = datetime.now().strftime('%Y-%m-%d')
initDate  = '2014-01-01'  
endDate   = today  # 튜플이 아닌 문자열로 수정

country   = [
    'United States', 'Euro Area', 'Japan', 
    'South Korea', 'China', 
]

indicator = ['GDP Growth Rate', 'Inflation Rate', 'Interest Rate',
             'Unemployment Rate', 'Employment Cost Index','Wage Growth',
             'Manufacturing PMI', 'Non Manufacturing PMI',
             'Balance of Trade', 
]


# To get historical data by specific country, indicator, start date and end date
mydata = te.getHistoricalData(country=country,  indicator=indicator, initDate=initDate, endDate=endDate, output_type='df')
print(mydata.head)
print("===============================================================================================================")





path_TE = r'C:\Users\서재영\Documents\Python Scripts\data\TE_data\TE.json'

# 파일이 이미 존재하는지 확인합니다.
if os.path.exists(path_TE):
    mode = 'a'  # 파일이 이미 존재하면 추가 모드로 열기
else:
    mode = 'w'  # 파일이 없으면 쓰기 모드로 열기

# To get your data into a json file
df = pd.DataFrame(mydata)
with open(path_TE, mode) as f:
    df.to_json(f, orient='records', lines=True)

