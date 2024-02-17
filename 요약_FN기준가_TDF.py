import pyautogui as PAUG
import win32gui
import win32com.client
import pygetwindow as PGW
import os
import subprocess
from datetime import datetime
import time
import pandas as pd
import pyperclip
import keyboard
from PIL import ImageGrab, Image

from datetime import timedelta

today_date = datetime.now()
# 파일 경로
save_path = 'C:/Covenant/data/요약.xlsx'

# 파일 읽어오기
try:
    df = pd.read_excel(save_path, sheet_name='FN기준가')
    
    # 첫번째 열 마지막 행의 값을 T_7 변수에 할당
    T_7 = df.iloc[-1, 0]  # 첫번째 열은 0번째 열이므로, iloc를 사용하여 인덱스 -1의 값 가져옴
    print("T_7 변수 값:", T_7)  # 변수 값 출력
except FileNotFoundError:
    print("파일을 찾을 수 없습니다.")
except Exception as e:
    print("오류 발생")
    
# T_7 = today_date - timedelta(days=7)


T0 = today_date.strftime("%Y%m%d")
T_77 = T_7.strftime("%Y%m%d")

print(T0)

# #--------------------------------------------
# def job():
# subprocess.run(["python", "C:\Users\USER\Desktop\Excel Project\231108 데일리 자동화.py"])

# # 매일 오전 2시에 job 함수를 실행
# schedule.every().day.at("18:53").do(job)

# while True:
#     schedule.run_pending()
#     time.sleep(1)
# #--------------------------------------------

#윈도우창 최대화 : PAUG.hotkey('win', 'up')
# active_window = win32gui.GetActiveWindow()
# foreground_window_handle = win32gui.GetForegroundWindow()



# ----------------시작단계 - 키보드 영어로 시작---------
def set_keyboard_layout(layout):
    try:
        # PowerShell 스크립트를 사용하여 키보드 레이아웃을 변경
        powershell_script = f'''
        $languageList = New-WinUserLanguageList {layout}
        Set-WinUserLanguageList $languageList -Force
        '''

        subprocess.run(["powershell", "-Command", powershell_script], check=True, shell=True)
        print(f"Keyboard layout set to {layout}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

def process_a_key():
    # Code to handle the next process when 'a' is pressed
    print("Processing 'a' key")

def on_key_event(e):
    if e.event_type == keyboard.KEY_DOWN:
        if e.name == 'a':
            process_a_key()
        elif e.name == 'ㅁ':
            # 'ㅁ'이 입력되면 키보드 레이아웃을 영어로 변경
            set_keyboard_layout('en-US')

# 키보드 이벤트 리스너 등록
keyboard.hook(on_key_event)
# ----------------시작단계 - 키보드 영어로 시작---------


# --------------------------------------
# FN스펙트럼 로그인

PAUG.hotkey("win","r")
FN스펙트럼 = "C:\\FnGuide\\FnSpectrum\\FnSpectrum.exe"
pyperclip.copy('C:\\FnGuide\\FnSpectrum\\FnSpectrum.exe')
time.sleep(1)
PAUG.hotkey('ctrl','v')
PAUG.press("enter")
time.sleep(5)



w_title = "Log In"  # 방금 실행한 창의 타이틀로 변경
window = PGW.getWindowsWithTitle(w_title)

if window:
    window[0].activate()  # 첫 번째 창을 활성화
    time.sleep(0.1)
    PAUG.press("enter")
    time.sleep(0.1)
    PAUG.press("enter")
    time.sleep(2)
    
else:
    print("FN스펙트럼 로그인 창을 찾을 수 없습니다.")




# -------------------------------------------------
#마우스 위치 좌표
mouse_x, mouse_y = PAUG.position()

# Print the coordinates
print(f"현재 마우스 좌표: ({mouse_x}, {mouse_y})")

# 스크린 위치 정의
screen_width = win32gui.GetWindowRect(win32gui.GetDesktopWindow())[2]
screen_height = win32gui.GetWindowRect(win32gui.GetDesktopWindow())[3]
center_x = screen_width / 2
center_y = screen_height / 2

# 화면 중앙의 좌표를 출력합니다.
print(f"화면 중앙 좌표: ({center_x}, {center_y})")
# -------------------------------------------------


펀드Tools좌표 = (52, 997)
Time시리즈좌표 = (85, 362)
Time돋보기좌표 = (204, 126)
Myport좌표 = (365, 501)
아이템좌표 = (769, 503)
엑셀다운좌표 = (597, 602)
기간좌표1 = (365, 544)
기간좌표2 = (449, 550)

time.sleep(2)
PAUG.click(펀드Tools좌표[0], 펀드Tools좌표[1], clicks=2)
time.sleep(1)
PAUG.click(Time시리즈좌표[0], Time시리즈좌표[1], clicks=2)
time.sleep(1)
PAUG.click(Time돋보기좌표)

PAUG.click(Myport좌표[0], Myport좌표[1], clicks=2)
time.sleep(1)
pyperclip.copy('운용펀드')
PAUG.hotkey('ctrl','a')
PAUG.hotkey('ctrl','v')
PAUG.press('enter')

PAUG.click(아이템좌표[0], 아이템좌표[1], clicks=2)
time.sleep(1)
pyperclip.copy('수정기준가')
PAUG.hotkey('ctrl','a')
PAUG.hotkey('ctrl','v')
PAUG.press('enter')

PAUG.click(기간좌표1[0], 기간좌표1[1], clicks=2)
PAUG.typewrite(T_77, interval=0.1)
time.sleep(1)

PAUG.click(기간좌표2[0], 기간좌표2[1], clicks=2)
PAUG.typewrite(T0, interval=0.1)
time.sleep(1)

PAUG.click(엑셀다운좌표)
time.sleep(30)

excel = win32com.client.GetActiveObject("Excel.Application")
excel_hwnd = win32gui.FindWindow(None, excel.Caption)
workbook = excel.ActiveWorkbook

if workbook is not None:
        today = datetime.today().strftime('%Y%m%d')
        file_FN기준가 = f'FN기준가.xlsx'          #file_FN기준가 = f'FN기준가 {today}.xlsx'
        pathFN기준가 = os.path.join('C:\Covenant\data\To_요약', file_FN기준가)

        # 시트 이름을 "FN기준가"로 변경
        workbook.Sheets(1).Name = "FN기준가"

        # FN기준가를 새로운 파일로 저장
        workbook.SaveAs(pathFN기준가)
        time.sleep(3)





        # Workbook_FN기준가 열기
        workbook_FN기준가 = excel.Workbooks.Open(pathFN기준가)
        time.sleep(0.1)
        worksheet_FN기준가 = workbook_FN기준가.Sheets('FN기준가')


        path요약 = r'C:\Covenant\data\요약.xlsx'
        workbook_요약 = excel.Workbooks.Open(path요약)

        worksheet_요약_FN기준가 = workbook_요약.Sheets('FN기준가')
        worksheet_요약_FN기준가.Activate()
        time.sleep(0.1)



        # 현재 사용 중인 행의 개수 파악
        max_row_요약 = worksheet_요약_FN기준가.UsedRange.Rows.Count
        print(f"최대행은 {max_row_요약}행 입니다")

        # A열의 최대 행의 값 가져오기
        max_value_요약 = worksheet_요약_FN기준가.Cells(max_row_요약, 1).Value

        # 복사할 행 선택
        for row_num in range(14, worksheet_FN기준가.UsedRange.Rows.Count + 1):
            row_data = []  # 각 행마다 초기화

            cell_value = worksheet_FN기준가.Cells(row_num, 1).Value

            # 현재 행의 A열 값이 최대 행의 A열 값보다 큰 경우 데이터를 복사
            if isinstance(cell_value, str) or str(cell_value) > str(max_value_요약):
                # 데이터를 복사
                row_data = [worksheet_FN기준가.Cells(row_num, col_num).Value for col_num in range(1, worksheet_FN기준가.UsedRange.Columns.Count + 1)]

                # 붙여넣을 행 선택 
                next_row = max_row_요약 + 1  # + 1로 수정
                

                # 데이터 복사하여 붙여넣기
                for col_num, value in enumerate(row_data, start=1):
                    worksheet_요약_FN기준가.Cells(next_row, col_num).Value = value
                
                print(f"행 {row_num}을 복사하여 요약_FN기준가 {next_row}행에 붙여넣었습니다.")
                max_row_요약 += 1
                
            else:
                
                print(f"행 {row_num}은 복사하지 않았습니다. 조건: {cell_value} > {max_value_요약}")

        time.sleep(0.3)
        workbook_FN기준가.Close()

        # Excel 애플리케이션 종료
        excel.Quit()

        time.sleep(1)
        PAUG.hotkey('alt','F4')
        time.sleep(1)
        PAUG.hotkey('alt','F4')
        time.sleep(1)
        PAUG.press("enter")

        print("FN기준가 저장완료")
else:
        print('No active workbook found.')




