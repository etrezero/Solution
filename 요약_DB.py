import pyautogui as PAUG
import win32gui
import win32con
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
from openpyxl import load_workbook
from openpyxl.utils.cell import get_column_letter
from datetime import timedelta
import ctypes
import openpyxl
import schedule
import xlrd
import xlwings
import pywin

ID = "T08186"
PW = "T202301!"

FOS = "C:\\KBFund\\Fo\\REAL\\Bin\\KBFundFO.exe"
BOS = r'"C:\Program Files (x86)\TOBESOFT\XPLATFORM\9.2.1\XPlatform.exe" -K "KBFundBo" -X "https://fund.kbstar.com:2185/kbfundbo/Resource/KBFundBo.xadl"'
MOS = r'"C:\Program Files (x86)\TOBESOFT\XPLATFORM\9.2.1\XPlatform.exe" -K "KBFundMo" -X "https://fund.kbstar.com:2195/kbfundmo/Resource/KBFundMo.xadl"'



today_date = datetime.now()
T0 = today_date.strftime("%Y%m%d")

T_1 = today_date - timedelta(days=1)
T_1D = T_1.strftime("%Y%m%d")

# #--------------------------------------------
# def job():
# subprocess.run(["python", "C:\Users\USER\Desktop\Excel Project\231108 데일리 자동화.py"])

# # 매일 오전 7:50시에 job 함수를 실행
# schedule.every().day.at("07:50").do(job)

# while True:
#     schedule.run_pending()
#     time.sleep(1)
# #--------------------------------------------



#윈도우창 최대화 : PAUG.hotkey('win', 'up')
# active_window = win32gui.GetActiveWindow()
# foreground_window_handle = win32gui.GetForegroundWindow()

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




센터좌표 = (960, 540)
펀드종합좌표 = (1133.5, 57)
자금현황좌표 = (399, 88)

자금총괄좌표 = (415, 176.5)
총괄좌표 = (77, 148)

조회좌표 = (1228, 239.5)

설정해지좌표 = (276.5, 151.5)
엑셀저장좌표 = (1059, 705) #엑셀 CSV로 저장
######## 엑셀저장좌표 = (1059, 725) #새 엑셀파일(XLS)로 붙여넣기 XXXXXXXXXXXXXXXXX
MOS화면번호좌표 = (1005, 43)
MOS돋보기좌표 = (153, 213)
부서좌표 = (320, 200)
부서좌표2 = (320, 216)
부서좌표3 = (320, 482)
네모칸좌표 = (193, 409)
확인좌표 = (616, 885)
MOS조회좌표 = (1165, 128)




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

# # 키보드 이벤트 리스너 등록
            
# keyboard.hook(on_key_event)
# ----------------시작단계 - 키보드 영어로 시작---------




# FOS 실행/로그인
# -------------------------------------------
PAUG.hotkey("win","r")
pyperclip.copy(FOS)
PAUG.hotkey('ctrl','v')
time.sleep(0.5)
PAUG.press("enter")
time.sleep(3)


pyperclip.copy(PW)
PAUG.hotkey('ctrl','v')
time.sleep(0.1)
PAUG.press("enter")
time.sleep(0.5)
PAUG.press("enter")
time.sleep(3)

# -------------------------------------------


def maximize_window_by_name(window_name):
    # 모든 윈도우 핸들을 열거
    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and window_name in win32gui.GetWindowText(hwnd):
            # 창이 보이고 이름이 일치하는 경우 창을 최대화
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
    
    win32gui.EnumWindows(callback, None)

# 특정 프로그램 창 또는 작업 창의 이름을 지정합니다.
window_name = "FundStar FOS"

# 함수를 호출하여 창을 최대화합니다.
maximize_window_by_name(window_name)








# 화면을 엑셀로 저장

PAUG.click(펀드종합좌표)
time.sleep(0.5)

PAUG.click(자금현황좌표)
time.sleep(0.5)

PAUG.click(자금총괄좌표)
time.sleep(0.5)

PAUG.click(총괄좌표)

time.sleep(0.5)


PAUG.click(조회좌표)     
time.sleep(7)

PAUG.click(960, 540, button='right')

PAUG.click(엑셀저장좌표)
time.sleep(3)


GridBox = r'C:\KBFund\Fo\REAL\DataSource\GridBox.csv'
df_GridBox = pd.read_csv(GridBox, encoding='euc-kr')

# # # Excel 파일로 저장
today = datetime.today().strftime('%Y%m%d')
file_자금총괄 = f'자금총괄.xlsx'  #file_자금총괄 = f'자금총괄 {today}.xlsx'
path자금총괄 = os.path.join('C:\Covenant\data\To_요약', file_자금총괄)

# # 열려 있던 GiredBox 창 찾아-활성화해-닫기
def bring_excel_to_front(excel):

    try:
        excel_hwnd = win32gui.FindWindow(None, excel.Caption)
        win32gui.ShowWindow(excel_hwnd, win32con.SW_RESTORE)  # Restore if minimized
        win32gui.SetForegroundWindow(excel_hwnd)

    except Exception as e:
        print("Error bringing Excel to the front:", e)

excel = win32com.client.GetActiveObject("Excel.Application")
bring_excel_to_front(excel)



time.sleep(1)
df_GridBox.to_excel(path자금총괄, sheet_name='자금총괄', index=False)
time.sleep(3)
excel.Quit()
time.sleep(1)



# #설정해지 조회
PAUG.click(설정해지좌표)       
time.sleep(0.5)

PAUG.click(조회좌표)
time.sleep(7)

PAUG.click(960, 540, button='right')
time.sleep(1)

PAUG.click(엑셀저장좌표)
time.sleep(1)


# # -------GridBox CSV 파일 읽고 설정해지 저장------------------------------
GridBox = r'C:\KBFund\Fo\REAL\DataSource\GridBox.csv'
df_GridBox = pd.read_csv(GridBox, encoding='euc-kr')


# # # 설정해지 파일 경로
today = datetime.today().strftime('%Y%m%d')
file_설정해지 = f'설정해지.xlsx'  #file_설정해지 = f'설정해지 {today}.xlsx'
path설정해지 = os.path.join('C:\Covenant\data\To_요약', file_설정해지)

# 열려 있던 GiredBox 창 찾아-활성화해-닫기
def bring_excel_to_front(excel):

    try:
        excel_hwnd = win32gui.FindWindow(None, excel.Caption)
        win32gui.ShowWindow(excel_hwnd, win32con.SW_RESTORE)  # Restore if minimized
        win32gui.SetForegroundWindow(excel_hwnd)

    except Exception as e:
        print("Error bringing Excel to the front:", e)

excel = win32com.client.GetActiveObject("Excel.Application")
bring_excel_to_front(excel)

df_GridBox.to_excel(path설정해지, sheet_name='설정해지', index=False)
time.sleep(3)
excel.Quit()
time.sleep(1)
# -------GridBox CSV 파일 읽고 설정해지 저장 <완료>------------------------------





# 파일 경로 및 이름 설정
today = datetime.today().strftime('%Y%m%d')
file_설정해지 = f'설정해지.xlsx'
path설정해지 = os.path.join('C:\Covenant\data\To_요약', file_설정해지)

file_자금총괄 = f'자금총괄.xlsx'
path자금총괄 = os.path.join('C:\Covenant\data\To_요약', file_자금총괄)

file_수정기준가 = f'수정기준가.xlsx'
path수정기준가 = os.path.join('C:\Covenant\data\To_요약', file_수정기준가)



# # -------자금총괄 파일 불러오기<시작>----------

# # Excel 인스턴스 생성
excel = win32com.client.Dispatch("Excel.Application")
excel.Visible = True  # Excel 창을 보이게 함

# # Workbook_자금총괄 열기
workbook_자금총괄 = excel.Workbooks.Open(path자금총괄)
time.sleep(0.5)
worksheet_자금총괄 = workbook_자금총괄.Sheets('자금총괄')
worksheet_자금총괄.Activate()
time.sleep(0.5)
PAUG.hotkey('ctrl', 'a')
time.sleep(0.5)
PAUG.hotkey('ctrl', 'c')


path요약 = r'C:\Covenant\data\요약.xlsx'
workbook_요약 = excel.Workbooks.Open(path요약)
time.sleep(0.5)

worksheet_요약_자금총괄 = workbook_요약.Sheets('자금총괄')
worksheet_요약_자금총괄.Activate()

time.sleep(0.5)

PAUG.hotkey('ctrl', 'home')
time.sleep(0.5)
PAUG.hotkey('ctrl', 'v')
time.sleep(0.5)

range_요약_자금총괄 = worksheet_요약_자금총괄.Range(worksheet_요약_자금총괄.Cells(3, 4), worksheet_요약_자금총괄.Cells(worksheet_요약_자금총괄.UsedRange.Rows.Count, worksheet_요약_자금총괄.UsedRange.Columns.Count))
range_요약_자금총괄.Value = range_요약_자금총괄.Value

print("요약_자금총괄 저장 완료")
# -----------------<자금총괄 저장 완료>-----------








# -------설정해지 파일 불러오기<시작>----------

# Excel 인스턴스 생성
excel = win32com.client.Dispatch("Excel.Application")
excel.Visible = True  # Excel 창을 보이게 함

# Workbook_설정해지 열기
workbook_설정해지 = excel.Workbooks.Open(path설정해지)
time.sleep(0.1)
worksheet_설정해지 = workbook_설정해지.Sheets('설정해지')

worksheet_설정해지.Activate()
time.sleep(0.1)
PAUG.hotkey('ctrl', 'a')
time.sleep(0.1)
PAUG.hotkey('ctrl', 'c')


worksheet_요약_설정해지 = workbook_요약.Sheets('설정해지')
worksheet_요약_설정해지.Activate()
time.sleep(0.1)
PAUG.hotkey('ctrl', 'home')
time.sleep(0.1)
PAUG.hotkey('ctrl', 'v')
time.sleep(0.1)

range_요약_설정해지 = worksheet_요약_설정해지.Range(worksheet_요약_설정해지.Cells(3, 4), worksheet_요약_설정해지.Cells(worksheet_요약_설정해지.UsedRange.Rows.Count, worksheet_설정해지.UsedRange.Columns.Count))
range_요약_설정해지.Value = range_요약_설정해지.Value

print("요약_설정해지 저장 완료")
# ---------------------------<설정해지 저장 완료>-----------


worksheet_자금총괄.Activate()
time.sleep(0.3)
try : 
        PAUG.hotkey('alt','F4')
        time.sleep(0.3)
except :

        PAUG.press('enter')


worksheet_설정해지.Activate()
time.sleep(0.3)
try : 
       PAUG.hotkey('alt','F4')

       time.sleep(0.1)
       PAUG.press('enter')
except :
        PAUG.press('enter')
        time.sleep(0.3)





# # # MOS 실행/로그인---------------------------------------

MOS화면번호좌표 = (1005, 38)
MOS돋보기좌표 = (153, 213)
부서좌표 = (287, 199)
부서좌표2 = (320, 216)
부서좌표3 = (268, 487)
네모칸좌표 = (193, 409)
확인좌표 = (616, 885)
MOS조회좌표 = (1165, 128)
센터좌표 = (960, 540)

# ----------------------/---------------------
PAUG.hotkey("win", "r")
time.sleep(0.5)

pyperclip.copy(MOS)
PAUG.hotkey("ctrl", "v")
PAUG.press("Enter")
time.sleep(5)

pyperclip.copy(PW)
PAUG.hotkey('ctrl','v')
time.sleep(0.5)

PAUG.press("enter")
time.sleep(8)


PAUG.click(MOS화면번호좌표)
pyperclip.copy("4110")
PAUG.hotkey('ctrl','v')
time.sleep(0.5)
PAUG.press("enter")
time.sleep(3)

PAUG.click(MOS돋보기좌표)
time.sleep(1)

PAUG.click(부서좌표)
time.sleep(1)

PAUG.press("Tab")
time.sleep(1)

# 키보드 이벤트 리스너 등록
keyboard.hook(on_key_event)
time.sleep(2)

pyperclip.copy("솔루션운용부")
PAUG.hotkey('ctrl','v')
time.sleep(1)
PAUG.click(350,484)

time.sleep(1)
PAUG.hotkey('alt','F4')

PAUG.click(192,409)
time.sleep(1)

PAUG.click(618,885)   #확인좌표
time.sleep(3)

PAUG.click(1167, 128)  #조회 좌표
time.sleep(10)

PAUG.click(960, 540, button='right')
time.sleep(1)

PAUG.click(990, 555)    #엑셀다운로드
time.sleep(7)


def bring_excel_to_front(excel):

    try:
        excel_hwnd = win32gui.FindWindow(None, excel.Caption)
        win32gui.ShowWindow(excel_hwnd, win32con.SW_RESTORE)  # Restore if minimized
        win32gui.SetForegroundWindow(excel_hwnd)

    except Exception as e:
        print("Error bringing Excel to the front:", e)

excel = win32com.client.GetActiveObject("Excel.Application")
bring_excel_to_front(excel)



time.sleep(0.5)
PAUG.hotkey('ctrl','a')
time.sleep(0.5)
PAUG.hotkey('ctrl','c')



def bring_excel_sheet_to_front(excel, workbook_name, sheet_name):
    try:
        # Excel 초기화
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = True  # Excel 창을 보이게 함

        # 현재 활성화된 워크북 가져오기
        workbook = excel.Workbooks(workbook_name)

        # 워크북이 존재하고, 지정된 시트 이름이 있는 경우
        if workbook is not None and sheet_name in [sheet.Name for sheet in workbook.Sheets]:
            excel_hwnd = win32gui.FindWindow(None, excel.Caption)
            win32gui.ShowWindow(excel_hwnd, win32con.SW_RESTORE)  # 최소화되어 있다면 복원
            win32gui.SetForegroundWindow(excel_hwnd)

            # 시트를 찾아서 최상위로 가져오기
            for sheet in workbook.Sheets:
                if sheet.Name == sheet_name:
                    sheet.Activate()
                    break

    except Exception as e:
        print(f"{sheet_name} 시트를 최상위로 가져오는 동안 오류 발생:", e)

# Excel 초기화와 보이게 설정하는 코드를 함수 내에 넣어주기
bring_excel_sheet_to_front(None, '요약', 'MOS4110')


PAUG.hotkey('ctrl','a')
time.sleep(0.5)
PAUG.hotkey('ctrl','v')
time.sleep(0.5)

print("MOS 저장완료")
# # ---<MOS 완료>--------------------------------







# # 요약 제외 Excel 닫기

# def close_all_except_요약_workbook(excel, workbook_name):
#     try:
#         # Excel 인스턴스가 이미 초기화되어 있다고 가정
#         excel.Visible = True  # Excel 창을 보이게 함
#         excel.DisplayAlerts = False  # 경고창 표시 안 함

#         # 열려 있는 모든 워크북 순회
#         for wb in list(excel.Workbooks):
#             # 지정된 워크북 이름이 아닌 경우 닫기
#             if wb.Name != workbook_name:
#                 wb.Close(SaveChanges=False)  # 변경사항을 저장하지 않고 닫음

#     except Exception as e:
#         print(f"오류 발생: {e}")

# # Excel Application 객체 가져오기
# excel = win32com.client.GetActiveObject("Excel.Application")

# # '요약' 워크북을 제외한 모든 워크북을 닫기
# close_all_except_요약_workbook(excel, '요약.xlsx')







# # ---<BOS>--------------------------------
BOS화면번호좌표 = (1000, 41)
BOS돋보기좌표 = (174, 203)
BOS부서좌표 = (389, 203)
BOS부서좌표2 = (389, 217)
BOS부서좌표3 = (420, 485)
BOS네모칸좌표 = (260, 384)
BOS확인좌표 = (616, 888)
BOS조회좌표 = (1168, 130)
BOS한줄엑셀좌표 = (1128, 270)


PAUG.hotkey("win", "r")
time.sleep(0.5)

pyperclip.copy(BOS)
PAUG.hotkey("ctrl", "v")
time.sleep(0.5)
PAUG.press("Enter")
time.sleep(7)

pyperclip.copy(PW)
PAUG.hotkey('ctrl','v')

time.sleep(0.5)
PAUG.press("enter")
time.sleep(8)




# def maximize_window_by_name(window_name):
#     # 모든 윈도우 핸들을 열거
#     def callback(hwnd, _):
#         if win32gui.IsWindowVisible(hwnd) and window_name in win32gui.GetWindowText(hwnd):
#             # 창이 보이고 이름이 일치하는 경우 창을 최대화
#             win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
    
#     win32gui.EnumWindows(callback, None)

# # 특정 프로그램 창 또는 작업 창의 이름을 지정합니다.
# window_name = "한국투자신탁운용"

# # 함수를 호출하여 창을 최대화합니다.
# maximize_window_by_name(window_name)




PAUG.click(BOS화면번호좌표)

time.sleep(1)
pyperclip.copy("8001")
PAUG.hotkey('ctrl','v')
time.sleep(1)
PAUG.press("enter")
time.sleep(3)

PAUG.press("Tab")
time.sleep(1)

pyperclip.copy(T_1D)
PAUG.hotkey('ctrl','v')
time.sleep(0.5)

PAUG.click(BOS돋보기좌표)

time.sleep(2)

PAUG.click(BOS부서좌표)
time.sleep(2)   

PAUG.press("Tab")
time.sleep(1)

# 키보드 이벤트 리스너 등록

# keyboard.hook(on_key_event)
# time.sleep(0.5)

pyperclip.copy("솔루션운용부")
PAUG.hotkey('ctrl','v')
time.sleep(0.5)
PAUG.click(BOS부서좌표3)
time.sleep(0.5)
PAUG.hotkey('alt','F4')

PAUG.click(BOS네모칸좌표)
time.sleep(0.5)

PAUG.click(BOS확인좌표)
time.sleep(1)

PAUG.click(BOS조회좌표)

time.sleep(7)

PAUG.click(BOS한줄엑셀좌표)
time.sleep(10)



def bring_excel_to_front():
    try:
        # Excel 초기화
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = True  # Excel 창을 보이게 함

        # 현재 활성화된 워크북 가져오기
        workbook = excel.ActiveWorkbook

        # 워크북이 존재하고, 저장되지 않았다면
        if workbook is not None and not workbook.Saved:
            # Excel 창 핸들 가져오기
            excel_hwnd = win32gui.FindWindow(None, excel.Caption)
            # 최소화되어 있다면 복원
            win32gui.ShowWindow(excel_hwnd, win32con.SW_RESTORE)
            # 창을 최상위로 가져오기
            win32gui.SetForegroundWindow(excel_hwnd)

    except Exception as e:
        print("Excel을 최상위로 가져오는 동안 오류 발생:", e)

# Excel 창을 최상위로 가져오기
bring_excel_to_front()


time.sleep(0.5)
PAUG.hotkey('ctrl','a')
time.sleep(0.5)
PAUG.hotkey('ctrl','c')

time.sleep(0.5)



def bring_excel_sheet_to_front(excel, workbook_name, sheet_name):
    try:
        # Excel 초기화
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = True  # Excel 창을 보이게 함

        # 현재 활성화된 워크북 가져오기
        workbook = excel.Workbooks(workbook_name)

        # 워크북이 존재하고, 지정된 시트 이름이 있는 경우
        if workbook is not None and sheet_name in [sheet.Name for sheet in workbook.Sheets]:
            excel_hwnd = win32gui.FindWindow(None, excel.Caption)
            win32gui.ShowWindow(excel_hwnd, win32con.SW_RESTORE)  # 최소화되어 있다면 복원
            win32gui.SetForegroundWindow(excel_hwnd)

            # 시트를 찾아서 최상위로 가져오기
            for sheet in workbook.Sheets:
                if sheet.Name == sheet_name:
                    sheet.Activate()
                    break

    except Exception as e:
        print(f"{sheet_name} 시트를 최상위로 가져오는 동안 오류 발생:", e)

# Excel 초기화와 보이게 설정하는 코드를 함수 내에 넣어주기
bring_excel_sheet_to_front(None, '요약', '통합명세부')


time.sleep(2)
PAUG.hotkey('ctrl','a')

time.sleep(0.1)
PAUG.hotkey('ctrl','v')


time.sleep(0.1)
PAUG.press('Enter')










# #BOS 3426 펀드판매사 정보
# # ---<BOS>--------------------------------
BOS화면번호좌표 = (1000, 41)
BOS돋보기좌표_3426 = (150, 203)
BOS부서좌표 = (389, 203)
BOS부서좌표2 = (389, 217)
BOS부서좌표3 = (420, 485)
BOS네모칸좌표 = (260, 384)
BOS확인좌표 = (616, 888)
BOS조회좌표 = (1168, 130)
BOS한줄엑셀좌표 = (1128, 270)


PAUG.click(BOS화면번호좌표)

time.sleep(1)
pyperclip.copy("3426")
PAUG.hotkey('ctrl','v')
time.sleep(1)
PAUG.press("enter")
time.sleep(3)


PAUG.click(123, 176)
time.sleep(1)
pyperclip.copy(T_1D)
PAUG.hotkey('ctrl','v')
time.sleep(0.5)

PAUG.click(258, 176)
time.sleep(1)
pyperclip.copy(T_1D)
PAUG.hotkey('ctrl','v')
time.sleep(0.5)

PAUG.click(BOS돋보기좌표_3426)

time.sleep(2)

PAUG.click(BOS부서좌표)
time.sleep(2)   

PAUG.press("Tab")
time.sleep(1)


pyperclip.copy("솔루션운용부")
PAUG.hotkey('ctrl','v')
time.sleep(0.5)
PAUG.click(BOS부서좌표3)
time.sleep(0.5)
PAUG.hotkey('alt','F4')

PAUG.click(BOS네모칸좌표)
time.sleep(0.5)

PAUG.click(620,887)
time.sleep(1)

PAUG.click(BOS조회좌표)
time.sleep(5)

PAUG.click(960, 540, button='right')
time.sleep(1)

PAUG.click(990, 555)    #엑셀다운로드
time.sleep(5)



def bring_excel_to_front():
    try:
        # Excel 초기화
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = True  # Excel 창을 보이게 함

        # 현재 활성화된 워크북 가져오기
        workbook = excel.ActiveWorkbook

        # 워크북이 존재하고, 저장되지 않았다면
        if workbook is not None and not workbook.Saved:
            # Excel 창 핸들 가져오기
            excel_hwnd = win32gui.FindWindow(None, excel.Caption)
            # 최소화되어 있다면 복원
            win32gui.ShowWindow(excel_hwnd, win32con.SW_RESTORE)
            # 창을 최상위로 가져오기
            win32gui.SetForegroundWindow(excel_hwnd)

    except Exception as e:
        print("Excel을 최상위로 가져오는 동안 오류 발생:", e)

# Excel 창을 최상위로 가져오기
bring_excel_to_front()


time.sleep(0.5)
PAUG.hotkey('ctrl','a')
time.sleep(1)
PAUG.hotkey('ctrl','c')
time.sleep(0.5)


def bring_excel_sheet_to_front(excel, workbook_name, sheet_name):
    try:
        # Excel 초기화
        excel = win32com.client.Dispatch("Excel.Application")

        # 현재 활성화된 워크북 가져오기
        workbook = excel.Workbooks(workbook_name)
        excel_hwnd = win32gui.FindWindow(None, excel.Caption)
        win32gui.ShowWindow(excel_hwnd, win32con.SW_RESTORE)  # 최소화되어 있다면 복원
        win32gui.SetForegroundWindow(excel_hwnd)

    
        # 시트를 찾아서 최상위로 가져오기
        for sheet in workbook.Sheets:
            if sheet.Name == sheet_name:
                sheet.Activate()
                break

    except Exception as e:
        print(f"{sheet_name} 시트를 최상위로 가져오는 동안 오류 발생:", e)

# Excel 초기화와 보이게 설정하는 코드를 함수 내에 넣어주기
bring_excel_sheet_to_front(None, '요약', 'BOS3426')



time.sleep(2)


PAUG.hotkey('ctrl','a')
time.sleep(1)

PAUG.hotkey('ctrl','v')
time.sleep(1)

PAUG.press('y')   #열이 달라도 붙여넣을거냐


excel = win32com.client.Dispatch("Excel.Application")
wb = excel.Workbooks
wb.Close(SaveChanges=False)  # 변경사항을 저장하지 않고 닫음