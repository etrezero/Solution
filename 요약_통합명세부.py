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


ID = "T08186"

PW = "T202403!"

FOS = "C:\\KBFund\\Fo\\REAL\\Bin\\KBFundFO.exe"
BOS = r'"C:\Program Files (x86)\TOBESOFT\XPLATFORM\9.2.1\XPlatform.exe" -K "KBFundBo" -X "https://fund.kbstar.com:2185/kbfundbo/Resource/KBFundBo.xadl"'
MOS = r'"C:\Program Files (x86)\TOBESOFT\XPLATFORM\9.2.1\XPlatform.exe" -K "KBFundMo" -X "https://fund.kbstar.com:2195/kbfundmo/Resource/KBFundMo.xadl"'



today_date = datetime.now()

T0 = today_date.strftime("%Y%m%d")

T_1 = today_date - timedelta(days=1)
T_1D = T_1.strftime("%Y%m%d")


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




excel = win32com.client.Dispatch("Excel.Application")
path요약 = r'C:\Covenant\data\요약.xlsx'
workbook_요약 = excel.Workbooks.Open(path요약)
time.sleep(0.5)





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

time.sleep(1)
PAUG.hotkey('ctrl','v')


time.sleep(1)
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


time.sleep(1)
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

PAUG.press('Enter')   #열이 달라도 붙여넣을거냐


excel = win32com.client.Dispatch("Excel.Application")
wb = excel.Workbooks
wb.Close()  # 변경사항을 저장하지 않고 닫음