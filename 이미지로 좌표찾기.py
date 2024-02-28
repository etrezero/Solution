import pyautogui as PAUG
import time
import win32gui
import win32com.client
import pygetwindow as PGW
import pywin
from PIL import ImageGrab, Image
import openpyxl
import os
from datetime import datetime

#마우스 위치 좌표
mouse_x, mouse_y = PAUG.position()
print(f"현재 마우스 좌표: ({mouse_x}, {mouse_y})")













# screen_width = win32gui.GetWindowRect(win32gui.GetDesktopWindow())[2]
# screen_height = win32gui.GetWindowRect(win32gui.GetDesktopWindow())[3]
# center_x = screen_width / 2
# center_y = screen_height / 2


# 설정해지 = r"C:\Users\USER\Desktop\python ws\설정해지.PNG"
# screen = ImageGrab.grab()
# image = Image.open(설정해지)
# match_position = PAUG.locateOnScreen(설정해지)
# if match_position:
#         x = match_position.left + match_position.width / 2
#         y = match_position.top + match_position.height / 2
#         PAUG.click(x, y)       
# else:
#         print("설정해지 파일과 일치하는 부분을 찾을 수 없습니다.")

# print(x, y)








# #마우스 위치 좌표
# mouse_x, mouse_y = PAUG.position()

# # PAUG.click(mouse_x, mouse_y)

# 센터좌표 = (960, 540)
# PAUG.click(센터좌표[0], 센터좌표[1], button='right')
# PAUG.moveTo(990, 555)
# print(f"현재 마우스 좌표: ({mouse_x}, {mouse_y})")

