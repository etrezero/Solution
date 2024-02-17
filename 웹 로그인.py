import win32com.client as win
import os
import subprocess 
import pyautogui as PAUG
import time

# os.startfile(Path2)
# import time
# time.sleep(10)



# print(PAUG.position())

# PAUG.click(clicks=2, interval=0.05)
# PAUG.click(button='right')
# PAUG.mouseInfo()

# PAUG.doubleClick()
# PAUG.write('startcoding', interval=0.05)
# PAUG.press('enter')
# PAUG.press('up')
# PAUG.hotkey('ctrl','c')

# import pyperclip
# pyperclip.copy('한글입력은 파이퍼클립')
# PAUG.hotkey('ctrl','v')


FNSpectrum = r'C:\KBFund\Fo\REAL\Bin\MainExe.exe'
# subprocess.run(FNSpectrum)


import webbrowser as WB
from selenium import webdriver
WB.open('https://www.naver.com/')
driver = webdriver.Edge()
# driver.get('https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fcalendar.naver.com%2Fmain%23%257B%2522sSection%2522%253A%2522scheduleMain%2522%252C%2522oParameter%2522%253A%257B%2522sViewType%2522%253A%2522month%2522%252C%2522sDate%2522%253A%25222023-10-01%2522%257D%257D')

# PAUG.mouseInfo()
# PAUG.moveTo(1356,448)
# PAUG.click()

from selenium.webdriver.common.by import By
# id = driver.find_element(By.CSS_SELECTOR."#id")
# id.send_keys('etrezero')

# PAUG.moveTo(1356,448)
# PAUG.click()

# PAUG.moveTo(935,570,1)
# PAUG.click()