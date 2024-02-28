import win32com.client
import win32gui
import win32con




#전체 핸들과 창의 이름 출력
def get_all_window_handles_with_names():
    def callback(hwnd, hwnds):
        if win32gui.IsWindowVisible(hwnd):
            window_text = win32gui.GetWindowText(hwnd)
            hwnds.append((hwnd, window_text))
        return True

    all_handles_with_names = []
    win32gui.EnumWindows(callback, all_handles_with_names)
    return all_handles_with_names

# 현재 활성화되어 있는 모든 창의 핸들 목록과 이름 가져오기
all_handles_with_names = get_all_window_handles_with_names()

# 출력 형식: 1행 2열로 출력
for hwnd, name in all_handles_with_names:
    print(f"핸들: {hwnd}, 이름: {name}")









def maximize_window_by_name(window_name):
    # 모든 윈도우 핸들을 열거
    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and window_name in win32gui.GetWindowText(hwnd):
            # 창이 보이고 이름이 일치하는 경우 창을 최대화
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
    
    win32gui.EnumWindows(callback, None)

# 특정 프로그램 창 또는 작업 창의 이름을 지정합니다.
window_name = "한국투자신탁운용"

# 함수를 호출하여 창을 최대화합니다.
maximize_window_by_name(window_name)






# #엑셀창 제일 앞으로
# def bring_excel_to_front():
#     try:
#         # Excel 초기화
#         excel = win32com.client.Dispatch("Excel.Application")
#         excel.Visible = True  # Excel 창을 보이게 함

#         # 현재 활성화된 워크북 가져오기
#         workbook = excel.ActiveWorkbook

#         # 워크북이 존재하고, 저장되지 않았다면
#         if workbook is not None and not workbook.Saved:
#             # Excel 창 핸들 가져오기
#             excel_hwnd = win32gui.FindWindow(None, excel.Caption)
#             # 최소화되어 있다면 복원
#             win32gui.ShowWindow(excel_hwnd, win32con.SW_RESTORE)
#             # 창을 최상위로 가져오기
#             win32gui.SetForegroundWindow(excel_hwnd)

#     except Exception as e:
#         print("Excel을 최상위로 가져오는 동안 오류 발생:", e)

# # Excel 창을 최상위로 가져오기
# bring_excel_to_front()


# #특정 엑셀창 제일 앞으로
# def bring_excel_sheet_to_front(excel, workbook_name, sheet_name):
#     try:
#         # Excel 초기화
#         excel = win32com.client.Dispatch("Excel.Application")
#         excel.Visible = True  # Excel 창을 보이게 함

#         # 현재 활성화된 워크북 가져오기
#         workbook = excel.Workbooks(workbook_name)

#         # 워크북이 존재하고, 지정된 시트 이름이 있는 경우
#         if workbook is not None and sheet_name in [sheet.Name for sheet in workbook.Sheets]:
#             excel_hwnd = win32gui.FindWindow(None, excel.Caption)
#             win32gui.ShowWindow(excel_hwnd, win32con.SW_RESTORE)  # 최소화되어 있다면 복원
#             win32gui.SetForegroundWindow(excel_hwnd)

#             # 시트를 찾아서 최상위로 가져오기
#             for sheet in workbook.Sheets:
#                 if sheet.Name == sheet_name:
#                     sheet.Activate()
#                     break

#     except Exception as e:
#         print(f"{sheet_name} 시트를 최상위로 가져오는 동안 오류 발생:", e)

# # Excel 초기화와 보이게 설정하는 코드를 함수 내에 넣어주기
# bring_excel_sheet_to_front(None, '요약', 'BOS3426')

