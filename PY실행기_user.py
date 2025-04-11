import os
import subprocess
import socket
from threading import Thread
from flask import request
from dash import Dash, dcc, html, Input, Output, State, ctx
from dash.dependencies import ALL

# 설정
BASE_PATH = r"C:\Covenant"
APP_PORT = 8050
START_PORT = 8051

# { script_key (ex: test.py_8051): (Popen, port) }
running_processes = {}

# 사용 가능한 포트 찾기
def find_available_port(start_port=START_PORT):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                port += 1

# Dash 앱 초기화
app = Dash(__name__)
server = app.server
app.title = "📂 Dash 실행기 (실행 중 앱 실시간 표시)"

# 레이아웃
app.layout = html.Div(
    style={'width': '50%', 'margin': 'auto', 'fontFamily': 'Arial', 'fontSize': '14px'},
    children=[
        html.H2("📁 실행 가능한 Python 파일", style={"textAlign": "center"}),

        dcc.Dropdown(
            id="script-dropdown",
            options=[
                {"label": f, "value": f}
                for f in os.listdir(BASE_PATH)
                if f.endswith(".py") and f != "main_launcher.py"
            ],
            placeholder="▶ 실행할 .py 파일 선택",
            style={"width": "70%", "margin": "auto"}
        ),

        html.Button("실행", id="run-button", n_clicks=0, style={"marginTop": "20px"}),

        dcc.Store(id="popup-port"),

        dcc.Loading(
            id="loading-output",
            type="circle",
            color="#007BFF",
            fullscreen=False,
            children=[
                html.Div(id="popup-script", style={"marginTop": "20px"}),
                html.Hr(),
                html.H4("🟢 현재 실행 중인 앱 목록", style={"marginTop": "30px"}),
                html.Div(id="running-list")
            ]
        )
    ]
)

# 실행 함수 (멀티 포트 + 고유 키)
def launch_script(script_name):
    full_path = os.path.join(BASE_PATH, script_name)
    port = find_available_port()
    script_key = f"{script_name}_{port}"

    def _run():
        command = f'python "{full_path}"'
        env = os.environ.copy()
        env["PORT"] = str(port)
        proc = subprocess.Popen(command, shell=True, env=env)
        running_processes[script_key] = (proc, port)
        proc.wait()
        running_processes.pop(script_key, None)

    Thread(target=_run, daemon=True).start()
    return port, script_key

# 실행 버튼 클릭 시 → 앱 실행 + 포트 반환
@app.callback(
    Output("running-list", "children"),
    Output("popup-port", "data"),
    Input("run-button", "n_clicks"),
    State("script-dropdown", "value"),
    prevent_initial_call=True
)
def run_script(n_clicks, selected_file):
    if selected_file:
        port, script_key = launch_script(selected_file)
        return generate_running_list(), port
    return generate_running_list(), None

# 새 탭으로 열기 위한 팝업 링크
@app.callback(
    Output("popup-script", "children"),
    Input("popup-port", "data"),
    prevent_initial_call=True
)
def open_browser_tab(port):
    if port:
        host = request.host.split(":")[0]
        url = f"http://{host}:{port}"
        return html.Div([
            html.Div("✅ 앱이 실행되었습니다. 아래 버튼을 눌러 새 탭에서 열어주세요.", style={"marginTop": "10px"}),
            html.Div(f"📡 실행 주소: {url}", style={"marginBottom": "5px", "color": "gray"}),
            html.A("🆕 실행된 앱 열기", href=url, target="_blank", style={
                "display": "inline-block",
                "padding": "8px 16px",
                "backgroundColor": "#007BFF",
                "color": "white",
                "borderRadius": "5px",
                "textDecoration": "none",
                "marginTop": "5px"
            })
        ])
    return ""

# Quit 버튼 클릭 시 프로세스 종료
@app.callback(
    Output("running-list", "children", allow_duplicate=True),
    Input({'type': 'quit-button', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def quit_running_script(n_clicks_list):
    triggered = ctx.triggered_id
    if triggered:
        script_key = triggered['index']
        proc, _ = running_processes.get(script_key, (None, None))
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait()
            running_processes.pop(script_key, None)
    return generate_running_list()

# 실행 중인 리스트 표시
def generate_running_list():
    if not running_processes:
        return html.Div("❌ 현재 실행 중인 앱이 없습니다.")

    host = request.host.split(":")[0]
    children = []

    for script_key, (_, port) in running_processes.items():
        url = f"http://{host}:{port}"
        file_name = script_key.split("_")[0]
        row = html.Div([
            html.Span("🟢 ", style={"marginRight": "5px", "color": "green"}),
            html.Span(f"{file_name}", style={"fontWeight": "bold", "marginRight": "10px"}),
            html.Span("열기:", style={"marginRight": "5px"}),
            html.A(f"{url}", href=url, target="_blank", style={"marginRight": "10px", "color": "blue"}),
            html.Button("Quit", id={'type': 'quit-button', 'index': script_key},
                        n_clicks=0, style={"backgroundColor": "crimson", "color": "white"})
        ], style={"marginBottom": "10px"})
        children.append(row)
    return html.Div(children)

# 실행기 시작
if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=APP_PORT)
