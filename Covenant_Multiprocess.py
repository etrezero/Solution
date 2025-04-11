import os
import socket
import subprocess
from threading import Thread
from flask import request
from multiprocessing import Process
from dash import Dash, dcc, html, Input, Output, State, ctx
from dash.dependencies import ALL

# 사용자 설정 목록 (GitHub 주소도 지원)
USER_CONFIGS = {
    "USER1": {"base": "https://github.com/etrezero/Solution", "port": 8040},
    "USER2": {"base": r"D:\code", "port": 8050},
}

START_PORT = 9000
running_processes = {}

# 포트 찾기
def find_available_port(start_port=START_PORT):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                port += 1

# GitHub URL이면 자동 클론 → 로컬경로 반환
def clone_if_url(base_path):
    if base_path.startswith("http"):
        repo_name = base_path.rstrip("/").split("/")[-1]
        local_path = os.path.join("D:/code", repo_name)
        if not os.path.exists(local_path):
            subprocess.run(["git", "clone", base_path, local_path], check=True)
        return local_path
    return base_path

# 사용자별 Dash 실행기
def run_dash_app(BASE_PATH, APP_PORT):
    app = Dash(__name__)
    server = app.server
    app.title = f"📂 Dash 실행기 ({BASE_PATH})"

    app.layout = html.Div([
        html.H2(f"📁 실행 가능 파일: {BASE_PATH}", style={"textAlign": "center"}),

        dcc.Dropdown(
            id="script-dropdown",
            options=[
                {"label": f, "value": f}
                for f in os.listdir(BASE_PATH)
                if (f.endswith(".py") or f.endswith(".pyc"))
            ],
            placeholder="▶ 실행할 .py 또는 .pyc 파일 선택",
            style={"width": "70%", "margin": "auto"}
        ),

        html.Button("실행", id="run-button", n_clicks=0, style={"marginTop": "20px"}),

        dcc.Store(id="popup-port"),

        dcc.Loading(children=[
            html.Div(id="popup-script", style={"marginTop": "20px"}),
            html.Hr(),
            html.H4("🟢 현재 실행 중인 앱 목록", style={"marginTop": "30px"}),
            html.Div(id="running-list")
        ], type="circle", fullscreen=False)
    ], style={'width': '50%', 'margin': 'auto', 'fontFamily': 'Arial', 'fontSize': '14px'})

    def launch_script(script_name):
        full_path = os.path.join(BASE_PATH, script_name)
        port = find_available_port()
        script_key = f"{script_name}_{port}"

        def _run():
            env = os.environ.copy()
            env["PORT"] = str(port)
            proc = subprocess.Popen(f'python "{full_path}"', shell=True, env=env)
            running_processes[script_key] = (proc, port)
            proc.wait()
            running_processes.pop(script_key, None)

        Thread(target=_run, daemon=True).start()
        return port, script_key

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

    @app.callback(
        Output("popup-script", "children"),
        Input("popup-port", "data"),
        prevent_initial_call=True
    )
    def open_browser_tab(port):
        if port:
            host = request.host.split(":")[0]
            target_url = f"http://{host}:{port}"
            polling_script = f"""
            <script>
            function checkApp() {{
                fetch("{target_url}", {{ mode: "no-cors" }})
                .then(() => window.location.href = "{target_url}")
                .catch(() => setTimeout(checkApp, 2000));
            }}
            checkApp();
            </script>
            """
            return html.Div([
                html.Div("✅ 앱이 실행되었습니다. 잠시 기다려주세요."),
                html.Div(f"📡 주소: {target_url}", style={"color": "gray"}),
                html.A("🆕 열기", href=target_url, target="_blank", style={
                    "color": "white", "backgroundColor": "#007BFF",
                    "padding": "8px 12px", "textDecoration": "none", "borderRadius": "5px"
                }),
                html.Script(polling_script)
            ])
        return ""

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

    def generate_running_list():
        if not running_processes:
            return html.Div("❌ 현재 실행 중인 앱이 없습니다.")
        host = request.host.split(":")[0]
        return html.Div([
            html.Div([
                html.Span(f"🟢 {script_key.split('_')[0]}", style={"fontWeight": "bold"}),
                html.A(f"열기: http://{host}:{port}", href=f"http://{host}:{port}", target="_blank",
                       style={"marginLeft": "10px", "marginRight": "10px", "color": "blue"}),
                html.Button("Quit", id={'type': 'quit-button', 'index': script_key},
                            n_clicks=0, style={"backgroundColor": "crimson", "color": "white"})
            ], style={"marginBottom": "10px"})
            for script_key, (_, port) in running_processes.items()
        ])

    app.run_server(debug=False, host="0.0.0.0", port=APP_PORT)

# === 전체 실행 ===
if __name__ == "__main__":
    processes = []
    for user_key, config in USER_CONFIGS.items():
        local_base = clone_if_url(config["base"])
        p = Process(target=run_dash_app, args=(local_base, config["port"]))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
