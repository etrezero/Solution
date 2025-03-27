from flask import Flask, request, jsonify, send_from_directory, render_template
from pathlib import Path
import traceback
import openai
import yaml
import io
import sys
import os
import json
import uuid
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import socket
import re
import requests
import random
import numpy as np
import yfinance
import requests





# 앱 초기화
app = Flask(__name__)


# 내부 저장소 경로 설정
BASE_DIR = Path("C:/covenant")
RENDER_DIR = BASE_DIR / "rendered"
RENDER_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = Path("C:/covenant/config.yaml")

CACHE_FILE = BASE_DIR / "cache" / "cache_코드생성기.json"
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)


# API 키 로드
def load_openai_key():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"{CONFIG_PATH} 파일이 존재하지 않습니다.")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["openai_api_key"]


# 캐시 저장
def save_to_cache(record):
    data = []
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    data.append(record)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# 유사 명령 찾기
def find_similar_command(command, top_n=1):
    if not CACHE_FILE.exists():
        return []
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    similar = [d for d in data if command in d['command'] or d['command'] in command]
    return similar[:top_n]



def Run_GPT(nl_command, error_msg=None):
    api_key = load_openai_key()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    similar = find_similar_command(nl_command)
    past_context = ""
    if similar:
        past_context = f"\n과거 예시:\n{similar[0]['command']}\n{similar[0]['cleaned_code']}"

    prompt = f"""
     사용자 명령: {nl_command}
    {past_context}
    - 실행 가능한 파이썬 코드를 생성해주세요.
    - 예시데이터를 사용하지 말아요. 반드시 실제 데이터를 사용해주세요.
    - 데이터는 가장 최근까지 업데이트된 데이터를 사용해주세요.
    - 데이터와 자료의 출처를 명시해주세요. 이 경우 HTML div 또는 주석으로 처리해주세요.
    - pandas와 plotly를 사용해주세요.
    - fig = px.line(...) 또는 go.Figure 사용
    - go.Figure를 사용할 때는 pio.write_html(fig, file='render.html', auto_open=False)를 사용해주세요. 
    - fig를 반환하는 방식으로 작성
    - 마크다운 없이 코드만 출력
    - 외부 파일을 읽거나 쓰지 마세요. 필요한 경우 데이터를 반드시 코드에 포함해주세요.
    - 표나 그래프 타이틀에는 자료의 출처를 표기해주세요.
    """

    if error_msg:
        missing = extract_missing_module(error_msg)
        module_hint = f"\n💡 오류 메시지로 보아 '{missing}' 모듈이 누락된 것으로 보입니다." if missing else ""
        prompt += f"\n\n오류 메시지:\n{error_msg}{module_hint}\n오류를 수정한 코드를 출력해주세요."
        
    if isinstance(error_msg, str) and "fig" in error_msg.lower():
        prompt += "\n\nfig 객체가 생성되지 않아 HTML 렌더링에 실패했습니다. 반드시 fig = ... 코드를 포함해주세요."

    print("[GPT에 전달된 프롬프트]\n", prompt)

    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1000
    }

    res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, verify=False)
    if res.status_code != 200:
        return f"# 오류 발생: {res.status_code}\n{res.text}", prompt
    return res.json()['choices'][0]['message']['content'].strip(), prompt



def clean_code(text):
    # 1. 모든 ```python ... ``` 블럭 추출
    code_blocks = re.findall(r"```(?:python)?(.*?)```", text, flags=re.DOTALL)

    # 2. 각 블럭별로 줄 단위 필터링 (자연어 설명 제거)
    cleaned = []
    for block in code_blocks:
        lines = block.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 자연어로 추정되는 줄 제거 (한글 문장 또는 마침표 등)
            if re.match(r'^[\uac00-\ud7a3a-zA-Z0-9 ,\'"()\[\]{}=<>:+\-/*%&|!~`^]+$', line):
                cleaned.append(line)

    return '\n'.join(cleaned).strip()





def extract_missing_module(error_msg):
    match = re.search(r"NameError: name '(\w+)' is not defined", error_msg)
    if match:
        return match.group(1)
    return None



def execute_python_code(code):
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    local_vars = {}
    error = ""


    safe_globals = {
    "pd": pd,
    "px": px,
    "go": go,
    "pio": pio,
    "np": np,
    "random": random,
    "yf": yfinance,
    "requests": requests,
    }


    try:
        exec(code, safe_globals, local_vars)
        fig = local_vars.get('fig')
        if fig:
            html = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
            filename = f"render_{uuid.uuid4().hex[:8]}.html"
            filepath = RENDER_DIR / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)
            return filepath.name, ""
        elif 'df' in local_vars:
            df = local_vars['df']
            html = df.to_html(index=False)
            filename = f"render_{uuid.uuid4().hex[:8]}.html"
            filepath = RENDER_DIR / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)
            return filepath.name, ""
    except Exception:
        error = traceback.format_exc()
        # 💡 누락된 모듈 이름 추출
        missing_module = extract_missing_module(error)
        if missing_module:
            error += f"\n\n추가로 '{missing_module}' 모듈이 필요한 것으로 보입니다. Run_GPT에 전달하여 반영해주세요."
    finally:
        sys.stdout = old_stdout
    return None, error



# API 엔드포인트
@app.route("/run", methods=["POST"])
def run_code():
    try:
        data = request.json
        command = data.get("command", "")
        gpt_reply, prompt = Run_GPT(command)
        cleaned_code = clean_code(gpt_reply)
        rendered_filename, error = execute_python_code(cleaned_code)

        save_to_cache({
            "command": command,
            "prompt": prompt,
            "GPT_reply": gpt_reply,
            "cleaned_code": cleaned_code,
            "output": rendered_filename,
            "error": error
        })

        if error:
            return jsonify({
                "prompt": prompt,
                "GPT_reply": gpt_reply,
                "cleaned_code": cleaned_code,
                "output": "",
                "error": error,
                "summary": "오류 발생"
            }), 200

        return jsonify({
            "prompt": prompt,
            "GPT_reply": gpt_reply,
            "cleaned_code": cleaned_code,
            "output": f"/rendered/{rendered_filename}",
            "error": "",
            "summary": "코드 실행 및 HTML 렌더링 성공"
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "summary": "서버 처리 중 오류 발생"
        }), 500


@app.route("/rendered/<path:filename>")
def serve_rendered_html(filename):
    if (RENDER_DIR / filename).exists():
        return send_from_directory(RENDER_DIR, filename)
    return f"파일을 찾을 수 없습니다: {filename}", 404





@app.route("/")
def index():
    return render_template("index_코드생성기.html")





# 포트 찾기 및 실행
if __name__ == '__main__':
    def find_available_port(start_port=8050, max_attempts=10):
        for port in range(start_port, start_port + max_attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("사용 가능한 포트를 찾을 수 없습니다.")

    port = find_available_port()
    print(f"서버가 실행 중입니다: http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
