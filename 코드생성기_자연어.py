import openai
import traceback
from flask import Flask, request, jsonify
import io
import sys
import yaml
import socket
import requests
import re
import subprocess
import json
from pathlib import Path




app = Flask(__name__)

# 🔐 config.yaml에서 OpenAI API 키 불러오기
def load_openai_key():
    with open(r"C:\Covenant\config.yaml", "r", encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config.get("openai_key")

openai.api_key = load_openai_key()

# 📁 캐시 파일 경로
CACHE_FILE = Path(r"C:\Covenant\cache\cache.json")
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

# 🧠 캐시 저장
def save_to_cache(record):
    data = []
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    data.append(record)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 🔍 캐시에서 유사 명령 검색
def find_similar_command(command, top_n=1):
    if not CACHE_FILE.exists():
        return []

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    similar = [d for d in data if command in d['command'] or d['command'] in command]
    return similar[:top_n]

# 🔄 GPT 호출
def Run_GPT(nl_command, error_msg=None):
    api_key = load_openai_key()
    similar = find_similar_command(nl_command)

    past_context = ""
    if similar:
        example = similar[0]
        past_context = f"""
        참고할 수 있는 과거 명령과 GPT 응답이 있습니다:

        - 이전 명령: {example['command']}
        - 이전 코드 예시: {example['cleaned_code']}
        - 실행 결과: {example['output']}
        """

    if error_msg:
        prompt = f"""
        사용자가 요청한 명령: {nl_command}

        아래는 GPT가 생성한 코드 실행 중 발생한 오류입니다:
        ```
        {error_msg}
        ```

        {past_context}

        이 오류를 수정하여 다시 완전한 파이썬 코드를 생성해주세요.
        실행 가능한 코드만 출력하고, 설명이나 마크다운 블록(예: ```python)은 포함하지 마세요.
        """
    else:
        prompt = f"""
        사용자의 명령: {nl_command}

        {past_context}

        위 명령을 기반으로 실행 가능한 파이썬 코드를 생성해주세요.
        Dash 앱에서 HTML 요소를 사용해 시각화 가능하도록 구성해주세요.
        마크다운 블록 없이 코드만 출력하세요.
        """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 700
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        verify=False
    )

    if response.status_code != 200:
        return f"# 오류 발생: {response.status_code}\n{response.text}"

    return response.json()['choices'][0]['message']['content'].strip()

# 🧼 코드 클린 처리
def clean_code(text: str) -> str:
    matches = re.findall(r"```python(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if matches:
        code = "\n\n".join(match.strip() for match in matches)
    else:
        code = text.strip()

    code = re.sub(r"#.*", "", code)
    code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", "", code, flags=re.DOTALL)
    return code.strip()

def clean_GPT(GPT_reply: str):
    cleaned_code = clean_code(GPT_reply)
    print("cleaned_code", cleaned_code)
    return cleaned_code

# ▶️ 코드 실행
def execute_python_code(code):
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    error_msg = ""
    try:
        exec(code, {})
    except Exception:
        error_msg = traceback.format_exc()
    sys.stdout = old_stdout
    output = redirected_output.getvalue()
    return output, error_msg

# 💬 실행 결과 요약
def summarize_result(output, error):
    api_key = load_openai_key()
    prompt = f"다음 에러 메시지를 사용자에게 이해하기 쉽게 설명해줘:\n{error}" if error \
             else f"다음 파이썬 실행 결과를 간단히 요약해서 설명해줘:\n{output}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 300
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        verify=False
    )

    if response.status_code != 200:
        return f"[요약 중 오류 발생: {response.status_code}] {response.text}"

    return response.json()['choices'][0]['message']['content'].strip()

# 🧠 API 엔드포인트
@app.route('/run', methods=['POST'])
def run_code():
    data = request.json
    nl_command = data.get('command', '')

    GPT_reply = Run_GPT(nl_command)
    cleaned_code = clean_GPT(GPT_reply)
    output, error = execute_python_code(cleaned_code)

    if error:
        GPT_reply_feedback = Run_GPT(nl_command, error_msg=error)
        cleaned_code_feedback = clean_GPT(GPT_reply_feedback)
        output_feedback, error_feedback = execute_python_code(cleaned_code_feedback)
        summary = summarize_result(output_feedback, error_feedback)

        save_to_cache({
            'command': nl_command,
            'GPT_reply': GPT_reply_feedback,
            'cleaned_code': cleaned_code_feedback,
            'output': output_feedback,
            'error': error_feedback,
            'summary': summary
        })

        return jsonify({
            'GPT_reply': GPT_reply_feedback,
            'cleaned_code': cleaned_code_feedback,
            'output': output_feedback,
            'error': error_feedback,
            'summary': summary,
            'feedback_used': True
        })

    summary = summarize_result(output, error)

    save_to_cache({
        'command': nl_command,
        'GPT_reply': GPT_reply,
        'cleaned_code': cleaned_code,
        'output': output,
        'error': error,
        'summary': summary
    })

    return jsonify({
        'GPT_reply': GPT_reply,
        'cleaned_code': cleaned_code,
        'output': output,
        'error': error,
        'summary': summary,
        'feedback_used': False
    })

# 🌐 프론트 웹 UI
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>자연어 → 파이썬 실행기</title>
        <style>
            body { font-family: sans-serif; padding: 20px; max-width: 800px; margin: auto; }
            textarea, pre { width: 100%; }
            textarea { height: 100px; }
            button { padding: 10px 20px; font-size: 16px; margin-top: 10px; }
            .section { margin-top: 30px; }
        </style>
    </head>
    <body>
        <h1>자연어 → 파이썬 코드 실행기</h1>
        <textarea id="command" placeholder="예: 1부터 10까지 합을 구하는 코드를 작성해줘"></textarea><br>
        <button onclick="runCommand()">실행</button>

        <div class="section">
            <h3>⛏️ 생성된 코드 (GPT 원문)</h3>
            <pre id="code"></pre>
        </div>
        <div class="section">
            <h3>🧹 클린된 실행 코드</h3>
            <pre id="cleaned_code"></pre>
        </div>
        <div class="section">
            <h3>📤 출력 결과</h3>
            <pre id="output"></pre>
        </div>
        <div class="section">
            <h3>💡 설명 요약</h3>
            <pre id="summary"></pre>
        </div>

        <script>
            async function runCommand() {
                const command = document.getElementById("command").value;
                const res = await fetch("/run", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ command })
                });
                const data = await res.json();
                document.getElementById("code").textContent = data.GPT_reply || '';
                document.getElementById("cleaned_code").textContent = data.cleaned_code || '';
                document.getElementById("output").textContent = data.output || '';
                document.getElementById("summary").textContent = data.summary || '';
            }
        </script>
    </body>
    </html>
    '''

# 🔌 포트 찾기
DEFAULT_PORT = 8051
def find_available_port(start_port=DEFAULT_PORT, max_attempts=10):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError("사용 가능한 포트를 찾을 수 없습니다.")

# 🚀 서버 실행
if __name__ == '__main__':
    port = find_available_port()
    print(f"서버가 실행 중입니다: http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
