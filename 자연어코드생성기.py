import openai
import traceback
from flask import Flask, request, jsonify
import io
import sys
import yaml
import socket
import requests
import re
import json
from pathlib import Path
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import uuid




# 앱 초기화
app = Flask(__name__)

# 내부 저장소 경로 설정
BASE_DIR = Path("/data/data/ru.iiec.pydroid3/files")
CONFIG_PATH = r"D:\\code\\config.yaml"
CACHE_FILE = BASE_DIR / "cache" / "cache.json"
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_openai_key():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"{CONFIG_PATH} 파일이 존재하지 않습니다.")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["openai_api_key"]

api_key = load_openai_key()


# 캐시 저장
def save_to_cache(record):
    data = []
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    data.append(record)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 유사 명령어 검색
def find_similar_command(command, top_n=1):
    if not CACHE_FILE.exists():
        return []
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    similar = [d for d in data if command in d['command'] or d['command'] in command]
    return similar[:top_n]




# GPT 호출
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
        
{error_msg}


        {past_context}

        이 오류를 수정하여 다시 완전한 파이썬 코드를 생성해주세요.
        실행 가능한 코드만 출력하고, 설명이나 마크다운 블록(예: 
python)은 포함하지 마세요.
        """
    else:
        prompt = f"""
        사용자의 명령: {nl_command}

        {past_context}

        위 명령을 기반으로 실행 가능한 파이썬 코드를 생성해주세요.
        다음 GitHub 코드 스타일을 참고하세요: https://github.com/etrezero/Solution
        - 데이터프레임은 pandas 사용
        - 시각화는 plotly로 interative하게
        - 변수명은 snake_case
        - fig = px.line(...) 또는 fig = go.Figure(...) 형식으로 작성
        - 그래프는 fig.show() 대신 fig 반환
        - 실행 가능한 코드만 출력하고 마크다운은 포함하지 마세요.
        - 질문에 대한 응답은 웹스크랩 등을 통해서 신뢰가능한 최근 실제데이터만 사용해주세요.
        """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1000
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




# GPT 코드 정리
def clean_code(text: str) -> str:
    matches = re.findall(r"python(.*?)", text, re.DOTALL | re.IGNORECASE)
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

# 파이썬 코드 실행
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


# 요약
def summarize_result(output, error):
    api_key = load_openai_key()
    prompt = f"웹이나 공개자료 등 실제 데이터를 활용해줘줘:\n{error}" if error \
             else f"응답을 숫자나 시계열자료 등으로 할 때 표와 그래프 등으로 가독성 있는 HTML 또는 이미지를 보여주는 데 필요한 데이터들을 확보해서 HTML 또는 이미지를 보여주는 데 도움을 줘:\n{output}"

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

# HTML 포맷
HTML_WRAPPER = """
<div style='background:#eef; padding:10px; border-radius:6px;'>
{content}
</div>
"""

# 출력 HTML 변환
def convert_output_to_html(output):
    try:
        df = pd.read_csv(io.StringIO(output))
        return df.to_html(index=False, classes='table', border=1)
    except Exception:
        pass

    try:
        global_vars = {}
        exec("import plotly.graph_objects as go\nimport plotly.express as px\n" + output, global_vars)
        fig = global_vars.get("fig")
        if fig:
            return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        return HTML_WRAPPER.format(content="그래프 렌더링 실패: " + str(e))

    return HTML_WRAPPER.format(content=output.replace('\n', '<br>'))




# 엔드포인트
@app.route('/run', methods=['POST'])
def run_code():
    data = request.json
    nl_command = data.get('command', '')
    GPT_reply = Run_GPT(nl_command)
    cleaned_code = clean_GPT(GPT_reply)
    output, error = execute_python_code(cleaned_code)
    html_output = convert_output_to_html(output)
    summary = summarize_result(output, error)
    return jsonify({
        'GPT_reply': GPT_reply,
        'cleaned_code': cleaned_code,
        'output': html_output,
        'error': error,
        'summary': summary
    })
    
    
    

@app.route('/')
def index():
    return '''
   

<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>*Covenant* 자연어 → 파이썬 실행기</title>
  <style>
    body { font-family: sans-serif; margin: 0; padding: 20px; background: #f9f9f9; display: flex; justify-content: center; }
    .container { width: 90%; max-width: 800px; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    textarea { width: 100%; height: 100px; font-size: 16px; padding: 10px; border-radius: 5px; border: 1px solid #ccc; }
    button { width: 100%; padding: 12px; font-size: 16px; margin-top: 10px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
    button:hover { background: #0056b3; }
    .section { margin-top: 30px; }
    pre, .html-output { background: #f1f1f1; padding: 10px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; }
    #loading { display: none; text-align: center; margin-top: 20px; }
    .spinner { margin: auto; border: 4px solid #f3f3f3; border-top: 4px solid #007bff; border-radius: 50%; width: 36px; height: 36px; animation: spin 1s linear infinite; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="container">
    <h1>자연어 → 파이썬 코드 실행기</h1>
    <textarea id="command" placeholder="예: 대한민국 인구추이를 그래프로 작성해줘">예: 대한민국 인구추이를 그래프로 작성해줘</textarea>
    <button id="run-btn" onclick="runCommand()">실행</button>



    <div id="loading">
      <div class="spinner"></div>
      <p>GPT 응답을 기다리는 중입니다...</p>
    </div>

    
    
    

<div class="section">
  <h3>📤 출력 결과</h3>
  <div id="output" class="html-output"></div>
</div>
<div class="section">
  <h3>🧹 클린된 실행 코드 출력 결과</h3>
  <div id="rendered_cleaned_code" class="html-output dash-container" onclick="showRenderedCleanedCode()"></div>
</div>

<script>
  let runInitiatedByUser = false;

  async function runCommand() {
    runInitiatedByUser = true;
    document.getElementById("loading").style.display = "block";
    const command = document.getElementById("command").value;
    try {
      const res = await fetch("/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command })
      });
      const data = await res.json();

      const codeEl = document.getElementById("code");
      const cleanedCodeEl = document.getElementById("cleaned_code");
      const renderedCodeEl = document.getElementById("rendered_cleaned_code");
      const outputEl = document.getElementById("output");
      const summaryEl = document.getElementById("summary");

      if (codeEl) codeEl.textContent = data.GPT_reply || '';
      if (cleanedCodeEl) cleanedCodeEl.textContent = data.cleaned_code || '';

      const renderedHTML = generateHTMLWrapper(data.output || '');
      if (renderedCodeEl) renderedCodeEl.innerHTML = renderedHTML;

      if (outputEl) outputEl.textContent = data.cleaned_code || '';
      if (summaryEl) summaryEl.textContent = data.summary || '';

    } catch (err) {
      alert("오류 발생: " + err.message);
    } finally {
      document.getElementById("loading").style.display = "none";
    }
  }

  function showRenderedCleanedCode() {
    const htmlContainer = document.getElementById("rendered_cleaned_code");
    if (htmlContainer) {
      const html = htmlContainer.innerHTML;
      const popup = window.open("", "렌더링된 출력", "width=800,height=600");
      popup.document.write(`<!DOCTYPE html><html><head><title>렌더링</title><link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'></head><body class='dash-bootstrap'>${html}</body></html>`);
    }
  }

  function generateHTMLWrapper(content) {
    const isHTML = /<[a-z][\s\S]*>/i.test(content);
    if (isHTML) {
      return `<div class='dash-bootstrap p-3 rounded border bg-light'>${content}</div>`;
    } else {
      return `<pre class='bg-light border rounded p-3'>${content}</pre>`;
    }
  }

  // 페이지 로드시 초기 실행
  window.onload = function () {
    runInitiatedByUser = false;
  };
</script>

</body>
</html>

    '''
    
    
    

# 포트 자동 선택
DEFAULT_PORT = 80
def find_available_port(start_port=DEFAULT_PORT, max_attempts=10):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError("사용 가능한 포트를 찾을 수 없습니다.")

# 서버 실행
if __name__ == '__main__':
    port = find_available_port()
    print(f"서버가 실행 중입니다: http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
    
    
    # 80 포트는 covenant.한국 으로 ddns 포워드 됨
    