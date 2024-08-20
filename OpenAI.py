import openai
import yaml
import requests
from flask import Flask, request, render_template_string

app = Flask(__name__)

# config.yaml 파일에서 OpenAI API 키를 읽어옵니다.
def load_openai_key():
    with open("C:\\Covenant\\config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config.get("openai_key")

# ChatGPT API에 요청을 보내는 함수
def chatgpt_response(prompt):
    openai.api_key = load_openai_key()
    
    # API 요청 보내기 (프록시 없이)
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",  # 최신 엔드포인트로 변경
        headers={
            "Authorization": f"Bearer {openai.api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4",  # 또는 "gpt-4"
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 350
        },
        verify=False  # SSL 인증서 검증 비활성화
    )
    
    # 응답 상태 코드 확인
    if response.status_code != 200:
        return f"Error: {response.status_code}, {response.text}"
    
    # 'choices' 키가 존재하는지 확인하고 반환
    response_json = response.json()
    if 'choices' in response_json:
        return response_json['choices'][0]['message']['content'].strip()
    else:
        return "Error: 'choices' not found in the response"

# 웹 대시보드 루트 경로
@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        prompt = request.form["prompt"]
        response = chatgpt_response(prompt)
    
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Covenant GPT</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f3f4f6;
                color: #333;
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 800px;
                margin: 50px auto;
                background-color: #ffffff;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                border-radius: 8px;
            }
            h1 {
                color: #1a73e8;
                text-align: center;
                font-size: 24px;
            }
            label {
                font-size: 18px;
                font-weight: bold;
                display: block;
                margin-bottom: 10px;
            }
            textarea {
                width: 100%;
                padding: 10px;
                font-size: 16px;
                border-radius: 4px;
                border: 1px solid #ddd;
                margin-bottom: 20px;
            }
            input[type="submit"] {
                background-color: #1a73e8;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                cursor: pointer;
                border-radius: 4px;
                display: block;
                width: 100%;
            }
            input[type="submit"]:hover {
                background-color: #1669c1;
            }
            h2 {
                color: #333;
                font-size: 20px;
                margin-top: 30px;
            }
            p {
                font-size: 16px;
                line-height: 1.5;
                background-color: #f1f1f1;
                padding: 10px;
                border-radius: 4px;
                word-wrap: break-word;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Covenant GPT</h1>
            <form method="post">
                <label for="prompt">커버넌트에게 물어보세요:</label>
                <textarea id="prompt" name="prompt" rows="4" placeholder="Type your question here..."></textarea>
                <input type="submit" value="Submit">
            </form>
            <h2>Response:</h2>
            <p>{{ response }}</p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, response=response)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
