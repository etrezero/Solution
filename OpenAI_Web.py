import os
import openai
from dash import Dash, html, dcc, Input, Output, State
from flask import Flask



# 환경 변수에서 OpenAI API 키 가져오기
def get_api_key():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API 키를 찾을 수 없습니다. 시스템 환경 변수에 'OPENAI_API_KEY'를 설정해주세요.")
    return api_key


api_key = get_api_key()
openai.api_key = api_key







# Flask 서버 설정
server = Flask(__name__)

# Dash 앱 설정
app = Dash(__name__, server=server)
app.title = "Covenant AI"

# 레이아웃 설정
app.layout = html.Div([
    html.H1("Covenant AI", 
            style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    html.Div([
        dcc.Textarea(
            id="input-text", 
            placeholder="커버넌트에게 물어보세요...",
            style={
                'width': '60%', 
                'padding': '10px',
                'borderRadius': '5px', 
                'border': '1px solid #bdc3c7',
                'resize': 'none',  # 사용자가 크기를 조절하지 못하게 설정
                'overflow': 'hidden',  # 텍스트가 넘칠 때 스크롤바를 숨김
                'minHeight': '50px',  # 기본 높이 설정
                'lineHeight': '1.5',  # 줄 간격 설정
            },
            rows=1  # 기본적으로 한 줄만 보이도록 설정
        ),
        
    ], style={'textAlign': 'center', 'margin': '20px 0'}),
    
    
    
    html.Button(
        "Send", 
            id="submit-button", n_clicks=0,
            style={
                'padding': '10px 20px', 
                'marginLeft': '10px', 
                'backgroundColor': '#2980b9', 
                'color': 'white', 
                'border': 'none', 
                'borderRadius': '5px', 
                'cursor': 'pointer'
            }
    ),
    
    
    
    
    html.Div(
        id="output-text", 
            style={
            'width': '60%', 
            'marginTop': '30px', 
            'padding': '20px', 
            'borderRadius': '5px', 
            'backgroundColor': '#ecf0f1', 
            'color': '#2c3e50', 
            'maxWidth': '80%', 
            'margin': 'auto', 
            'textAlign': 'center',
            'minHeight': '100px'
        }
    ),
    
    # JavaScript를 통해 Textarea의 높이를 내용에 맞게 조절하는 기능 추가
    html.Script('''
        const textarea = document.getElementById("input-text");
        textarea.addEventListener("input", function() {
            this.style.height = "auto";  // 먼저 높이를 자동으로 설정하여 높이를 리셋
            this.style.height = (this.scrollHeight) + "px";  // 컨텐츠의 높이에 맞춰 조정
        });
    ''')
])

# 콜백 설정
@app.callback(
    Output("output-text", "children"),
    [Input("submit-button", "n_clicks")],
    [State("input-text", "value")]
)
def update_output(n_clicks, value):
    if n_clicks > 0 and value:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                
                
                
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": value}
            ]
        )
        response_text = response['choices'][0]['message']['content'].strip()
        return html.Div([
            html.P(f"Q: {value}", style={'fontWeight': 'bold'}),
            html.P(f"A: {response_text}")
        ])
    return "Enter a question and press Send to get an answer."

# Flask 서버 실행
if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0")
