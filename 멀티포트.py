


from flask import Flask
import socket
import dash

# Flask 서버 생성
server = Flask(__name__)
app = dash.Dash(__name__, server=server)
app.title = 'TDF_판매회사'





# 기본 포트 설정 ============================= 여러개 실행시 충돌 방지

DEFAULT_PORT = 8051

def find_available_port(start_port=DEFAULT_PORT, max_attempts=10):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))  # 실제 바인딩을 시도
                return port  # 사용 가능한 포트 반환
            except OSError:
                continue  # 이미 사용 중이면 다음 포트 확인
    raise RuntimeError("사용 가능한 포트를 찾을 수 없습니다.")

port = find_available_port()
print(f"사용 중인 포트: {port}")  # 디버깅용


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=port)

# ==================================================================




