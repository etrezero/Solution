# Python 3.11 Slim 이미지 기반
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# Waitress 서버로 애플리케이션 실행
CMD ["waitress-serve", "--host", "0.0.0.0", "--port", "8000", "app:server"]
