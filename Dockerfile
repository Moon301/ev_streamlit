FROM python:3.9-slim

WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

RUN mkdir -p /root/.streamlit
COPY .streamlit/config.toml /root/.streamlit/config.toml

COPY sample/ /app/sample/

# 포트 8509 노출
EXPOSE 8509

# Streamlit을 8509 포트에서 실행
CMD ["streamlit", "run", "main.py", "--server.port", "8509", "--server.address", "0.0.0.0"] 