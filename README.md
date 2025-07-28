# Streamlit 데이터 시각화 대시보드

## Docker로 실행하기
### 1. Docker 명령어 직접 사용
```bash
# 이미지 빌드
docker build -t ev_streamlit .

# 컨테이너 실행
docker run -p 8509:8509 ev_streamlit
```

### 2. 접속 방법
애플리케이션이 실행되면 다음 URL로 접속할 수 있습니다:
- http://localhost:8509

## 로컬에서 실행하기
```bash
pip install -r requirements.txt
streamlit run main.py --server.port 8509
```

## 기능
- CSV 파일 업로드 및 시각화
- 다양한 차트 타입 (Line, Scatter, Bar)
- 데이터 필터링 및 범위 설정
- 실시간 미리보기