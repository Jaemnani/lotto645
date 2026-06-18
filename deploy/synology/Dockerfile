# Synology Container Manager 용 FastAPI 웹 서버 이미지
# 빌드 컨텍스트 = repo 루트:
#   docker build -f deploy/synology/Dockerfile -t lotto645-web .
# (deploy/synology/docker-compose.yml 이 자동으로 이 경로/컨텍스트로 빌드함)

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Seoul

WORKDIR /app

# 슬림 의존성만 설치 (torch 제외 → 이미지 가벼움)
COPY deploy/synology/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 서빙에 필요한 코드/모델만 복사
COPY web/               /app/web/
COPY model_m03_claude/  /app/model_m03_claude/
COPY run_web.py         /app/run_web.py

# 모델 체크포인트(best_m03.npz)가 이미지에 없어도 기동 30초 후 DB 에서 자동 재학습됨.

EXPOSE 8000

# .env 는 compose 의 env_file 로 주입 (이미지에 시크릿 굽지 않음)
CMD ["python", "run_web.py"]
