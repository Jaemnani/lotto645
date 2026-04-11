# Oracle Cloud 서버 세팅 기록

## 서버 정보

| 항목 | 값 |
|------|-----|
| 서비스 | Oracle Cloud Free Tier |
| 인스턴스 | VM.Standard.E2.1.Micro |
| OS | Ubuntu 24.04 |
| 공인 IP | 140.245.65.47 |
| 사용자 | ubuntu |
| 지역 | ap-chuncheon-1 |
| RAM | 1GB / CPU 1코어 |
| SSH 키 | ~/.ssh/oracle-ssh-key-2026-04-09.key |

---

## SSH 접속

```bash
chmod 600 ~/.ssh/oracle-ssh-key-2026-04-09.key
ssh -i ~/.ssh/oracle-ssh-key-2026-04-09.key ubuntu@140.245.65.47
```

---

## 서버 초기 세팅 (2026-04-09 ~ 2026-04-11)

### 1. 패키지 설치

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git iptables-persistent
```

### 2. 프로젝트 클론

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/Jaemnani/lotto645.git
cd lotto645
```

### 3. GitHub PAT 설정 (push/pull 인증)

```bash
git config --global credential.helper store
git remote set-url origin https://Jaemnani:TOKEN@github.com/Jaemnani/lotto645.git
```

### 4. Python 가상환경 및 패키지 설치

```bash
cd ~/workspace/lotto645
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install supabase
```

### 5. .env 파일 생성

```bash
cat > .env << 'EOF'
SUPABASE_URL=https://ytspwwbbhkceexogvhkz.supabase.co
SUPABASE_KEY=eyJ...   # Supabase anon public key
ADMIN_KEY=5c767349c1a12e6cabc10ce3dda301bd4c8c0a8197ffa11d1c17590f1b098c2e
HOST=0.0.0.0
PORT=8000
EOF
```

### 6. 모델 및 데이터 파일 복사 (로컬 → 서버)

gitignore 대상 파일이므로 scp로 직접 복사:

```bash
# 로컬 맥에서 실행
scp -i ~/.ssh/oracle-ssh-key-2026-04-09.key \
  /Users/ohyeahdani_m1/workspace/lotto645/model_m02_claude/best_m02.pth \
  ubuntu@140.245.65.47:~/workspace/lotto645/model_m02_claude/

scp -i ~/.ssh/oracle-ssh-key-2026-04-09.key \
  /Users/ohyeahdani_m1/workspace/lotto645/data/history_from_cafe.csv \
  ubuntu@140.245.65.47:~/workspace/lotto645/data/
```

---

## systemd 데몬 등록

### 서비스 파일 생성

```bash
sudo nano /etc/systemd/system/lotto645.service
```

```ini
[Unit]
Description=Lotto645 FastAPI Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/workspace/lotto645
EnvironmentFile=/home/ubuntu/workspace/lotto645/.env
ExecStart=/home/ubuntu/workspace/lotto645/.venv/bin/python run_web.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 서비스 시작

```bash
sudo systemctl daemon-reload
sudo systemctl enable lotto645
sudo systemctl start lotto645
sudo systemctl status lotto645
```

### 서비스 관리 명령어

```bash
sudo systemctl restart lotto645   # 재시작
sudo systemctl stop lotto645      # 중지
sudo systemctl status lotto645    # 상태 확인
journalctl -u lotto645 -f         # 실시간 로그
```

---

## 방화벽 설정

### Oracle Cloud 보안 목록

네트워킹 → 가상 클라우드 네트워크 → lotto645-vcn → 보안 목록 → Default Security List → 수신 규칙 추가

- 소스 CIDR: `0.0.0.0/0`
- IP 프로토콜: TCP
- 대상 포트 범위: `8000`

### Ubuntu iptables

```bash
sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
sudo netfilter-persistent save   # 재부팅 후에도 유지
```

---

## 배포 흐름

### 코드 변경 시

```bash
# 로컬에서
git add .
git commit -m "..."
git push origin master

# 서버에서
cd ~/workspace/lotto645
git pull origin master
sudo systemctl restart lotto645
```

### 프론트엔드 빌드 시

```bash
# 로컬에서
cd frontend
npm run build          # web/static/ 에 빌드됨
cd ..
git add web/static/ frontend/src/
git commit -m "feat: 프론트 빌드 배포"
git push origin master

# 서버에서
git pull origin master
sudo systemctl restart lotto645
```

---

## 현재 동작 중인 기능

| 기능 | 상태 |
|------|------|
| FastAPI 백엔드 | ✅ 실행 중 (systemd) |
| LSTM 모델 번호 추출 | ✅ 동작 확인 |
| Supabase DB 연결 | ✅ 동작 확인 |
| 번호 저장 (user_extractions) | ✅ 동작 확인 |
| 토요일 21:05 자동 통계 스케줄러 | ✅ 등록 완료 |
| 공지사항 팝업 | ✅ 배포 완료 (추첨 후 동작 예정) |

---

## 자동화 스케줄

| 시점 | 동작 | 방식 |
|------|------|------|
| 매주 금요일 11:00 | 카페 크롤링 + Supabase 동기화 | 로컬 crontab (아이맥/맥미니) |
| 매주 토요일 21:05 | 추첨 결과 fetch + 통계 계산 | 서버 내 APScheduler |

---

## 접속 주소

```
http://140.245.65.47:8000
```
