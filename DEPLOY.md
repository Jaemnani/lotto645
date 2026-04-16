# 배포 가이드

## 전체 아키텍처

```
맥미니 (항상 켜짐)
  └── 기존 파이프라인 유지 (크롤링 → 학습 → GitHub push)

Oracle Cloud ARM A1 VM (무료, 영구)
  ├── FastAPI 서버 (번호 생성, API)
  ├── APScheduler (토요일 21:05 자동 통계)
  └── 주기적 git pull (최신 모델 동기화)

Supabase (무료)
  └── PostgreSQL DB (추첨결과, 사용자추출, 공지사항)
```

---

## 1. Supabase 설정

1. [supabase.com](https://supabase.com) 가입 → 새 프로젝트 생성
2. **Settings → Database → Connection string (URI)** 복사
3. `.env` 파일에 `DATABASE_URL`로 설정

---

## 2. Oracle Cloud VM 설정

### 인스턴스 생성
- Shape: **VM.Standard.A1.Flex** (ARM, Always Free)
- OCPU: 1, RAM: 6GB (또는 더 많이)
- OS: Ubuntu 22.04 Minimal

### 초기 설정
```bash
# 의존성 설치
sudo apt update && sudo apt install -y python3-pip python3-venv git nginx

# 프로젝트 클론
git clone https://github.com/YOUR_USERNAME/lotto645.git
cd lotto645

# 가상환경 + 패키지 설치
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_web.txt

# 환경변수 설정
cp .env.example .env
nano .env   # DATABASE_URL, ADMIN_KEY 입력
```

### 서버 실행 (systemd 서비스)
```bash
# /etc/systemd/system/lotto.service 생성
sudo nano /etc/systemd/system/lotto.service
```

```ini
[Unit]
Description=Lotto Web Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/lotto645
EnvironmentFile=/home/ubuntu/lotto645/.env
ExecStart=/home/ubuntu/lotto645/venv/bin/python run_web.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable lotto
sudo systemctl start lotto
sudo systemctl status lotto
```

### Nginx 리버스 프록시 (HTTPS)
```nginx
# /etc/nginx/sites-available/lotto
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/lotto /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# HTTPS (선택, 도메인 있을 때)
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d YOUR_DOMAIN
```

### 모델 자동 동기화 (매주 월요일)

Oracle 서버(Ubuntu)에서는 crontab 사용:
```bash
crontab -e
# 추가:
0 2 * * 1 /home/ubuntu/lotto645/sync_model.sh >> /var/log/lotto_sync.log 2>&1
```

### 파이프라인 자동 실행 (매주 금요일, 로컬 macOS)

`~/Library/LaunchAgents/com.lotto645.pipeline.plist` 파일 생성:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.lotto645.pipeline</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/ohyeahdani_m1/workspace/venv_common/bin/python</string>
        <string>/Users/ohyeahdani_m1/workspace/lotto645/pipeline.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/ohyeahdani_m1/workspace/lotto645</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>5</integer>
        <key>Hour</key>
        <integer>10</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/Users/ohyeahdani_m1/workspace/lotto645/logs/pipeline_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/ohyeahdani_m1/workspace/lotto645/logs/pipeline_stderr.log</string>
</dict>
</plist>
```

```bash
mkdir -p ~/workspace/lotto645/logs
launchctl load ~/Library/LaunchAgents/com.lotto645.pipeline.plist
launchctl list | grep lotto645   # 등록 확인
```

---

## 3. Oracle 방화벽 설정

Oracle Cloud 콘솔 → VCN → Security Lists → Ingress Rules 추가:
- Port 80 (HTTP)
- Port 443 (HTTPS, 선택)

VM 내부 방화벽:
```bash
sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

---

## 4. 프론트엔드 빌드

```bash
cd frontend
npm install
npm run build   # → web/static/ 에 빌드됨 (FastAPI가 자동 서빙)
```

개발 시 API URL 설정 (`.env.local`):
```
VITE_API_URL=http://YOUR_ORACLE_IP:8000
```

---

## 5. 관리자 API

토요일 수동 트리거 (자동 스케줄러 외):
```bash
curl -X POST https://YOUR_DOMAIN/api/admin/fetch-and-calc \
  -H "X-Admin-Key: YOUR_ADMIN_KEY"
```
