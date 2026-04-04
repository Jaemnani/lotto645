"""
Oracle Cloud 서버 실행 엔트리포인트
사용법: python run_web.py
"""

import os
import uvicorn

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    reload = os.environ.get("DEV", "").lower() == "true"

    uvicorn.run(
        "web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
