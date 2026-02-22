#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Embedding Services 시작 ==="

# 1. Ollama 확인
echo ""
echo "[1/4] Ollama 확인 중..."
if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "  Ollama 실행 중."
else
    echo "  Ollama가 실행되고 있지 않습니다."
    echo "  Windows에서 Ollama를 먼저 시작해 주세요."
    echo "  (시스템 트레이에서 Ollama 아이콘 확인 또는 'ollama serve' 실행)"
fi

# 2. TEI (Docker) 시작
echo ""
echo "[2/4] TEI 시작 중..."
echo "  WSL2 Docker를 사용하여 TEI를 시작합니다."
echo "  (첫 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다)"

# WSL2 Docker를 사용하는 경우
if command -v wsl &> /dev/null; then
    wsl docker compose -f "$(wslpath "$PROJECT_DIR/docker/docker-compose.yml")" up -d 2>/dev/null || \
    docker compose -f "$PROJECT_DIR/docker/docker-compose.yml" up -d
else
    docker compose -f "$PROJECT_DIR/docker/docker-compose.yml" up -d
fi

# TEI 헬스 체크 대기
echo "  TEI 로딩 대기 중..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "  TEI 준비 완료."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "  TEI가 아직 로딩 중입니다. 'docker logs tei-embeddings'로 상태를 확인하세요."
    fi
    sleep 5
done

# 3. vLLM 확인 (VLLM_MODELS가 설정된 경우)
echo ""
echo "[3/4] vLLM 확인 중..."
if curl -s http://localhost:8081/health > /dev/null 2>&1; then
    echo "  vLLM 실행 중."
    MODEL=$(curl -s http://localhost:8081/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')" 2>/dev/null || echo "unknown")
    echo "  현재 모델: $MODEL"
else
    echo "  vLLM이 실행되고 있지 않습니다."
    echo "  managed 모드: 게이트웨이가 첫 요청 시 자동으로 컨테이너를 시작합니다."
    echo "  remote 모드: 원격 vLLM 서버를 먼저 시작해 주세요."
fi

# 4. Gateway 시작
echo ""
echo "[4/4] Embedding Gateway 시작 중 (port 8000)..."
cd "$PROJECT_DIR"
uv run embedding-gateway
