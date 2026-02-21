# Embedding Gateway

Ollama와 [Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference)를 하나의 OpenAI 호환 API로 묶어주는 경량 게이트웨이.

## 구조

```
                        ┌──────────────┐
                        │   Client     │
                        └──────┬───────┘
                               │  POST /v1/embeddings
                        ┌──────▼───────┐
                        │   Gateway    │  :8000
                        └──┬───────┬───┘
                           │       │
                ┌──────────▼──┐ ┌──▼──────────┐
                │   Ollama    │ │     TEI      │
                │   :11434    │ │    :8080     │
                └─────────────┘ └──────────────┘
```

- **Gateway** (FastAPI) — 모델 이름 기반으로 요청을 적절한 백엔드로 라우팅
- **Ollama** — 로컬 실행, `bge-m3` 등 다국어 임베딩 모델
- **TEI** — Docker (GPU), `multilingual-e5-large-instruct` 등

## 요구사항

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- Docker (GPU 사용 시 NVIDIA Container Toolkit 필요)

## 설치

```bash
git clone https://github.com/BAEM1N/embedding_gateway.git
cd embedding_gateway

# 의존성 설치
uv sync

# 환경변수 설정
cp .env.example .env
```

## 모델 준비

```bash
# Ollama 임베딩 모델 다운로드
ollama pull bge-m3

# TEI 컨테이너 실행 (GPU)
docker compose -f docker/docker-compose.yml up -d

# TEI 컨테이너 실행 (CPU only)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.cpu.yml up -d
```

## 실행

```bash
# Gateway 서버 시작
uv run embedding-gateway

# 또는 전체 서비스 한번에
bash scripts/start-all.sh
```

## API 사용

OpenAI Embeddings API와 동일한 인터페이스:

```bash
# Ollama 백엔드 (bge-m3)
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["안녕하세요", "Hello"], "model": "bge-m3"}'

# TEI 백엔드
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["안녕하세요", "Hello"], "model": "intfloat/multilingual-e5-large-instruct"}'

# 사용 가능한 모델 목록
curl http://localhost:8000/v1/models

# 서비스 상태 확인
curl http://localhost:8000/health
```

각 백엔드에 직접 접근도 가능:
- Ollama: `http://localhost:11434`
- TEI: `http://localhost:8080`

## 포트 구성

| 서비스 | 포트 | 설명 |
|--------|------|------|
| Gateway | 8000 | 통합 API 엔드포인트 |
| Ollama | 11434 | Ollama 기본 포트 |
| TEI | 8080 | Text Embeddings Inference |

## Docker GPU 참고

Rancher Desktop은 Windows에서 GPU 패스스루를 지원하지 않음. WSL2 Ubuntu에 Docker Engine을 직접 설치해야 함:

```bash
# WSL2 Ubuntu 설치 후
bash docker/setup-wsl-docker.sh
```

자세한 내용은 `docker/setup-wsl-docker.sh` 참고.

## 테스트

```bash
uv run pytest tests/ -v
```

## License

MIT
