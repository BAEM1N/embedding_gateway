# Embedding Gateway

Ollama, [TEI (Text Embeddings Inference)](https://github.com/huggingface/text-embeddings-inference), [vLLM](https://github.com/vllm-project/vllm)을 하나의 OpenAI 호환 API로 묶어주는 임베딩 게이트웨이.

각 백엔드는 로컬이든 원격이든 상관없이 `BASE_URL`만 지정하면 사용 가능하며, `DOCKER_IMAGE`를 설정하면 로컬 Docker 컨테이너를 자동 관리(모델 스와핑)합니다.

## 구조

```
                            ┌──────────────┐
                            │    Client    │
                            └──────┬───────┘
                                   │  POST /v1/embeddings
                            ┌──────▼───────┐
                            │   Gateway    │  :8000
                            │   (FastAPI)  │
                            └──┬─────┬──┬──┘
                               │     │  │
              ┌────────────────▼─┐ ┌─▼──▼────────────────┐
              │     Ollama       │ │  TEI       vLLM      │
              │    :11434        │ │ :8080     :8081      │
              │   (always remote)│ │ (managed / remote)   │
              └──────────────────┘ └──────────────────────┘
```

- **Gateway** (FastAPI) -- 모델 이름 기반으로 요청을 적절한 백엔드로 라우팅
- **Ollama** -- 항상 원격 모드. GGUF 양자화 모델 (`bge-m3`, `qwen3-embedding`, Jina v4 등)
- **TEI** -- HuggingFace safetensors fp16 모델 (`multilingual-e5`, `KURE-v1` 등)
- **vLLM** -- TEI가 지원하지 못하는 모델용 (`jina-embeddings-v3` 등)

### Managed vs Remote 모드

TEI와 vLLM 백엔드는 두 가지 모드를 지원합니다:

| 모드 | 조건 | 동작 |
|------|------|------|
| **Managed** | `DOCKER_IMAGE` 설정 | 로컬 Docker 컨테이너 자동 관리 (시작/중지/모델 스왑) |
| **Remote** | `DOCKER_IMAGE` 비움 | 원격 서버에 HTTP 프록시만 수행 (Docker 관리 없음) |

이를 통해 게이트웨이를 PC-A에서 실행하고, TEI는 PC-B, vLLM은 PC-C, Ollama는 PC-D에서 각각 운영하는 분산 구성이 가능합니다.

## 요구사항

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) (로컬 또는 원격)
- Docker + NVIDIA Container Toolkit (managed 모드 사용 시)

## 설치

```bash
git clone https://github.com/BAEM1N/embedding_gateway.git
cd embedding_gateway

# 의존성 설치
uv sync

# 환경변수 설정
cp .env.example .env
# .env 파일을 편집하여 백엔드 URL, 모델 목록 등을 설정
```

## 모델 준비

### Ollama 모델

```bash
# 기본 다국어 임베딩 모델
ollama pull bge-m3

# 전체 모델 일괄 다운로드
bash scripts/pull-ollama-models.sh

# Jina v4 GGUF (Q8_0 양자화)
ollama pull hf.co/jinaai/jina-embeddings-v4-text-retrieval-GGUF:Q8_0
```

### TEI (Docker, managed 모드)

```bash
# docker-compose로 TEI 시작 (GPU)
wsl -d Ubuntu-24.04 -- docker compose -f docker/docker-compose.yml up -d

# CPU only
wsl -d Ubuntu-24.04 -- docker compose -f docker/docker-compose.yml \
  -f docker/docker-compose.cpu.yml up -d

# 여러 모델을 미리 캐시 (선택)
bash scripts/prefetch-tei-models.sh
```

### vLLM (Docker, managed 모드)

```bash
# .env에서 vLLM 활성화
VLLM_MODELS=jinaai/jina-embeddings-v3
VLLM_DOCKER_IMAGE=vllm/vllm-openai:latest

# 게이트웨이가 자동으로 vLLM 컨테이너를 관리합니다.
# 수동 실행도 가능:
wsl -d Ubuntu-24.04 -- docker run -d \
  --name vllm-embeddings --gpus all \
  -p 8081:8000 \
  -v vllm-model-cache:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model jinaai/jina-embeddings-v3 \
  --task embed --dtype float16 --max-model-len 8192
```

## 실행

```bash
# Gateway 서버 시작
uv run embedding-gateway

# 또는 전체 서비스 한번에 (Ollama 확인 + TEI + vLLM + Gateway)
bash scripts/start-all.sh
```

## API 사용

OpenAI Embeddings API와 동일한 인터페이스:

```bash
# Ollama 백엔드 (bge-m3)
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["안녕하세요", "Hello"], "model": "bge-m3"}'

# TEI 백엔드 (multilingual-e5)
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello", "model": "intfloat/multilingual-e5-large-instruct"}'

# vLLM 백엔드 (Jina v3)
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello", "model": "jinaai/jina-embeddings-v3"}'

# 사용 가능한 모델 목록
curl http://localhost:8000/v1/models

# 서비스 상태 확인
curl http://localhost:8000/health

# Readiness probe (하나 이상의 백엔드가 healthy이면 ready)
curl http://localhost:8000/health/ready
```

## Playground

웹 브라우저에서 임베딩을 테스트하고 모델 간 비교를 할 수 있는 UI:

```
http://localhost:8000/playground
```

- **Embed** 탭: 단일 모델로 임베딩 생성, 배치 모드, 코사인 유사도 매트릭스
- **Compare** 탭: 여러 모델을 선택하여 동일 텍스트의 임베딩 결과를 나란히 비교

## 포트 구성

| 서비스 | 포트 | 설명 |
|--------|------|------|
| Gateway | 8000 | 통합 API 엔드포인트 |
| Ollama | 11434 | Ollama 기본 포트 |
| TEI | 8080 | Text Embeddings Inference |
| vLLM | 8081 | vLLM OpenAI-compatible API |

## 원격 백엔드 구성 예시

게이트웨이만 로컬에서 실행하고, 백엔드는 다른 PC에서 운영:

```env
# Gateway (이 PC에서 실행)
GATEWAY_HOST=0.0.0.0
GATEWAY_PORT=8000

# Ollama (PC-A)
OLLAMA_BASE_URL=http://192.168.1.100:11434

# TEI (PC-B) -- DOCKER_IMAGE를 비우면 원격 모드
TEI_BASE_URL=http://192.168.1.200:8080
TEI_DOCKER_IMAGE=

# vLLM (PC-C) -- DOCKER_IMAGE를 비우면 원격 모드
VLLM_BASE_URL=http://192.168.1.300:8081
VLLM_DOCKER_IMAGE=
```

## 지원 모델 요약

| 모델 | 백엔드 | 비고 |
|------|--------|------|
| bge-m3, bge-large | Ollama | GGUF 양자화 |
| qwen3-embedding (0.6b/4b/8b) | Ollama | GGUF 양자화 |
| nomic-embed-text (v1.5/v2-moe) | Ollama | GGUF |
| snowflake-arctic-embed | Ollama | GGUF |
| jina-embeddings-v4 | Ollama | GGUF Q8_0 |
| multilingual-e5-base/large | TEI | safetensors fp16 |
| KURE-v1 | TEI | safetensors fp16 |
| jina-embeddings-v3 | vLLM | fp16, `--task embed` |

> Jina v5는 transformers >= 5.0.0이 필요하여 현재 vLLM/Ollama 모두 미지원.

## Docker GPU 참고

Rancher Desktop은 Windows에서 GPU 패스스루를 지원하지 않음. WSL2 Ubuntu에 Docker Engine을 직접 설치해야 합니다:

```bash
# WSL2 Ubuntu 설치 후
bash docker/setup-wsl-docker.sh
```

자세한 내용은 `docker/setup-wsl-docker.sh` 참고.

## 벤치마크

```bash
# 전체 모델 레이턴시 벤치마크
python scripts/benchmark.py
```

결과는 `scripts/benchmark_result.json`에 저장됩니다.

## 테스트

```bash
uv run pytest tests/ -v
```

## License

MIT
