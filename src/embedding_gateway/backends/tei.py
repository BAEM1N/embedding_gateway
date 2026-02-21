import asyncio
import logging
import subprocess
import traceback

import httpx

from embedding_gateway.backends.base import EmbeddingBackend
from embedding_gateway.models import EmbeddingData, EmbeddingResponse, UsageInfo

logger = logging.getLogger(__name__)


class TEIBackend(EmbeddingBackend):
    def __init__(
        self,
        base_url: str,
        default_model: str,
        available_models: list[str],
        docker_image: str,
        container_name: str = "tei-embeddings",
        wsl_distro: str = "Ubuntu-24.04",
        swap_timeout: float = 120.0,
        timeout: float = 120.0,
        hf_token: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.available_models = available_models
        self.docker_image = docker_image
        self.container_name = container_name
        self.wsl_distro = wsl_distro
        self.swap_timeout = swap_timeout
        self.hf_token = hf_token
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        self.current_model: str | None = None
        self._swap_lock = asyncio.Lock()

    async def _detect_current_model(self) -> str | None:
        """TEI /info 엔드포인트에서 현재 로딩된 모델 확인."""
        try:
            r = await self.client.get("/info", timeout=5.0)
            if r.status_code == 200:
                return r.json().get("model_id")
        except Exception:
            pass
        return None

    async def initialize(self) -> None:
        """시작 시 현재 TEI 컨테이너의 모델을 감지."""
        self.current_model = await self._detect_current_model()
        if self.current_model:
            logger.info(f"TEI current model: {self.current_model}")
        else:
            logger.info("TEI container not running or not healthy")

    def _docker_cmd(self, *args: str) -> list[str]:
        return ["wsl", "-d", self.wsl_distro, "--", "docker", *args]

    async def _run_cmd(
        self, cmd: list[str], timeout: float = 30.0
    ) -> tuple[int, str, str]:
        """Run a command using subprocess.run in a thread (Windows-safe)."""
        cmd_str = " ".join(cmd)
        logger.debug(f"Running: {cmd_str}")

        def _sync_run() -> subprocess.CompletedProcess:
            return subprocess.run(
                cmd, capture_output=True, timeout=timeout
            )

        try:
            result = await asyncio.to_thread(_sync_run)
            stdout = result.stdout.decode(errors="replace")
            stderr = result.stderr.decode(errors="replace")
            if result.returncode != 0:
                logger.warning(
                    f"Command failed (rc={result.returncode}): {cmd_str}\n"
                    f"stderr: {stderr}"
                )
            else:
                logger.debug(f"Command OK (rc=0): {cmd_str}")
            return result.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out ({timeout}s): {cmd_str}")
            return -1, "", "timeout"
        except Exception as e:
            logger.error(f"Command exception: {cmd_str} → {e}")
            return -2, "", str(e)

    async def _swap_model(self, model_id: str) -> None:
        """컨테이너를 교체하여 다른 모델 로딩."""
        async with self._swap_lock:
            # Lock 획득 후 다시 확인 (다른 요청이 이미 swap 했을 수 있음)
            if model_id == self.current_model:
                return

            logger.info(f"Swapping TEI model: {self.current_model} → {model_id}")

            # 1. 기존 컨테이너 제거
            rc, _, stderr = await self._run_cmd(
                self._docker_cmd("rm", "-f", self.container_name),
                timeout=15.0,
            )
            if rc != 0:
                logger.warning(f"Container remove returned rc={rc}: {stderr}")

            # 2. 새 컨테이너 시작
            env_args: list[str] = []
            if self.hf_token:
                env_args = ["-e", f"HUGGING_FACE_HUB_TOKEN={self.hf_token}"]

            run_cmd = self._docker_cmd(
                "run", "-d",
                "--name", self.container_name,
                "--gpus", "all",
                "-p", "8080:80",
                "-v", "tei-model-cache:/data",
                *env_args,
                self.docker_image,
                "--model-id", model_id,
                "--dtype", "float16",
                "--max-batch-tokens", "16384",
                "--max-concurrent-requests", "64",
            )
            rc, stdout, stderr = await self._run_cmd(run_cmd, timeout=30.0)
            if rc != 0:
                raise RuntimeError(
                    f"Failed to start TEI container for {model_id} "
                    f"(rc={rc}): {stderr.strip()}"
                )

            logger.info(f"TEI container started, waiting for health...")

            # 3. health 대기
            await self._wait_healthy()
            self.current_model = model_id
            logger.info(f"TEI model swapped to: {model_id}")

    async def _wait_healthy(self) -> None:
        """TEI가 healthy 될 때까지 대기."""
        deadline = asyncio.get_event_loop().time() + self.swap_timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = await self.client.get("/health", timeout=5.0)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(2.0)
        raise TimeoutError(
            f"TEI did not become healthy within {self.swap_timeout}s"
        )

    async def embed(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None = None,
    ) -> EmbeddingResponse:
        # 모델이 다르면 교체
        if model != self.current_model:
            if model not in self.available_models:
                raise ValueError(f"Model '{model}' not in available TEI models")
            logger.info(
                f"TEI model switch: {self.current_model} → {model}"
            )
            await self._swap_model(model)

        try:
            response = await self.client.post(
                "/v1/embeddings",
                json={"input": texts, "model": model},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response else ""
            raise RuntimeError(
                f"TEI returned HTTP {e.response.status_code}: {body}"
            ) from e

        data = response.json()

        embeddings_data = []
        for d in data["data"]:
            emb = d["embedding"]
            if dimensions:
                emb = emb[:dimensions]
            embeddings_data.append(EmbeddingData(embedding=emb, index=d["index"]))

        return EmbeddingResponse(
            data=embeddings_data,
            model=model,
            usage=UsageInfo(
                prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                total_tokens=data.get("usage", {}).get("total_tokens", 0),
            ),
        )

    async def health_check(self) -> dict:
        try:
            r = await self.client.get("/health", timeout=5.0)
            return {
                "status": "healthy" if r.status_code == 200 else "unhealthy",
                "current_model": self.current_model,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "current_model": self.current_model,
            }

    async def list_models(self) -> list[str]:
        return list(self.available_models)

    async def close(self) -> None:
        await self.client.aclose()
