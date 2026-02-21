import pytest
from unittest.mock import AsyncMock, patch

from embedding_gateway.models import EmbeddingData, EmbeddingResponse, UsageInfo


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "backends" in data


@pytest.mark.asyncio
async def test_health_ready_endpoint(client):
    response = await client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data


@pytest.mark.asyncio
async def test_models_endpoint(client):
    response = await client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)


@pytest.mark.asyncio
async def test_embeddings_unknown_model(client):
    response = await client.post(
        "/v1/embeddings",
        json={"input": "test", "model": "nonexistent-model-xyz"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_embeddings_with_mock_ollama(client):
    mock_response = EmbeddingResponse(
        data=[EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0)],
        model="bge-m3",
        usage=UsageInfo(prompt_tokens=5, total_tokens=5),
    )

    with patch(
        "embedding_gateway.backends.ollama.OllamaBackend.embed",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        response = await client.post(
            "/v1/embeddings",
            json={"input": "Hello world", "model": "bge-m3"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]
        assert data["model"] == "bge-m3"


@pytest.mark.asyncio
async def test_embeddings_batch_input(client):
    mock_response = EmbeddingResponse(
        data=[
            EmbeddingData(embedding=[0.1, 0.2], index=0),
            EmbeddingData(embedding=[0.3, 0.4], index=1),
        ],
        model="bge-m3",
        usage=UsageInfo(prompt_tokens=10, total_tokens=10),
    )

    with patch(
        "embedding_gateway.backends.ollama.OllamaBackend.embed",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        response = await client.post(
            "/v1/embeddings",
            json={"input": ["Hello", "World"], "model": "bge-m3"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2


@pytest.mark.asyncio
async def test_playground_page(client):
    response = await client.get("/playground")
    assert response.status_code == 200
    assert "Embedding Gateway Playground" in response.text
