import pytest


@pytest.mark.asyncio
async def test_health_reports_backend_status(client):
    response = await client.get("/health")
    data = response.json()
    # All three backends should be reported (even if unhealthy when offline)
    assert "ollama" in data["backends"]
    assert "tei" in data["backends"]
    assert "vllm" in data["backends"]
    for backend_info in data["backends"].values():
        assert "status" in backend_info
