"""Το API δεν γράφει ποτέ στον πίνακα `predictions` (Φάση 5): τα GET δεν έχουν παρενέργειες.

Η καταγραφή προβλέψεων γίνεται ΜΟΝΟ από την εντολή `python -m elfantasy.model.record_predictions`.
"""

from api_support import ADMIN_HEADERS
from sqlalchemy import func, select

from elfantasy.db import models


def predictions_count(engine) -> int:
    with engine.connect() as connection:
        return connection.execute(select(func.count()).select_from(models.predictions)).scalar_one()


def test_no_endpoint_writes_predictions(client, api_engine):
    before = predictions_count(api_engine)
    first = client.get("/rankings?limit=1").json()["items"][0]["player_id"]
    paths = [
        "/health",
        "/rankings",
        "/rankings?limit=500&active_only=false",
        "/players?search=player",
        "/availability",
        f"/predict/{first}",
        f"/predict/{first}?include_features=true",
        "/docs",
        "/openapi.json",
    ]
    for path in paths:
        assert client.get(path).status_code == 200, path
    assert client.post("/admin/refresh", headers=ADMIN_HEADERS).status_code == 200
    assert predictions_count(api_engine) == before == 0
