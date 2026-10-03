"""Tests του `GET /players`: αναζήτηση χωρίς διάκριση πεζών/κεφαλαίων και τόνων, φίλτρο ομάδας,
όρια και ταξινόμηση. Τα ονόματα προέρχονται από τα πραγματικά fixtures (π.χ. `BRISSETT, O'SHAE J`,
`LAWSON, A.J.`, `LUWAWU-CABARROT, TIMOTHE`).
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from elfantasy.db import models


def ids(response) -> list[str]:
    assert response.status_code == 200, response.text
    return [item["player_id"] for item in response.json()["items"]]


class TestSearch:
    def test_a_player_is_found_by_a_part_of_the_name(self, real_client):
        response = real_client.get("/players?search=vezenkov")
        assert ids(response) == ["P003469"]
        assert response.json() == {
            "total": 1,
            "limit": 20,
            "items": [
                {
                    "player_id": "P003469",
                    "name": "VEZENKOV, SASHA",
                    "team_code": "OLY",
                    "team_name": response.json()["items"][0]["team_name"],
                    "last_season": 2026,
                    "is_active": True,
                }
            ],
        }
        assert response.json()["items"][0]["team_name"]  # από τον πίνακα teams

    @pytest.mark.parametrize(
        "query",
        ["VEZENKOV", "VeZeNkOv", "vezen", "zenkov", "vezenkóv", "VEZENKÓV", "sasha vezenkov"],
    )
    def test_case_accents_substrings_and_the_order_of_the_words_do_not_matter(
        self, real_client, query
    ):
        assert ids(real_client.get("/players", params={"search": query})) == ["P003469"]

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("šimonović", "PLRU"),  # τόνοι και caron: SIMONOVIC
            ("o'shae", "P014127"),  # BRISSETT, O'SHAE J
            ("oshae brissett", "P014127"),
            ("aj lawson", "P014734"),  # LAWSON, A.J.
            ("a.j.", "P014734"),
            ("luwawu cabarrot", "P012080"),  # LUWAWU-CABARROT
            ("luwawu-cabarrot", "P012080"),
            ("saint supery", "P011706"),
        ],
    )
    def test_punctuation_in_the_names_does_not_get_in_the_way(self, real_client, query, expected):
        assert expected in ids(real_client.get("/players", params={"search": query}))

    def test_every_word_of_the_query_must_match(self, real_client):
        assert ids(real_client.get("/players?search=vezenkov+hoard")) == []

    def test_no_match_is_an_empty_result(self, real_client):
        body = real_client.get("/players?search=zzzzzzzz").json()
        assert body == {"total": 0, "limit": 20, "items": []}

    def test_a_blank_search_behaves_like_no_search(self, real_client):
        everybody = real_client.get("/players?limit=100").json()["total"]
        assert real_client.get("/players?search=%20%20").json()["total"] == everybody
        assert real_client.get("/players?search=").json()["total"] == everybody

    def test_a_search_that_is_too_long_is_422(self, real_client):
        assert real_client.get(f"/players?search={'a' * 101}").status_code == 422
        assert real_client.get(f"/players?search={'a' * 100}").status_code == 200


class TestPlayersData:
    def test_an_inactive_player_is_listed_with_the_team_of_the_last_row(self, real_client):
        body = real_client.get("/players?search=larkin").json()
        (item,) = body["items"]
        assert item["player_id"] == "P007200"
        assert item["team_code"] == "IST"
        assert item["last_season"] == 2025
        assert item["is_active"] is False

    def test_old_style_ids_are_listed(self, real_client):
        items = real_client.get("/players?search=arslan").json()["items"]
        assert [item["player_id"] for item in items] == ["PARN"]
        assert items[0]["name"] == "ARSLAN, ENDER"

    def test_the_active_players_come_first_then_the_names_in_order(self, real_client):
        items = real_client.get("/players?limit=100").json()["items"]
        flags = [item["is_active"] for item in items]
        assert flags == sorted(flags, reverse=True)  # πρώτα όλοι οι ενεργοί
        assert any(flags) and not all(flags)
        for group in (True, False):
            names = [item["name"] for item in items if item["is_active"] is group]
            assert names == sorted(names)

    def test_the_number_of_players_matches_the_database(self, real_client, fixture_database_url):
        from sqlalchemy import func

        from elfantasy.db.session import get_engine

        engine = get_engine(fixture_database_url)
        with engine.connect() as connection:
            total = connection.execute(
                select(func.count()).select_from(models.players)
            ).scalar_one()
        engine.dispose()
        assert real_client.get("/players").json()["total"] == total


class TestLimitAndTeam:
    def test_the_default_limit_is_20(self, real_client):
        body = real_client.get("/players").json()
        assert body["limit"] == 20
        assert len(body["items"]) == 20
        assert body["total"] > 20

    def test_a_smaller_and_the_maximum_limit(self, real_client):
        assert len(real_client.get("/players?limit=3").json()["items"]) == 3
        assert len(real_client.get("/players?limit=100").json()["items"]) == 100

    @pytest.mark.parametrize("limit", ["0", "-1", "101", "abc", ""])
    def test_a_limit_out_of_range_is_422(self, real_client, limit):
        assert real_client.get(f"/players?limit={limit}").status_code == 422

    def test_the_team_filter(self, real_client):
        upper = real_client.get("/players?team=TEL&limit=100").json()
        lower = real_client.get("/players?team=tel&limit=100").json()
        assert upper == lower
        assert upper["total"] > 0
        assert {item["team_code"] for item in upper["items"]} == {"TEL"}

    def test_the_team_and_the_name_combine(self, real_client):
        assert ids(real_client.get("/players?team=OLY&search=vezenkov")) == ["P003469"]
        assert ids(real_client.get("/players?team=TEL&search=vezenkov")) == []

    def test_an_unknown_team_is_404_with_the_valid_codes(self, real_client):
        response = real_client.get("/players?team=ZZZ")
        assert response.status_code == 404
        detail = response.json()["detail"]
        assert detail.startswith("unknown team code 'ZZZ'; valid codes: ")
        assert "OLY" in detail and "TEL" in detail

    @pytest.mark.parametrize("team", ["AB", "ABCD", "T-1", "%20OLY"])
    def test_a_malformed_team_is_422(self, real_client, team):
        assert real_client.get(f"/players?team={team}").status_code == 422

    def test_the_synthetic_league_works_the_same(self, client, synthetic_league):
        response = client.get("/players?search=player%201&limit=100")
        assert response.status_code == 200
        names = [item["name"] for item in response.json()["items"]]
        # κάθε λέξη («player», «1») πρέπει να εμφανίζεται κάπου στο όνομα
        expected = [name for name in synthetic_league.players["name"] if "1" in name]
        assert sorted(names) == sorted(expected)
        assert response.json()["total"] == len(expected)
