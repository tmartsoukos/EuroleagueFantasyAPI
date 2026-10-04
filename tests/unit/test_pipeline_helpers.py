"""Tests των βοηθητικών συναρτήσεων του ingest/pipeline.py (χωρίς βάση και χωρίς δίκτυο)."""

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from elfantasy.ingest import pipeline


class TestSeasons:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("2016-2026", list(range(2016, 2027))),
            ("2025", [2025]),
            ("2024,2025", [2024, 2025]),
            ("2018, 2016-2017", [2016, 2017, 2018]),
            ("2020,2020", [2020]),
        ],
    )
    def test_parse_seasons(self, text, expected):
        assert pipeline.parse_seasons(text) == expected

    @pytest.mark.parametrize("text", ["", "2026-2016", "abc", "2016-"])
    def test_invalid_seasons(self, text):
        with pytest.raises(ValueError):
            pipeline.parse_seasons(text)

    @pytest.mark.parametrize(
        ("today", "expected"),
        [
            (date(2026, 10, 3), 2026),  # η σεζόν 2026-27 ξεκίνησε στις 24/09/2026
            (date(2027, 3, 1), 2026),
            (date(2027, 7, 31), 2026),
            (date(2027, 8, 1), 2027),
            (date(2026, 1, 15), 2025),
        ],
    )
    def test_current_season(self, today, expected):
        assert pipeline.current_season(today) == expected


class TestRecords:
    def test_python_values_convert_missing_numbers_and_timestamps(self):
        frame = pd.DataFrame(
            {
                "integers": [1, 2],
                "nullable": pd.array([3, None], dtype="Int64"),
                "floats": [1.5, np.nan],
                "text": ["a", None],
                "flags": [True, False],
                "when": pd.to_datetime(["2026-10-07 18:45", None]),
                "day": [date(2026, 10, 7), date(2026, 10, 8)],
            }
        )
        records = pipeline.to_records(frame, list(frame.columns))
        assert records[0] == {
            "integers": 1,
            "nullable": 3,
            "floats": 1.5,
            "text": "a",
            "flags": True,
            "when": datetime(2026, 10, 7, 18, 45),
            "day": date(2026, 10, 7),
        }
        assert records[1]["nullable"] is None
        assert records[1]["floats"] is None
        assert records[1]["text"] is None
        assert records[1]["when"] is None
        assert records[1]["flags"] is False

    def test_types_are_native_python(self):
        frame = pd.DataFrame({"a": np.array([1, 2], dtype="int64"), "b": np.array([True, False])})
        record = pipeline.to_records(frame, ["a", "b"])[0]
        assert type(record["a"]) is int
        assert type(record["b"]) is bool

    def test_only_the_requested_columns_are_kept_in_order(self):
        frame = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        assert list(pipeline.to_records(frame, ["c", "a"])[0]) == ["c", "a"]

    def test_empty_frame(self):
        assert pipeline.to_records(pd.DataFrame({"a": []}), ["a"]) == []


class TestScores:
    def test_add_scores_uses_the_scoring_module(self):
        frame = pd.DataFrame(
            {
                "points": [14, 0],
                "total_reb": [4, 4],
                "assists": [5, 0],
                "steals": [0, 1],
                "blocks_favour": [0, 0],
                "fouls_received": [4, 0],
                "fg2_made": [5, 0],
                "fg2_attempted": [8, 1],
                "fg3_made": [0, 0],
                "fg3_attempted": [4, 2],
                "ft_made": [4, 0],
                "ft_attempted": [6, 0],
                "turnovers": [2, 4],
                "blocks_against": [1, 1],
                "fouls_committed": [3, 1],
                "won": [True, True],
            }
        )
        scored = pipeline.add_scores(frame)
        assert scored["pir"].tolist() == [12, -4]  # Larkin και Hazer (FANTASY_RULES.md, §7)
        assert scored["fantasy_score"].tolist() == [13.2, -3.6]
        assert "pir" not in frame.columns  # δεν τροποποιεί την είσοδο


class TestCommandLine:
    def test_rps_above_the_maximum_is_rejected(self, capsys):
        with pytest.raises(SystemExit) as error:
            pipeline.main(["--rps", "2.5"])
        assert error.value.code == 2
        assert "--rps" in capsys.readouterr().err

    def test_invalid_seasons_are_rejected(self, capsys):
        with pytest.raises(SystemExit) as error:
            pipeline.main(["--seasons", "2026-2016"])
        assert error.value.code == 2

    def test_parser_defaults(self):
        args = pipeline.build_parser().parse_args([])
        assert args.rps == 1.0
        assert not args.update and not args.no_fetch
        assert args.seasons is None and args.db is None
