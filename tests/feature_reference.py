"""Αργή, «αφελής» υλοποίηση αναφοράς των features, για διασταύρωση του `build_features`.

Η υλοποίηση αυτή δεν μοιράζεται κώδικα με το `elfantasy.features.build`: για κάθε γραμμή
φιλτράρει ρητά τους αγώνες που έγιναν αυστηρά πριν (με βρόχους και απλό pandas/numpy) και
υπολογίζει κάθε feature από τον ορισμό του. Είναι O(γραμμές) ανά γραμμή, άρα χρησιμοποιείται σε
δείγμα γραμμών μικρών datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NaN = float("nan")


def sort_time(tipoff, game_date) -> pd.Timestamp:
    """Χρόνος ταξινόμησης: η ώρα έναρξης, αλλιώς το τέλος της ημέρας του αγώνα."""
    if pd.notna(tipoff):
        return pd.Timestamp(tipoff)
    return pd.Timestamp(game_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


def _mean_or_nan(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else NaN


def _is_appearance(row) -> bool:
    return (not row.dnp) and row.minutes > 0


class Reference:
    """Υπολογίζει τα features μιας γραμμής (παίκτης, αγώνας) από τον ορισμό τους."""

    def __init__(self, history: pd.DataFrame, games: pd.DataFrame, extra_games=()):
        self.history = history.copy()
        self.history["st"] = [
            sort_time(t, d)
            for t, d in zip(history["tipoff_utc"], history["game_date"], strict=True)
        ]
        # Όλοι οι αγώνες που ξέρουμε (παιγμένοι και τυχόν μελλοντικοί), με σκορ ή NaN.
        records = []
        for row in games.itertuples():
            records.append(
                {
                    "season": int(row.season),
                    "gamecode": int(row.gamecode),
                    "game_date": pd.Timestamp(row.game_date),
                    "st": sort_time(row.tipoff_utc, row.game_date),
                    "home": row.home_code,
                    "away": row.away_code,
                    "home_score": float(row.home_score),
                    "away_score": float(row.away_score),
                }
            )
        for row in extra_games:
            records.append(dict(row))
        self.games = pd.DataFrame(records)
        self.by_game = {
            key: group for key, group in self.history.groupby(["season", "gamecode"], sort=False)
        }
        self._team_cache: dict[str, pd.DataFrame] = {}
        self.league_events = self._league_events()

    # ----- ομάδες -----
    def team_games(self, team: str) -> pd.DataFrame:
        if team in self._team_cache:
            return self._team_cache[team]
        mask = (self.games["home"] == team) | (self.games["away"] == team)
        out = self.games[mask].sort_values(["st", "season", "gamecode"]).copy()
        out["team_score"] = np.where(out["home"] == team, out["home_score"], out["away_score"])
        out["opp_score"] = np.where(out["home"] == team, out["away_score"], out["home_score"])
        out["opp"] = np.where(out["home"] == team, out["away"], out["home"])
        self._team_cache[team] = out.reset_index(drop=True)
        return self._team_cache[team]

    def _box(self, season: int, gamecode: int, team: str):
        """(PIR, πλήρες;) της ομάδας σε έναν αγώνα, ή None αν δεν υπάρχει boxscore."""
        frame = self.by_game.get((season, gamecode))
        if frame is None:
            return None
        rows = frame[frame["team_code"] == team]
        if len(rows) == 0:
            return None
        minutes = float(rows["minutes"].sum())
        overtimes = max(round((minutes - 200.0) / 25.0), 0)
        ok = abs(minutes - (200.0 + 25.0 * overtimes)) <= 1.5
        return float(rows["pir"].sum()), ok

    def _allowed(self, season: int, gamecode: int, opp: str):
        """PIR που δέχτηκε μια ομάδα = PIR του αντιπάλου (μόνο αν το boxscore του είναι πλήρες)."""
        box = self._box(season, gamecode, opp)
        if box is None or not box[1]:
            return None
        return box[0]

    def _league_events(self):
        events = []
        for row in self.games.itertuples():
            if np.isnan(row.home_score):
                continue
            for opp in (row.home, row.away):
                value = self._allowed(row.season, row.gamecode, opp)
                if value is not None:
                    events.append((row.st, int(row.season), value))
        return events

    def league_mean(self, season: int, time: pd.Timestamp) -> float:
        in_season = [v for st, s, v in self.league_events if s == season and st < time]
        earlier = sorted({s for _, s, _ in self.league_events if s < season})
        fallback = (
            _mean_or_nan(v for _, s, v in self.league_events if s == earlier[-1])
            if earlier
            else NaN
        )
        if len(in_season) >= 20 or np.isnan(fallback):
            return _mean_or_nan(in_season)
        return fallback

    def team_block(self, team: str, season: int, date: pd.Timestamp, time: pd.Timestamp, with_def):
        games = self.team_games(team)
        before = games[games["st"] < time]
        out = {
            "rest_days": NaN,
            "short_rest": NaN,
            "games_last7": float(((date - before["game_date"]).dt.days < 7).sum()),
            "games_season": float((before["season"] == season).sum()),
        }
        if len(before):
            rest = (date - before["game_date"].iloc[-1]).days
            out["rest_days"] = float(min(rest, 14))
            out["short_rest"] = float(rest <= 2)
        played = before[before["team_score"].notna() & before["opp_score"].notna()]
        win = (played["team_score"] > played["opp_score"]).astype(float).to_numpy()
        margin = (played["team_score"] - played["opp_score"]).to_numpy()
        same = (played["season"] == season).to_numpy()
        out["win_pct_5"] = _mean_or_nan(win[-5:])
        out["pd_5"] = _mean_or_nan(margin[-5:])
        out["win_pct_season"] = _mean_or_nan(win[same])
        out["pd_season"] = _mean_or_nan(margin[same])
        if with_def:
            allowed = []
            for g in before.itertuples():
                value = self._allowed(g.season, g.gamecode, g.opp)
                if value is not None:
                    allowed.append(value)
            league = self.league_mean(season, time)
            for window in (5, 10):
                raw = _mean_or_nan(allowed[-window:])
                out[f"def_pir_{window}"] = raw / league if league > 0 else NaN
        return out

    # ----- παίκτες -----
    def features(self, row: dict) -> dict:
        """Features για μια γραμμή `row` με κλειδιά: player_id, season, gamecode, team_code,
        opp_code (ή None), home (ή NaN), game_date, tipoff_utc."""
        player = row["player_id"]
        season = int(row["season"])
        date = pd.Timestamp(row["game_date"])
        time = sort_time(row["tipoff_utc"], row["game_date"])
        mine = self.history[(self.history["player_id"] == player) & (self.history["st"] < time)]
        mine = mine.sort_values(["st", "season", "gamecode"])
        apps = mine[(~mine["dnp"]) & (mine["minutes"] > 0)]
        out: dict[str, float] = {}

        fantasy = apps["fantasy_score"].to_numpy()
        pir = apps["pir"].to_numpy()
        minutes = apps["minutes"].to_numpy()
        starter = apps["is_starter"].astype(float).to_numpy()
        for n in (3, 5, 10, 20):
            out[f"fantasy_mean_{n}"] = _mean_or_nan(fantasy[-n:])
            out[f"pir_mean_{n}"] = _mean_or_nan(pir[-n:])
        for name, values in (("fantasy_ewm", fantasy), ("pir_ewm", pir)):
            recent = values[-12:][::-1]
            weights = 0.5 ** (np.arange(len(recent)) / 3.0)
            out[name] = float((weights * recent).sum() / weights.sum()) if len(recent) else NaN
        out["pir_std_5"] = float(np.std(pir[-5:], ddof=1)) if len(pir) >= 2 else NaN
        last_minutes = minutes[-5:]
        out["pir_per_min_5"] = (
            float(pir[-5:].sum() / last_minutes.sum()) if last_minutes.sum() >= 10.0 else NaN
        )
        out["min_mean_3"] = _mean_or_nan(minutes[-3:])
        out["min_mean_5"] = _mean_or_nan(minutes[-5:])
        out["min_last"] = float(minutes[-1]) if len(minutes) else NaN
        out["min_trend_5"] = (
            float(np.polyfit(np.arange(len(last_minutes)), last_minutes, 1)[0])
            if len(last_minutes) >= 3
            else NaN
        )
        out["starter_rate_5"] = _mean_or_nan(starter[-5:])

        same = (apps["season"] == season).to_numpy()
        out["fantasy_season_mean"] = _mean_or_nan(fantasy[same])
        out["pir_season_mean"] = _mean_or_nan(pir[same])
        out["min_season_mean"] = _mean_or_nan(minutes[same])
        out["games_played_season"] = float(same.sum())
        out["games_played_total"] = float(len(apps))
        out["days_since_last_appearance"] = (
            float(min((date - apps["game_date"].iloc[-1]).days, 365)) if len(apps) else NaN
        )
        streak = 0
        for record in mine.iloc[::-1].itertuples():
            if _is_appearance(record):
                break
            streak += 1
        out["dnp_streak"] = float(streak)

        has_game = row.get("opp_code") is not None and not pd.isna(row.get("home"))
        home = float(row["home"]) if pd.notna(row.get("home")) else NaN
        out["home"] = home
        for name in ("rest_days", "short_rest", "games_last7", "games_season"):
            out[f"team_{name}"] = NaN
        for name in (
            "rest_days",
            "games_last7",
            "win_pct_5",
            "pd_5",
            "win_pct_season",
            "pd_season",
        ):
            out[f"opp_{name}"] = NaN
        for name in ("win_pct_5", "pd_5", "win_pct_season", "pd_season"):
            out[f"team_{name}"] = NaN
        out["opp_def_pir_5"] = NaN
        out["opp_def_pir_10"] = NaN
        out["missed_last10"] = NaN
        if has_game:
            team = row["team_code"]
            own = self.team_block(team, season, date, time, with_def=False)
            for name, value in own.items():
                out[f"team_{name}"] = value
            opp = self.team_block(row["opp_code"], season, date, time, with_def=True)
            for name, value in opp.items():
                if name in ("short_rest", "games_season"):
                    continue
                out[f"opp_{name}"] = value
            out["missed_last10"] = self._missed(player, team, season, row["gamecode"], time)
        return out

    def _missed(self, player: str, team: str, season: int, gamecode: int, time) -> float:
        games = self.team_games(team)
        keys = list(zip(games["season"], games["gamecode"], strict=True))
        position = keys.index((season, gamecode))
        rows = self.history[
            (self.history["player_id"] == player) & (self.history["team_code"] == team)
        ]
        positions = [keys.index((int(r.season), int(r.gamecode))) for r in rows.itertuples()]
        positions.append(position)
        first = min(positions)
        missed = 0
        for index in range(max(position - 10, first), position):
            frame = self.by_game.get(keys[index])
            if frame is None or not (frame["team_code"] == team).any():
                continue  # η ομάδα δεν έχει καμία γραμμή στον αγώνα: η συμμετοχή δεν είναι γνωστή
            mask = (frame["player_id"] == player) & (~frame["dnp"]) & (frame["minutes"] > 0)
            missed += 0 if bool(mask.any()) else 1
        return float(missed)
