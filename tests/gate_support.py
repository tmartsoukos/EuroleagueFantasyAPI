"""Κοινός έλεγχος του quality gate: το committed μοντέλο πρέπει να είναι out-of-sample στη σεζόν του
held-out fixture. Ζει εδώ (και όχι μέσα στο tests/quality) ώστε να δοκιμάζεται και ως συνάρτηση.
"""

from __future__ import annotations

REFIT_MESSAGE = (
    "the committed artifact was refit through the quality-gate season (train "
    "--final-refit-through): its MAE on the held-out fixture is in-sample and the gate cannot "
    "be evaluated. Commit only the artifact of a plain `python -m elfantasy.model.train`, or "
    "regenerate the fixture for a season the artifact has not seen."
)


def check_out_of_sample(metrics: dict, fixture_season: int) -> None:
    """Σηκώνει `AssertionError` αν το artifact έχει δει τη σεζόν του fixture (π.χ. με
    `--final-refit-through`), γιατί τότε το MAE του gate είναι in-sample και δεν λέει τίποτα."""
    protocol = metrics["protocol"]
    assert protocol["test"]["season"] == fixture_season, (
        f"the metrics were computed on season {protocol['test']['season']}, "
        f"the fixture is season {fixture_season}"
    )
    assert protocol["test_mae_is_out_of_sample"] is True, REFIT_MESSAGE
    assert max(protocol["final_fit"]["seasons"]) < fixture_season, REFIT_MESSAGE
