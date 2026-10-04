"""Έλεγχοι υγιεινής του repo (Φάση 5): κανένα μυστικό στα αρχεία που ανεβαίνουν στο git.

Τα credentials της παραγωγικής βάσης ζουν ΜΟΝΟ σε αρχεία που αγνοεί το git (`.env`, `.env.txt`...)
και σε μεταβλητές περιβάλλοντος. Τα tests ΔΕΝ διαβάζουν ποτέ αρχεία `.env*` εκτός από το
`.env.example`.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Κωδικοί που επιτρέπονται σε URL μέσα στο repo: κράτηση θέσης (ΚΕΦΑΛΑΙΑ και κάτω παύλες), «***»
# (κρυμμένος κωδικός) και οι ψεύτικες τιμές που χρησιμοποιούν τα tests και τα docs.
PLACEHOLDER_PASSWORD = re.compile(r"^(?:[A-Z][A-Z_]*|\*{3}|p|pw|pass|password|secret|S3cr3t-Pa55)$")
URL_WITH_PASSWORD = re.compile(
    r"postgres(?:ql)?(?:\+\w+)?://([^:/@\s'\"`]+):([^@\s'\"`]+)@(\[[0-9a-fA-F:]+\]|[^:/?#\s'\"`]+)"
)
# Φάση 6: ένα URL προς τον ΤΟΠΙΚΟ server του CI (service container `postgres:17` στο ci.yml,
# `postgresql://postgres:postgres@localhost:5432/postgres`) έχει ανώδυνο κωδικό μιας χρήσεως: δεν
# είναι πραγματικό μυστικό. Οποιοσδήποτε άλλος host (π.χ. Supabase) πρέπει να έχει κράτηση θέσης.
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "[::1]"}
KEY_PATTERNS = {
    "JWT (π.χ. Supabase anon/service_role key)": re.compile(
        r"eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}"
    ),
    "Supabase secret key": re.compile(r"sb_secret_[A-Za-z0-9_-]{10,}"),
    "Supabase personal access token": re.compile(r"sbp_[0-9a-f]{20,}"),
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".sql",
    ".toml",
    ".yml",
    ".yaml",
    ".txt",
    ".cfg",
    ".ini",
    ".json",
    ".example",
}


def suspicious_database_urls(text: str) -> list[str]:
    """URL βάσης με κωδικό που δεν είναι κράτηση θέσης ούτε δείχνει σε τοπικό server."""
    found = []
    for match in URL_WITH_PASSWORD.finditer(text):
        if match.group(3).lower() in LOOPBACK_HOSTS:
            continue  # τοπικός server (π.χ. το service container του CI)
        if not PLACEHOLDER_PASSWORD.match(match.group(2)):
            found.append(match.group(0))
    return found


@pytest.mark.parametrize(
    "text",
    [
        "postgresql://postgres:postgres@localhost:5432/postgres",  # το service container του CI
        "postgresql+psycopg://user:anything@127.0.0.1/db",
        "postgres://user:anything@[::1]:5432/db",
        "postgresql://postgres.PROJECT_REF:PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres",
        "postgresql://user:***@db.abc.supabase.co/postgres",
    ],
)
def test_loopback_urls_and_placeholders_are_not_suspicious(text):
    assert suspicious_database_urls(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "postgresql://postgres.abcdefgh:Xk3p9sLmQ2@aws-0-eu-central-1.pooler.supabase.com:5432/postgres",
        "postgresql://postgres:hunter22@db.abcdefgh.supabase.co:5432/postgres",
        "postgresql://postgres:postgres@localhost.example.com:5432/postgres",  # δεν είναι loopback
        "postgresql://postgres:postgres@10.0.0.5:5432/postgres",
    ],
)
def test_remote_urls_with_a_real_looking_password_are_suspicious(text):
    assert len(suspicious_database_urls(text)) == 1


def candidate_files() -> list[Path]:
    """Τα αρχεία που είναι ή θα γίνουν tracked από το git (όχι τα ignored), ΧΩΡΙΣ κανένα `.env*`
    εκτός από το `.env.example`."""
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("git is not available")
    listing = subprocess.run(
        [git, "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout.splitlines()
    files = []
    for name in listing:
        path = ROOT / name
        if path.name.startswith(".env") and path.name != ".env.example":
            continue  # ποτέ δεν ανοίγουμε αρχεία μυστικών
        if path.suffix in TEXT_SUFFIXES or path.name == ".env.example":
            if path.is_file():
                files.append(path)
    return files


def test_env_example_contains_only_placeholders():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for match in URL_WITH_PASSWORD.finditer(text):
        assert PLACEHOLDER_PASSWORD.match(match.group(2)), "το .env.example έχει πραγματικό κωδικό"
    active = [line for line in text.splitlines() if line.startswith("DATABASE_URL=")]
    assert active == [
        "DATABASE_URL=sqlite:///data/elfantasy.db"
    ]  # η προεπιλογή είναι τοπική SQLite
    assert "postgresql://" in text  # το παράδειγμα για το Supabase υπάρχει (σχολιασμένο)
    assert "TEST_DATABASE_URL" in text


def test_the_gitignore_keeps_every_env_file_out_except_the_example():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in lines and ".env.*" in lines and "!.env.example" in lines
    assert lines.index("!.env.example") > lines.index(".env.*")  # η εξαίρεση ακολουθεί τον κανόνα


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
@pytest.mark.parametrize(
    ("name", "ignored"),
    [
        (".env", True),
        (".env.txt", True),
        (".env.local", True),
        (".env.production", True),
        (".env.example", False),
    ],
)
def test_git_ignores_the_secret_files_and_not_the_example(name, ignored):
    # το `git check-ignore` δουλεύει με τα πρότυπα του .gitignore και δεν ανοίγει το αρχείο
    result = subprocess.run(
        [shutil.which("git"), "check-ignore", "-q", name], cwd=ROOT, capture_output=True
    )
    assert (result.returncode == 0) is ignored


def test_no_file_in_the_repository_contains_a_real_database_password_or_key():
    problems = []
    skip_dirs = {"tests", "data", "models", ".venv"}
    for path in candidate_files():
        relative = path.relative_to(ROOT)
        if relative.parts[0] in skip_dirs:
            continue  # τα tests χρησιμοποιούν ψεύτικους κωδικούς· ελέγχονται από τα ίδια τα tests
        text = path.read_text(encoding="utf-8", errors="replace")
        if suspicious_database_urls(text):
            problems.append(f"{relative}: URL βάσης με κωδικό που δεν είναι κράτηση θέσης")
        for label, pattern in KEY_PATTERNS.items():
            if pattern.search(text):
                problems.append(f"{relative}: μοιάζει με {label}")
    assert not problems, "\n".join(problems)


def test_the_scan_never_opens_secret_files():
    names = {path.name for path in candidate_files()}
    assert not any(name.startswith(".env") and name != ".env.example" for name in names)
