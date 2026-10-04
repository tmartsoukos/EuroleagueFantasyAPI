# Euroleague Fantasy Points Predictor API

[![CI](https://github.com/tmartsoukos/EuroleagueFantasyAPI/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tmartsoukos/EuroleagueFantasyAPI/actions/workflows/ci.yml)
[![coverage](https://raw.githubusercontent.com/tmartsoukos/EuroleagueFantasyAPI/badges/coverage.svg)](https://github.com/tmartsoukos/EuroleagueFantasyAPI/actions/workflows/ci.yml)

Υπηρεσία FastAPI που προβλέπει το fantasy score ενός παίκτη Euroleague για τον επόμενο αγώνα του. Χρησιμοποιεί ιστορικά boxscores των σεζόν 2016 έως 2026 (πακέτο `euroleague-api`), ένα μοντέλο XGBoost που ελέγχεται από quality gate (MAE σε σεζόν που το μοντέλο δεν έχει δει), βάση SQLite τοπικά και Supabase Postgres στην παραγωγή, και δημοσιεύεται αυτόματα στο Render μετά από κάθε επιτυχημένο build στο `main`.

> **Το πλήρες README (εγκατάσταση, εκτέλεση, tests, deploy) ολοκληρώνεται στη Φάση 7.**

## Τεκμηρίωση

| Αρχείο | Περιεχόμενο |
|---|---|
| [`docs/API.md`](docs/API.md) | Endpoints, ρυθμίσεις, διαθεσιμότητα παικτών, ασφάλεια |
| [`docs/MODEL.md`](docs/MODEL.md) | Features, backtest, αποτελέσματα, quality gate |
| [`docs/DATABASE.md`](docs/DATABASE.md) | Σχήμα, migrations, Supabase |
| [`docs/INGESTION.md`](docs/INGESTION.md), [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) | Λήψη και καθαρισμός δεδομένων |
| [`docs/DEPLOY.md`](docs/DEPLOY.md) | CI/CD, deploy στο Render, οδηγίες βήμα-βήμα |
| [`FANTASY_RULES.md`](FANTASY_RULES.md) | Ο τύπος του fantasy score |
