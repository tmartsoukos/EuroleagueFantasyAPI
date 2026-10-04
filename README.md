# Euroleague Fantasy Points Predictor API

[![CI](https://github.com/tmartsoukos/EuroleagueFantasyAPI/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tmartsoukos/EuroleagueFantasyAPI/actions/workflows/ci.yml)
[![coverage](https://raw.githubusercontent.com/tmartsoukos/EuroleagueFantasyAPI/badges/coverage.svg)](https://github.com/tmartsoukos/EuroleagueFantasyAPI/actions/workflows/ci.yml)

Υπηρεσία FastAPI που προβλέπει το **fantasy score** ενός παίκτη Euroleague για τον **επόμενο αγώνα** της ομάδας του. Χρησιμοποιεί ιστορικά boxscores των σεζόν 2016 έως 2026 (πακέτο `euroleague-api`), ένα μοντέλο XGBoost που ελέγχεται από quality gate (MAE σε σεζόν που το μοντέλο δεν έχει δει), βάση SQLite τοπικά και Supabase Postgres στην παραγωγή. Το CI (lint, tests, quality gate) μπορεί να το δημοσιεύει αυτόματα στο Render μετά από κάθε επιτυχημένο build στο `main`, αφού στηθεί η υπηρεσία (βλ. [`docs/DEPLOY.md`](docs/DEPLOY.md)).

> Ανεπίσημο project. Δεν σχετίζεται με τη EuroLeague. Τα δεδομένα προέρχονται από δημόσια endpoints που χρησιμοποιεί το πακέτο `euroleague-api`· οι όροι χρήσης τους δεν έχουν ελεγχθεί (βλ. [Περιορισμοί](#περιορισμοί)).

## Περιεχόμενα

1. [Τι προβλέπει](#τι-προβλέπει)
2. [Αρχιτεκτονική](#αρχιτεκτονική)
3. [Πόσο καλό είναι το μοντέλο](#πόσο-καλό-είναι-το-μοντέλο)
4. [Γρήγορη εκκίνηση (setup)](#γρήγορη-εκκίνηση-setup)
5. [Χρήση του API](#χρήση-του-api)
6. [Ρυθμίσεις](#ρυθμίσεις)
7. [Δεδομένα και ingestion](#δεδομένα-και-ingestion)
8. [Εκπαίδευση του μοντέλου](#εκπαίδευση-του-μοντέλου)
9. [Βάση δεδομένων: SQLite και Supabase](#βάση-δεδομένων-sqlite-και-supabase)
10. [Tests και ποιότητα](#tests-και-ποιότητα)
11. [CI/CD και deploy στο Render](#cicd-και-deploy-στο-render)
12. [Δομή του repo](#δομή-του-repo)
13. [Περιορισμοί](#περιορισμοί)
14. [Τεκμηρίωση](#τεκμηρίωση)

## Τι προβλέπει

Το fantasy score ενός παίκτη σε έναν αγώνα είναι το **PIR** (Performance Index Rating) και, αν η ομάδα του κερδίσει, προστίθεται **μπόνους 10% της απόλυτης τιμής του PIR**:

```
PIR = πόντοι + ριμπάουντ + ασίστ + κλεψίματα + κοψίματα + φάουλ που δέχεται
      − αστοχημένα σουτ εντός πεδιάς − αστοχημένες βολές − λάθη
      − μπλοκ που δέχεται − φάουλ που κάνει
fantasy = PIR + 0,1 × |PIR|   αν η ομάδα κερδίσει, αλλιώς PIR
```

Δηλαδή σε νίκη θετικό PIR γίνεται ×1,1 και αρνητικό PIR γίνεται ×0,9 (−4 → −3,6): το μπόνους προστίθεται πάντα. Ο κανόνας αυτός προκύπτει **εμπειρικά** από τα επίσημα σύνολα βαθμών των νικητών κάθε αγωνιστικής του Fantasy Challenge (δεν αναφέρεται σε επίσημο κείμενο). Ο ακριβής ορισμός, η αντιστοίχιση με τις στήλες του boxscore, τα παραδείγματα και ό,τι **δεν** έχει επιβεβαιωθεί βρίσκονται στο [`FANTASY_RULES.md`](FANTASY_RULES.md). Το API επιστρέφει την «ακατέργαστη» πρόβλεψη: δεν περιλαμβάνει captain ×2 ή πάγκο ×0,5, που είναι επιλογές ρόστερ.

Το `predicted_fantasy` είναι μια **τυπική** τιμή (προσεγγίζει τη διάμεσο, όχι τον μέσο όρο) και ισχύει **υπό την προϋπόθεση ότι ο παίκτης αγωνίζεται**. Το μοντέλο δεν προβλέπει τραυματισμούς· γι' αυτό υπάρχει χειροκίνητο override διαθεσιμότητας (`out`, `doubtful`) που εφαρμόζεται μετά την πρόβλεψη.

## Αρχιτεκτονική

```
euroleague_api ──► ingest (fetch με cache, clean, scoring) ──► SQLite (τοπικά) / Supabase Postgres (παραγωγή)
                                                                       │
                                  features (42, χωρίς διαρροή) ◄───────┘
                                          │
                                  XGBoost (models/model.joblib) ──► Predictor ──► FastAPI ──► Render
                                                                                       ▲
                                                          χειροκίνητο override διαθεσιμότητας (X-API-Key)
```

| Στρώμα | Πακέτο | Τι κάνει |
|---|---|---|
| Δεδομένα | `elfantasy.ingest` | Λήψη boxscores με περιορισμό ρυθμού και cache, καθαρισμός, υπολογισμός `pir` και `fantasy_score` (`elfantasy.scoring`) |
| Βάση | `elfantasy.db` | Σχήμα (SQLAlchemy), migrations Postgres, μεταφορά SQLite → Postgres |
| Features και μοντέλο | `elfantasy.features`, `elfantasy.model` | 42 features χωρίς διαρροή, backtest, quality gate, `Predictor` |
| API | `elfantasy.api` | `/health`, `/predict`, `/rankings`, `/players`, `/availability`, `/admin/refresh` |

## Πόσο καλό είναι το μοντέλο

Honest backtest: εκπαίδευση στις σεζόν 2016–2024, αξιολόγηση στη σεζόν **2025** που το committed μοντέλο δεν έχει δει ποτέ (MAE σε μονάδες fantasy score):

| Μοντέλο | MAE (test 2025) |
|---|---|
| Καθολικός μέσος | 7,038 |
| Μέσος όρος σεζόν του παίκτη (καλύτερο naive baseline) | 6,124 |
| Ridge | 5,946 |
| **XGBoost (committed μοντέλο)** | **5,886** |

Το μοντέλο είναι καλύτερο από το καλύτερο naive baseline κατά 0,24 πόντους (3,9%, 95% CI από −0,29 έως −0,19) και σε κάθε τμήμα των δεδομένων. Η βελτίωση είναι σαφής αλλά μέτρια: η πρόβλεψη ενός μεμονωμένου αγώνα έχει μεγάλο εγγενή θόρυβο (R² 0,24)· η αξία της είναι κυρίως στη **σύγκριση παικτών** (ranking). Το quality gate του CI απαιτεί MAE < 6,00 και αποτυγχάνει αλλιώς. Λεπτομέρειες: [`docs/MODEL.md`](docs/MODEL.md).

## Γρήγορη εκκίνηση (setup)

**Προαπαιτούμενα:** Python **3.12 ή νεότερη** (οι εκδόσεις numpy/scipy/xgboost του project το απαιτούν), git. Δεν χρειάζεται κανένα κλειδί API.

```bash
git clone https://github.com/tmartsoukos/EuroleagueFantasyAPI.git
cd EuroleagueFantasyAPI

python -m venv .venv
source .venv/bin/activate            # Windows (PowerShell): .venv\Scripts\Activate.ps1
                                     # Windows (Git Bash):   source .venv/Scripts/activate

pip install -r requirements-dev.txt -c constraints.txt   # ακριβείς εκδόσεις, ίδιες με του CI και του Render
pip install --no-deps -e .                               # το πακέτο elfantasy σε editable μορφή
```

> **Windows:** όρισε `PYTHONUTF8=1` και `PYTHONIOENCODING=utf-8` (π.χ. `$env:PYTHONUTF8 = "1"` στο PowerShell) πριν τρέξεις εργαλεία που τυπώνουν ονόματα με τόνους. Αν το `pip install` αποτύχει με `OSError: No such file` στο `pixeltable_pgserver`, η διαδρομή του venv είναι πολύ μακριά (όριο 260 χαρακτήρων των Windows)· δημιούργησε το venv σε πιο κοντό φάκελο.

**Έλεγχος ότι όλα δουλεύουν** (offline, χωρίς δεδομένα, περίπου 4 έως 7 λεπτά):

```bash
pytest -q
ruff check . && ruff format --check .
```

**Δεδομένα.** Το repo περιέχει το εκπαιδευμένο μοντέλο (`models/model.joblib`) αλλά **όχι** τη βάση δεδομένων (είναι στο `.gitignore`). Για να τρέξει το API με πραγματικά δεδομένα φτιάξε την τοπική βάση SQLite (η πρώτη φορά παίρνει περίπου **1,5 ώρα** λόγω του ορίου ρυθμού του API· μετά διαβάζει από cache σε μερικά δευτερόλεπτα):

```bash
python -m elfantasy.ingest.pipeline --seasons 2016-2026
```

**Εκκίνηση του API:**

```bash
uvicorn elfantasy.api.main:app --reload
```

Άνοιξε το Swagger στο <http://127.0.0.1:8000/docs>. Χωρίς βάση το API ξεκινά σε **degraded** κατάσταση (`GET /health` απαντά 503 με τον λόγο· ποτέ stack trace).

## Χρήση του API

| Endpoint | Περιγραφή |
|---|---|
| `GET /health` | Κατάσταση βάσης και μοντέλου, έκδοση μοντέλου, ημερομηνία δεδομένων, commit (503 αν κάτι δεν δουλεύει) |
| `GET /predict/{player_id}` | Πρόβλεψη fantasy score και PIR για τον επόμενο αγώνα του παίκτη |
| `GET /rankings` | Ranking ενεργών παικτών κατά προβλεπόμενο fantasy score (`limit`, `offset`, `team`, `include_unavailable`, `active_only`) |
| `GET /players?search=` | Αναζήτηση παικτών και εύρεση του `player_id` |
| `GET`/`POST`/`DELETE /availability` | Διαθεσιμότητα παικτών (τραυματισμοί/απουσίες). Τα `POST` και `DELETE` απαιτούν `X-API-Key` |
| `POST /admin/refresh` | Ξαναδιαβάζει τη βάση μετά από νέο ingestion (απαιτεί `X-API-Key`) |

```bash
# Βρες το player_id
curl -s "http://127.0.0.1:8000/players?search=vezenkov"

# Πρόβλεψη για τον επόμενο αγώνα
curl -s http://127.0.0.1:8000/predict/P003469

# Top 3 της Olympiacos
curl -s "http://127.0.0.1:8000/rankings?team=OLY&limit=3"

# Σήμανε έναν παίκτη ως τραυματία (ορίζεις πρώτα ADMIN_API_KEY στο περιβάλλον του server)
curl -s -X POST http://127.0.0.1:8000/availability \
  -H "X-API-Key: $ADMIN_API_KEY" -H "Content-Type: application/json" \
  -d '{"player_id": "P003469", "status": "out", "note": "τραυματισμός"}'
```

Παράδειγμα απάντησης του `/predict/P003469` (συντομευμένο):

```json
{
  "player_id": "P003469",
  "name": "VEZENKOV, SASHA",
  "team_code": "OLY",
  "next_game": {"game_date": "2026-10-09", "opponent_code": "IST", "home": true},
  "predicted_fantasy": 20.13,
  "model_predicted_fantasy": 20.13,
  "predicted_pir": 18.64,
  "availability": {"status": "available"},
  "notes": []
}
```

Με `status: "out"` το `predicted_fantasy` γίνεται `0.0` (το `model_predicted_fantasy` μένει όπως το έβγαλε το μοντέλο) και ο παίκτης εξαιρείται από το `/rankings`. Πλήρης αναφορά (παράμετροι, κωδικοί σφάλματος, ασφάλεια): [`docs/API.md`](docs/API.md).

## Ρυθμίσεις

Οι ρυθμίσεις διαβάζονται από μεταβλητές περιβάλλοντος και από αρχείο `.env` στη ρίζα του repo (το `.env` είναι στο `.gitignore`· πρότυπο: [`.env.example`](.env.example)). **Ποτέ μυστικά σε αρχείο που γίνεται commit.**

| Μεταβλητή | Προεπιλογή | Σημασία |
|---|---|---|
| `DATABASE_URL` | `sqlite:///data/elfantasy.db` | Βάση δεδομένων. Σε παραγωγή: Supabase Postgres (`postgresql://…`, session pooler) |
| `MODEL_PATH` | `models/model.joblib` | Αρχείο μοντέλου |
| `MAE_THRESHOLD` | `6.00` | Όριο MAE του quality gate |
| `ADMIN_API_KEY` | (κενό) | Κλειδί για τα προστατευμένα endpoints. **Κενό = τα endpoints είναι κλειστά (503)**, ποτέ ανοιχτά |
| `DATA_DIR` | `data` | Φάκελος για cache, αναφορές και logs του ingestion |

## Δεδομένα και ingestion

```bash
# Πλήρης λήψη και φόρτωση (cache στο data/raw, ρυθμός 1 αίτημα/δευτερόλεπτο)
python -m elfantasy.ingest.pipeline --seasons 2016-2026

# Μόνο η τρέχουσα σεζόν: νέοι αγώνες και ανανεωμένο πρόγραμμα
python -m elfantasy.ingest.pipeline --update

# Χωρίς δίκτυο: επεξεργασία μόνο από το cache
python -m elfantasy.ingest.pipeline --seasons 2016-2026 --no-fetch

# Έλεγχοι ποιότητας πάνω στη βάση (pir == valuation, πλήθη, αγώνες που λείπουν)
python -m elfantasy.ingest.verify --accept-missing 2018/21
```

Το pipeline δεν χάνει σιωπηλά αγώνες (το πακέτο `euroleague-api` το κάνει), σέβεται το όριο ρυθμού του Cloudflare (HTTP 429), και υπολογίζει το `pir` από τα στατιστικά, το οποίο επαληθεύεται ότι ταυτίζεται με τη στήλη `Valuation` του API σε **όλες** τις γραμμές. Ένας παιγμένος αγώνας (2018/21) λείπει οριστικά από την πηγή (κενό boxscore). Λεπτομέρειες: [`docs/INGESTION.md`](docs/INGESTION.md), [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## Εκπαίδευση του μοντέλου

```bash
python -m elfantasy.model.train              # πλήρες backtest και επιλογή (~75 s)
python -m elfantasy.model.train --no-tune    # γρήγορο (~10 s)
```

Το script γράφει `models/model.joblib` και `models/metrics.json` **μόνο αν** το honest test MAE είναι κάτω από το threshold (exit code 3 αλλιώς, χωρίς να αλλάξει κανένα υπάρχον αρχείο). Το committed μοντέλο εκπαιδεύτηκε στις σεζόν έως 2024 και αξιολογήθηκε στο 2025, ώστε το quality gate να τρέχει σε δεδομένα που δεν έχει δει. Για να ανανεώσεις την πρόβλεψη με νεότερα δεδομένα δεν χρειάζεται επανεκπαίδευση: τα features διαβάζουν πάντα όλο το ιστορικό της βάσης. Λεπτομέρειες: [`docs/MODEL.md`](docs/MODEL.md).

Καταγραφή και αξιολόγηση προβλέψεων στον πίνακα `predictions`:

```bash
python -m elfantasy.model.record_predictions   # γράφει τις προβλέψεις της ημέρας (idempotent)
python -m elfantasy.model.evaluate_recorded    # MAE όταν παιχτούν οι αγώνες
```

## Βάση δεδομένων: SQLite και Supabase

Τοπικά και στα tests η βάση είναι SQLite. Στην παραγωγή είναι Supabase Postgres, με Row Level Security σε όλους τους πίνακες (deny-all για το REST API του Supabase· η εφαρμογή συνδέεται απευθείας ως `postgres`). Το σχήμα του Postgres δημιουργείται **μόνο** από τα migrations:

```bash
export DATABASE_URL='postgresql://postgres.PROJECT_REF:PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres?sslmode=require'

python -m elfantasy.db.migrate --status     # τι έχει εφαρμοστεί
python -m elfantasy.db.migrate              # εφαρμογή των migrations
python -m elfantasy.db.transfer --from sqlite:///data/elfantasy.db   # μεταφορά + επαλήθευση (πλήθη, aggregates, ψηφιακό αποτύπωμα)
```

Το URL το δίνει το Supabase (*Project → Connect → Session pooler*). Χρησιμοποίησε το **session pooler** (IPv4) και όχι το direct connection (μόνο IPv6 στο free plan). Ο κωδικός με ειδικούς χαρακτήρες (`@ : / ? # %`) θέλει URL-encoding. Λεπτομέρειες, RLS και περιορισμοί του free tier: [`docs/DATABASE.md`](docs/DATABASE.md).

## Tests και ποιότητα

```bash
pytest -q                                   # όλα (~2.000 tests, coverage ~99%)
pytest tests/quality -q                     # μόνο το quality gate του μοντέλου
pytest --cov=elfantasy --cov-report=term-missing
ruff check . && ruff format --check .
```

- Τα tests τρέχουν **offline** και σε προσωρινή απομονωμένη βάση: δεν αγγίζουν ποτέ το `data/` ή την παραγωγική βάση.
- Τα tests με marker `postgres` (migrations, μεταφορά, API σε πραγματικό Postgres) χρησιμοποιούν έναν **τοπικό** ενσωματωμένο Postgres (`pixeltable-pgserver`) ή το `TEST_DATABASE_URL`, μόνο αν δείχνει σε localhost. Με `ELFANTASY_REQUIRE_POSTGRES=1` αποτυγχάνουν αντί να παραλείπονται (έτσι τρέχει το CI).
- Το **quality gate** φορτώνει το committed μοντέλο και ένα held-out fixture της σεζόν 2025 και αποτυγχάνει αν το MAE ξεπεράσει το `MAE_THRESHOLD`.

## CI/CD και deploy στο Render

Σε κάθε push και pull request τρέχει το GitHub Actions workflow [`ci.yml`](.github/workflows/ci.yml):

```
push / PR ─► lint ───────┐
          ─► test ───────┼─► deploy (μόνο push στο main) ─► smoke check του /health
          ─► quality-gate┘
          test ─► coverage-badge (μόνο main)
```

- **lint:** `ruff check` και `ruff format --check`.
- **test:** όλα τα tests σε Python 3.12 με πραγματικό Postgres (service container) και όριο coverage 95%.
- **quality-gate:** αποτυγχάνει αν το MAE του committed μοντέλου ξεπεράσει το threshold.
- **deploy:** μόνο αν περάσουν και τα τρία, καλεί το Deploy Hook του Render (secret `RENDER_DEPLOY_HOOK_URL`), και μετά ελέγχει ότι το `/health` είναι `ok` και δηλώνει το νέο commit. Αν το secret λείπει, το job παραλείπεται χωρίς σφάλμα.

Οι εκδόσεις των πακέτων είναι καρφωμένες στο [`constraints.txt`](constraints.txt) (ίδιες με της εκπαίδευσης του μοντέλου). Το Render στήνεται από το [`render.yaml`](render.yaml) (Blueprint, free plan, Φρανκφούρτη, **auto-deploy κλειστό**). Βήμα-βήμα οδηγίες για τη δημιουργία της υπηρεσίας, τα `DATABASE_URL` και `ADMIN_API_KEY`, το Deploy Hook, το rollback και τη διαδικασία επανεκπαίδευσης: [`docs/DEPLOY.md`](docs/DEPLOY.md). Το free plan «κοιμάται» μετά από περίπου 15 λεπτά αδράνειας και το πρώτο αίτημα μετά από αυτό παίρνει περίπου ένα λεπτό ή περισσότερο.

Ανανέωση δεδομένων στην παραγωγή:

```bash
DATABASE_URL='postgresql://…' python -m elfantasy.ingest.pipeline --update    # νέοι αγώνες στο Supabase
curl -s -X POST -H "X-API-Key: $ADMIN_API_KEY" https://<όνομα>.onrender.com/admin/refresh
```

## Δομή του repo

```
src/elfantasy/
  scoring.py        PIR και fantasy score (καθαρές συναρτήσεις)
  config.py         ρυθμίσεις (περιβάλλον και .env)
  ingest/           fetch με cache και rate limit, clean, pipeline, verify
  db/               σχήμα, migrations (SQL), migrate, transfer, engine
  features/         42 features χωρίς διαρροή (μία διαδρομή για training και πρόβλεψη)
  model/            train, artifact, Predictor, holdout, καταγραφή και αξιολόγηση προβλέψεων
  api/              FastAPI: routers, schemas, διαθεσιμότητα, health
models/             model.joblib και metrics.json (committed, ~250 KB)
tests/              unit, integration, quality, fixtures
scripts/            εργαλεία του CI (badge, smoke check, quality summary)
docs/               αναλυτική τεκμηρίωση
.github/workflows/  ci.yml
render.yaml         Render Blueprint
constraints.txt     καρφωμένες εκδόσεις πακέτων
```

## Περιορισμοί

- **Οι κανόνες fantasy επιβεβαιώνονται εν μέρει από επίσημη πηγή.** Το ότι οι πόντοι βασίζονται στο PIR, το captain ×2, ο πάγκος ×0,5 και το DNP = 0 αναφέρονται στις επίσημες σελίδες της EuroLeague. Το μπόνους νίκης και ο χειρισμός του αρνητικού PIR **δεν** αναφέρονται σε επίσημο κείμενο· προκύπτουν από τρίτες πηγές και από τα επίσημα σύνολα βαθμών 29 αγωνιστικών των σεζόν 2024-25 και 2025-26. Για παλαιότερες σεζόν δεν βρέθηκε πηγή, και ο σημερινός τύπος εφαρμόζεται αναδρομικά στα ιστορικά δεδομένα (βλ. [`FANTASY_RULES.md`](FANTASY_RULES.md)).
- **Δεν προβλέπονται τραυματισμοί, DNP ή αλλαγές ρόστερ.** Η διαθεσιμότητα είναι χειροκίνητη (override). Ένας παίκτης που άλλαξε ομάδα το καλοκαίρι θεωρείται στην παλιά ομάδα μέχρι να εμφανιστεί σε αγώνα της νέας.
- **Ο θόρυβος ενός αγώνα είναι μεγάλος** (R² 0,24)· η πρόβλεψη είναι χρήσιμη για σύγκριση παικτών, όχι για ακριβή αριθμό. Το μοντέλο έχει αρνητικό bias περίπου −0,8 (προσεγγίζει τη διάμεσο).
- **Το MAE του 2025 δεν είναι εγγυημένο για τη σεζόν 2026-27.**
- **Το free tier του Render και του Supabase έχουν περιορισμούς:** αδράνεια και cold start στο Render, παύση του project του Supabase μετά από μία εβδομάδα αδράνειας, 512 MB μνήμη.
- **Δεδομένα:** το API της EuroLeague δεν έχει επίσημους όρους χρήσης που να έχουν ελεγχθεί εδώ, και το πακέτο `euroleague-api` είναι GPLv3 (χρησιμοποιείται ως εξάρτηση, δεν ενσωματώνεται). Το repo δεν περιέχει αρχείο άδειας (LICENSE).
- Το API δεν έχει rate limiting ούτε CORS· αν χρειαστεί κλήση από browser σε άλλο domain πρέπει να προστεθεί CORS.

## Τεκμηρίωση

> Τα έγγραφα του `docs/` δείχνουν τις εντολές με διαδρομές Windows (`.venv/Scripts/python.exe`). Σε Linux ή macOS χρησιμοποίησε `python` μέσα στο ενεργοποιημένο venv (ή `.venv/bin/python`).

| Αρχείο | Περιεχόμενο |
|---|---|
| [`FANTASY_RULES.md`](FANTASY_RULES.md) | Ο τύπος του fantasy score, πηγές, παραδείγματα, ό,τι δεν επιβεβαιώθηκε |
| [`docs/API.md`](docs/API.md) | Endpoints, ρυθμίσεις, διαθεσιμότητα παικτών, ασφάλεια |
| [`docs/MODEL.md`](docs/MODEL.md) | Features, backtest, αποτελέσματα, quality gate, `Predictor` |
| [`docs/INGESTION.md`](docs/INGESTION.md) | Λήψη, καθαρισμός και αποθήκευση δεδομένων, αποτελέσματα του πλήρους τρεξίματος |
| [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) | Το πακέτο `euroleague-api`: endpoints, στήλες, παγίδες, όριο ρυθμού |
| [`docs/DATABASE.md`](docs/DATABASE.md) | Σχήμα, migrations, μεταφορά στο Supabase, RLS |
| [`docs/DEPLOY.md`](docs/DEPLOY.md) | CI/CD, Render, οδηγίες βήμα-βήμα |
