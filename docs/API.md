# API πρόβλεψης fantasy score (Φάση 4)

Το έγγραφο περιγράφει την υπηρεσία FastAPI που εξυπηρετεί τις προβλέψεις του μοντέλου της Φάσης 3 (`docs/MODEL.md`): τα endpoints, την ερμηνεία των τιμών, τη διαθεσιμότητα παικτών (τραυματισμοί), την ασφάλεια, τις ρυθμίσεις, την εκτέλεση τοπικά και στο Render, την ανανέωση δεδομένων, τα tests και τους περιορισμούς. Τα παραδείγματα responses είναι **πραγματικά**: προέρχονται από τρέξιμο του `uvicorn` στις 2026-10-03 πάνω στην πραγματική βάση `data/elfantasy.db` (δεδομένα έως 02/10/2026) και στο committed `models/model.joblib` (έκδοση `20261003T140043Z-7ae47948`).

## 1. Σύνοψη

| Endpoint | Πρόσβαση | Σκοπός |
|---|---|---|
| `GET /health` | δημόσιο | Κατάσταση βάσης και μοντέλου: 200 `ok` ή 503 `degraded` (το `HEAD /health` δίνει τον ίδιο κωδικό) |
| `GET /predict/{player_id}` | δημόσιο | Πρόβλεψη fantasy score και PIR για τον επόμενο αγώνα ενός παίκτη |
| `GET /rankings` | δημόσιο | Παίκτες ταξινομημένοι κατά προβλεπόμενο fantasy score |
| `GET /players` | δημόσιο | Αναζήτηση παικτών κατά όνομα (εύρεση του `player_id`) |
| `GET /availability` | δημόσιο | Λίστα των χειροκίνητων εγγραφών διαθεσιμότητας |
| `POST /availability` | `X-API-Key` | Καταχώρηση ή αντικατάσταση διαθεσιμότητας παίκτη (upsert) |
| `DELETE /availability/{player_id}` | `X-API-Key` | Αφαίρεση του override διαθεσιμότητας |
| `POST /admin/refresh` | `X-API-Key` | Ανανέωση δεδομένων από τη βάση μετά από νέο ingestion |
| `GET /docs`, `/redoc`, `/openapi.json` | δημόσιο | Swagger UI, ReDoc και το σχήμα OpenAPI |

Το `GET /` ανακατευθύνει (307) στο `/docs`.

### Αρχεία

| Αρχείο | Ρόλος |
|---|---|
| `src/elfantasy/api/main.py` | `create_app(settings, engine, predictor, clock)` και το `app` για το `uvicorn`, μετατροπή σφαλμάτων σε JSON |
| `src/elfantasy/api/state.py` | Φόρτωση βάσης, μοντέλου και υπηρεσίας στο startup/shutdown, degraded κατάσταση |
| `src/elfantasy/api/services.py` | `PredictionService`: προβλέψεις, rankings, αναζήτηση, ονόματα ομάδων, ανανέωση (χωρίς FastAPI) |
| `src/elfantasy/api/availability.py` | Λογική του override και αποθήκευση στον πίνακα `player_availability` (χωρίς FastAPI) |
| `src/elfantasy/api/health.py` | Σύνθεση της απάντησης του `/health` |
| `src/elfantasy/api/schemas.py` | Μοντέλα Pydantic αιτημάτων και απαντήσεων, με περιγραφές και παραδείγματα |
| `src/elfantasy/api/deps.py` | Dependencies: κατάσταση, `X-API-Key`, κανονικοποίηση `player_id` |
| `src/elfantasy/api/responses.py` | Κοινή τεκμηρίωση των απαντήσεων σφάλματος στο OpenAPI |
| `src/elfantasy/api/routers/` | Λεπτοί routers: `health`, `predict` (και `/rankings`), `players`, `availability`, `admin` |
| `src/elfantasy/db/models.py` | Νέος πίνακας `player_availability` (ενότητα 7)· σχήμα, migrations και Postgres: `docs/DATABASE.md` |

## 2. Πώς τρέχει

Οι εντολές τρέχουν από τη **ρίζα του repo**, γιατί οι προεπιλεγμένες διαδρομές είναι σχετικές (`sqlite:///data/elfantasy.db`, `models/model.joblib`).

```bash
# Τοπικά (Git Bash). Το κλειδί διαχειριστή ορίζεται μόνο στο περιβάλλον της διεργασίας.
ADMIN_API_KEY="βάλε-μια-μεγάλη-τυχαία-τιμή" \
  .venv/Scripts/python.exe -m uvicorn elfantasy.api.main:app --port 8000

# Ισοδύναμα, αν το venv είναι ενεργό:
uvicorn elfantasy.api.main:app --port 8000
```

```powershell
# PowerShell
$env:ADMIN_API_KEY = "βάλε-μια-μεγάλη-τυχαία-τιμή"
.venv\Scripts\python.exe -m uvicorn elfantasy.api.main:app --port 8000
```

Μετά το πρώτο μήνυμα `Application startup complete` η υπηρεσία είναι έτοιμη (στο laptop περίπου 5,5 έως 6,5 s από την εκκίνηση της διεργασίας: εισαγωγές βιβλιοθηκών, φόρτωση μοντέλου, ανάγνωση βάσης και υπολογισμός των πρώτων προβλέψεων). Το Swagger είναι στο <http://127.0.0.1:8000/docs>.

**Render** (το `render.yaml` γράφεται στη Φάση 6): εντολή εκκίνησης

```bash
uvicorn elfantasy.api.main:app --host 0.0.0.0 --port $PORT
```

με health check path `/health`. Το πακέτο `elfantasy` πρέπει να είναι εγκατεστημένο (`pip install .`) ή να οριστεί `PYTHONPATH=src`. Η εφαρμογή πρέπει να τρέχει με **μία** διεργασία (worker): ο `Predictor` και η cache των προβλέψεων ζουν μέσα στη διεργασία και το `/admin/refresh` ανανεώνει μόνο τη διεργασία που το δέχεται (ενότητα 11). Το `app` ΔΕΝ διαβάζει βάση ή μοντέλο στο import: αν λείπουν, η υπηρεσία ξεκινά σε degraded κατάσταση (ενότητα 3, `/health`).

## 3. Ρυθμίσεις περιβάλλοντος

Διαβάζονται από μεταβλητές περιβάλλοντος και, αν δεν υπάρχουν εκεί, από το αρχείο `.env` (`.env.example`). Η ανάγνωση γίνεται στο startup, όχι στο import.

| Μεταβλητή | Προεπιλογή | Σημασία στο API |
|---|---|---|
| `DATABASE_URL` | `sqlite:///data/elfantasy.db` | Βάση ιστορικού, προγράμματος και διαθεσιμότητας. Στην παραγωγή το Supabase Postgres μέσω του session pooler (`postgresql://postgres.<ref>:PASSWORD@aws-0-<region>.pooler.supabase.com:5432/postgres?sslmode=require`, βλ. `docs/DATABASE.md`)· τα URL `postgresql://…` μετατρέπονται σε `postgresql+psycopg://…`. Ένα αρχείο SQLite που δεν υπάρχει **δεν δημιουργείται**: η υπηρεσία γίνεται degraded. Σε Postgres το API **δεν δημιουργεί ποτέ πίνακες** (τους δημιουργούν τα migrations)· ο κωδικός δεν εμφανίζεται ποτέ σε logs ή μηνύματα |
| `MODEL_PATH` | `models/model.joblib` | Το αποθηκευμένο μοντέλο. Αν λείπει ή είναι ασύμβατο, η υπηρεσία γίνεται degraded |
| `ADMIN_API_KEY` | κενό | Κλειδί για τα προστατευμένα endpoints (ενότητα 8). **Κενό = τα endpoints είναι κλειστά** |
| `MAE_THRESHOLD` | `6.00` | Δεν χρησιμοποιείται από το API (μόνο από την εκπαίδευση και το quality gate). Το `/health` δείχνει το threshold που καταγράφηκε στο `metrics.json` |
| `DATA_DIR` | `data` | Δεν χρησιμοποιείται από το API (cache, αναφορές και logs του ingestion) |

Μία κοινή `engine` (από το `DATABASE_URL`) χρησιμοποιείται από την υπηρεσία, τη διαθεσιμότητα και τον `Predictor` (`Predictor.load(model_path, engine=engine)`).

## 4. Ερμηνεία των τιμών

Η πλήρης συζήτηση βρίσκεται στο `docs/MODEL.md` (ενότητες 9 και 14). Τα σημεία που αφορούν όσους καταναλώνουν το API:

- **Το `predicted_fantasy` είναι «τυπική» τιμή και όχι μέσος όρος.** Το μοντέλο ελαχιστοποιεί το MAE και προσεγγίζει τη **διάμεσο**: έχει αρνητικό bias περίπου −0,8 πόντων (−0,77 στο test του 2025), δηλαδή υποτιμά κατά μέσο όρο. Η υποτίμηση δεν είναι ίδια για κάθε παίκτη.
- **Υποθέτει ότι ο παίκτης αγωνίζεται.** Δεν προβλέπει τραυματισμούς, απουσίες ή αλλαγές ρόστερ· ένας παίκτης του βάθους με 50% πιθανότητα DNP έχει την ίδια πρόβλεψη με έναν βασικό με τα ίδια λεπτά. Γι' αυτό υπάρχει το χειροκίνητο override διαθεσιμότητας (ενότητα 7).
- **Δεν περιλαμβάνει captain ×2 ή πάγκο ×0,5.** Είναι επιλογές ρόστερ και εφαρμόζονται εκ των υστέρων από τον καταναλωτή (`FANTASY_RULES.md`).
- **Το `predicted_pir` είναι ανεξάρτητο μοντέλο**, όχι υπολογισμένο από το fantasy. Στους ενεργούς παίκτες η συσχέτιση είναι 0,997, αλλά στο 13% των παικτών το fantasy είναι ελαφρά μικρότερο από το PIR, παρότι για θετικό PIR ο τύπος δίνει fantasy ≥ PIR. Μην υπολογίζεις το ένα από το άλλο. Το `predicted_pir` **δεν επηρεάζεται από τη διαθεσιμότητα**: ακόμη και για παίκτη `out` είναι η ακατέργαστη τιμή του μοντέλου.
- **Μέγεθος σφάλματος.** Το honest MAE είναι 5,91 πόντοι fantasy (R² 0,24)· το μοντέλο είναι καλύτερο από το naive baseline κατά 3,9%. Η αξία της πρόβλεψης είναι στη σύγκριση παικτών και στον μέσο όρο πολλών αγώνων, όχι στον ακριβή αριθμό ενός αγώνα.
- **Ο επόμενος αγώνας** είναι ο πρώτος αγώνας της ομάδας του παίκτη με `played = false` και `game_date` από **σήμερα (UTC)** και μετά. Ο αγώνας της σημερινής ημέρας παραμένει «επόμενος» μέχρι να ενημερωθεί η βάση (νέο ingestion και `POST /admin/refresh`).
- **`is_active`.** Ο παίκτης έχει γραμμή αγώνα (και ως DNP) στη νεότερη σεζόν που έχει ξεκινήσει ή μέσα στις τελευταίες 45 ημέρες. Αν είναι `false` (π.χ. έφυγε από τη λίγκα, ή δεν έχει ακόμη παίξει στη νέα σεζόν), εξακολουθεί να υπάρχει πρόβλεψη με την ομάδα της τελευταίας γραμμής του και τον επόμενο αγώνα της, αλλά η ομάδα και ο αγώνας μπορεί να είναι παλιά. Τα `/rankings` δείχνουν από προεπιλογή μόνο ενεργούς.
- **`n_prior_appearances`.** Όσο μικρότερο, τόσο λιγότερο αξιόπιστη η πρόβλεψη (0 = μόνο το πλαίσιο του αγώνα έχει σήμα).

Οι τιμές προβλέψεων στρογγυλεύονται σε **2 δεκαδικά** στην απάντηση· τα `features` (όταν ζητούνται) σε 4.

### Σημειώσεις (`notes`) στο `/predict`

| Σημείωση | Πότε εμφανίζεται |
|---|---|
| `player is marked out: effective prediction is 0` | Διαθεσιμότητα `out` |
| `player is marked doubtful: prediction assumes the player plays` | Διαθεσιμότητα `doubtful` |
| `expected return date has passed: the availability record may be outdated` | `out` ή `doubtful` με `expected_return` πριν από σήμερα (οι εγγραφές δεν λήγουν μόνες τους) |
| `no scheduled game: neutral context` | `next_game` είναι `null` (π.χ. offseason): τα features πλαισίου λείπουν και η πρόβλεψη βασίζεται στη φόρμα του παίκτη |
| `player is not active (no games in the latest season or in the last 45 days): team and next game may be outdated` | `is_active` είναι `false` |
| `player has no previous appearances: prediction is based on the game context only` | `n_prior_appearances` είναι 0 |
| `player has only N previous appearance(s): prediction is less reliable` | `n_prior_appearances` από 1 έως 4 |

Η σειρά είναι σταθερή: πρώτα οι σημειώσεις διαθεσιμότητας, μετά του πλαισίου.

## 5. Endpoints

Σε όλα τα παραδείγματα η βάση είναι `http://127.0.0.1:8000` και τα responses είναι πραγματικά (μερικά συντομευμένα όπου σημειώνεται). Στα Windows το `curl` της PowerShell είναι ψευδώνυμο: χρησιμοποίησε `curl.exe`.

### 5.1 `GET /health`

Ελαφρύς έλεγχος (3 μικρά queries και ανάγνωση των ήδη φορτωμένων μετρικών· **δεν υπολογίζει προβλέψεις**), κατάλληλος για το Render.

```bash
curl -s http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "model": {
    "version": "20261003T140043Z-7ae47948",
    "selected_model": "xgb_pseudohuber_09",
    "test_mae": 5.908791,
    "threshold": 6.0,
    "trained_through_season": 2024
  },
  "database": {
    "ok": true,
    "players": 1212,
    "latest_played_game_date": "2026-10-02",
    "next_scheduled_game_date": "2026-10-07"
  },
  "data_age_days": 1,
  "data_loaded_through": "2026-10-02",
  "problems": []
}
```

- **HTTP 200 με `status: ok`** μόνο αν η βάση απαντά και περιέχει παίκτες **και** το μοντέλο φορτώθηκε και υπολόγισε τις πρώτες προβλέψεις. Αλλιώς **HTTP 503 με `status: degraded`**, και η λίστα `problems` λέει ποιος έλεγχος απέτυχε. Τα μηνύματα είναι σταθερά και σύντομα: δεν περιέχουν διαδρομές αρχείων, URL βάσης, credentials ή stack traces (η λεπτομέρεια γράφεται μόνο στο log του server).
- `database.latest_played_game_date` είναι το data cutoff **της βάσης** (ζωντανό query) και `data_loaded_through` το cutoff **που έχουν φορτώσει οι προβλέψεις**. Αν διαφέρουν, η βάση έχει νεότερα δεδομένα και χρειάζεται `POST /admin/refresh`.
- `data_age_days`: ημέρες από τον τελευταίο παιγμένο αγώνα μέχρι σήμερα (UTC). Δεν είναι πρόβλημα στο offseason.
- `model.test_mae` και `threshold` προέρχονται από το `metrics.json` του μοντέλου· `null` όπου λείπουν.

Πραγματική degraded απάντηση με `MODEL_PATH` που δεν υπάρχει (HTTP 503):

```json
{
  "status": "degraded",
  "model": null,
  "database": {"ok": true, "players": 1212, "latest_played_game_date": "2026-10-02", "next_scheduled_game_date": "2026-10-07"},
  "data_age_days": 1,
  "data_loaded_through": null,
  "problems": ["model: model artifact is missing or incompatible"]
}
```

Με βάση που δεν υπάρχει το `database.ok` είναι `false`, τα υπόλοιπα πεδία της βάσης `null` και το πρόβλημα `database: database is not available or not initialised`. Σε degraded κατάσταση τα `/predict`, `/rankings` και `/players` απαντούν 503 με `{"detail": "prediction model is not available"}` (ή `"database is not available"`), ενώ το `GET /availability` δουλεύει όσο η βάση φτάνει.

### 5.2 `GET /predict/{player_id}`

| Παράμετρος | Τύπος | Περιγραφή |
|---|---|---|
| `player_id` (path) | string | `P` + 6 ψηφία (π.χ. `P007200`) ή παλαιά μορφή `P` + 3 γράμματα (π.χ. `PADF`). Γίνεται strip και κεφαλαία πριν από τον έλεγχο, άρα `%20plru%20` ισοδυναμεί με `PLRU` |
| `include_features` (query) | bool, προεπιλογή `false` | Αν `true`, η απάντηση περιλαμβάνει και το `features` με τις τιμές των 42 features |

```bash
curl -s http://127.0.0.1:8000/predict/P003469
```

```json
{
  "player_id": "P003469",
  "name": "VEZENKOV, SASHA",
  "team_code": "OLY",
  "team_name": "OLYMPIACOS PIRAEUS",
  "is_active": true,
  "next_game": {
    "season": 2026,
    "gamecode": 40,
    "game_date": "2026-10-09",
    "tipoff_utc": "2026-10-09T18:15:00Z",
    "opponent_code": "IST",
    "opponent_name": "ANADOLU EFES ISTANBUL",
    "home": true
  },
  "predicted_fantasy": 20.13,
  "model_predicted_fantasy": 20.13,
  "predicted_pir": 18.64,
  "availability": {"status": "available", "source": null, "note": null, "expected_return": null, "updated_at": null},
  "n_prior_appearances": 286,
  "last_appearance_date": "2026-10-01",
  "model_version": "20261003T140043Z-7ae47948",
  "notes": []
}
```

| Πεδίο | Σημασία |
|---|---|
| `predicted_fantasy` | **Αποτελεσματική** πρόβλεψη μετά το override διαθεσιμότητας: ίση με το `model_predicted_fantasy`, εκτός από τους παίκτες `out` όπου είναι `0.0` |
| `model_predicted_fantasy` | Ακατέργαστη έξοδος του μοντέλου |
| `predicted_pir` | Πρόβλεψη PIR από ανεξάρτητο μοντέλο, πάντα ακατέργαστη τιμή |
| `availability` | Η εγγραφή διαθεσιμότητας. Χωρίς εγγραφή: `status: "available"` και όλα τα άλλα `null` |
| `next_game` | Ο επόμενος αγώνας της ομάδας του παίκτη, ή `null`. Το `tipoff_utc` είναι ISO-8601 σε UTC με το επίθημα `Z` (`null` αν η ώρα δεν είναι γνωστή) |
| `team_code`, `team_name` | Η ομάδα της **τελευταίας γραμμής** του παίκτη στη βάση· τα ονόματα ομάδων διαβάζονται από τον πίνακα `teams` και κρατιούνται σε cache στο startup και στο refresh |
| `features` | Μόνο με `include_features=true`: οι 42 τιμές των features (4 δεκαδικά, `null` όπου λείπουν) |

Παίκτης που δεν είναι ενεργός (ο Shane Larkin, τελευταία γραμμή τον Απρίλιο 2026, καμία στη σεζόν 2026):

```bash
curl -s http://127.0.0.1:8000/predict/P007200
```

```json
{
  "player_id": "P007200",
  "name": "LARKIN, SHANE",
  "team_code": "IST",
  "team_name": "ANADOLU EFES ISTANBUL",
  "is_active": false,
  "next_game": {"season": 2026, "gamecode": 40, "game_date": "2026-10-09", "tipoff_utc": "2026-10-09T18:15:00Z", "opponent_code": "OLY", "opponent_name": "OLYMPIACOS PIRAEUS", "home": false},
  "predicted_fantasy": 7.84,
  "model_predicted_fantasy": 7.84,
  "predicted_pir": 8.2,
  "availability": {"status": "available", "source": null, "note": null, "expected_return": null, "updated_at": null},
  "n_prior_appearances": 258,
  "last_appearance_date": "2026-04-08",
  "model_version": "20261003T140043Z-7ae47948",
  "notes": ["player is not active (no games in the latest season or in the last 45 days): team and next game may be outdated"]
}
```

Σφάλματα:

| Κωδικός | Πότε | Απάντηση |
|---|---|---|
| 404 | Έγκυρη μορφή αλλά άγνωστος παίκτης (π.χ. `PXXXXXX`) | `{"detail": "player not found; use /players?search=<name> to look up a player_id"}` |
| 422 | Μορφή που δεν ταιριάζει στο `^P[A-Z0-9]{3,6}$` (π.χ. `abc`), ή άκυρη τιμή `include_features` | `{"detail": [{"type": "string_pattern_mismatch", "loc": ["path", "player_id"], "msg": "String should match pattern '^P[A-Z0-9]{3,6}$'", "input": "abc", "ctx": {"pattern": "^P[A-Z0-9]{3,6}$"}}]}` |
| 503 | Το μοντέλο ή η βάση δεν είναι διαθέσιμα | `{"detail": "prediction model is not available"}` |

### 5.3 `GET /rankings`

| Παράμετρος | Προεπιλογή | Περιγραφή |
|---|---|---|
| `limit` | 50 | Μέγιστο πλήθος παικτών (1 έως 500) |
| `offset` | 0 | Πόσοι παίκτες παραλείπονται από την αρχή (≥ 0) |
| `team` | όλες | Κωδικός ομάδας, 3 χαρακτήρες, πεζά ή κεφαλαία (π.χ. `oly`). Άγνωστος κωδικός δίνει 404 με τους έγκυρους κωδικούς· κωδικός που δεν έχει 3 χαρακτήρες δίνει 422 |
| `include_unavailable` | `false` | Αν `true`, περιλαμβάνονται και οι παίκτες `out` |
| `active_only` | `true` | Αν `false`, όλοι οι γνωστοί παίκτες της βάσης (και όσοι έχουν φύγει από τη λίγκα) |

```bash
curl -s "http://127.0.0.1:8000/rankings?team=OLY&limit=3"
```

```json
{
  "meta": {"as_of": "2026-10-03", "model_version": "20261003T140043Z-7ae47948", "total": 14, "limit": 3, "offset": 0},
  "items": [
    {"rank": 1, "player_id": "P003469", "name": "VEZENKOV, SASHA", "team_code": "OLY", "predicted_fantasy": 20.13, "model_predicted_fantasy": 20.13, "predicted_pir": 18.64, "next_opponent_code": "IST", "next_home": true, "next_game_date": "2026-10-09", "availability_status": "available", "is_active": true},
    {"rank": 2, "player_id": "P010042", "name": "MONTERO, JEAN", "team_code": "OLY", "predicted_fantasy": 17.46, "model_predicted_fantasy": 17.46, "predicted_pir": 15.99, "next_opponent_code": "IST", "next_home": true, "next_game_date": "2026-10-09", "availability_status": "available", "is_active": true},
    {"rank": 3, "player_id": "P009849", "name": "DORSEY, TYLER", "team_code": "OLY", "predicted_fantasy": 15.4, "model_predicted_fantasy": 15.4, "predicted_pir": 14.51, "next_opponent_code": "IST", "next_home": true, "next_game_date": "2026-10-09", "availability_status": "available", "is_active": true}
  ]
}
```

Οι πέντε πρώτοι του `GET /rankings?limit=5` στη σημερινή βάση (262 ενεργοί παίκτες):

| rank | player_id | Παίκτης | Ομάδα | predicted_fantasy | Επόμενος αγώνας |
|---|---|---|---|---|---|
| 1 | `P009846` | BRYANT, ELIJAH | HTA | 20,9 | 2026-10-08, εκτός, PAM |
| 2 | `P003469` | VEZENKOV, SASHA | OLY | 20,13 | 2026-10-09, εντός, IST |
| 3 | `P011286` | BACON, DWAYNE | DUB | 18,08 | 2026-10-08, εντός, RED |
| 4 | `P013369` | JONES, CARLIK | PAR | 17,86 | 2026-10-08, εκτός, MAD |
| 5 | `P012796` | KABENGELE, MFIONDU | DUB | 17,86 | 2026-10-08, εντός, RED |

**Κανόνες ταξινόμησης και σελιδοποίησης**

- Η ταξινόμηση είναι φθίνουσα κατά το **αποτελεσματικό** `predicted_fantasy` (μετά το override), και σε ισοβαθμία κατά `player_id`.
- Το `rank` είναι η θέση στη φιλτραρισμένη λίστα και είναι **συνεχόμενο ανεξάρτητα από το `offset`** (με `offset=3` το πρώτο στοιχείο έχει `rank` 4). Το `meta.total` είναι το πλήθος μετά τα φίλτρα και πριν από το `limit`/`offset`.
- Οι παίκτες `out` **εξαιρούνται** (και δεν μετρούν στο `total`). Με `include_unavailable=true` εμφανίζονται **πάντα στο τέλος**, με αποτελεσματική τιμή `0.0` (η ακατέργαστη τιμή του μοντέλου μένει στο `model_predicted_fantasy`), ταξινομημένοι μεταξύ τους κατά την πρόβλεψη του μοντέλου. Το «πάντα στο τέλος» ισχύει ρητά και όταν κάποιος άλλος παίκτης έχει αρνητική πρόβλεψη (η τιμή 0 των `out` θα τους έβαζε αλλιώς πριν από αυτόν).
- Οι `doubtful` εμφανίζονται κανονικά στη θέση τους, με `availability_status: "doubtful"`.
- Μια ομάδα χωρίς ενεργούς παίκτες (ή όπου όλοι είναι `out`) δίνει HTTP 200 με `items: []` και `total: 0`. Οι έγκυροι κωδικοί `team` είναι όλες οι ομάδες του πίνακα `teams` (32, και όσες δεν παίζουν πια).

Σφάλματα: 404 για άγνωστο `team` (π.χ. `{"detail": "unknown team code 'ZZZ'; valid codes: ASV, BAM, BAR, BAS, BER, BES, BUD, CAN, CSK, DAR, DUB, DYR, GAL, HTA, IST, KHI, MAD, MAL, MCO, MIL, MUN, OLY, PAM, PAN, PAR, PRS, RED, TEL, ULK, UNK, VIR, ZAL"}`), 422 για παραμέτρους εκτός ορίων (π.χ. `limit=501`: `"Input should be less than or equal to 500"`), 503 σε degraded κατάσταση.

### 5.4 `GET /players`

Βρίσκει το `player_id` ενός παίκτη.

| Παράμετρος | Προεπιλογή | Περιγραφή |
|---|---|---|
| `search` | χωρίς φίλτρο | Κείμενο αναζήτησης στο όνομα (έως 100 χαρακτήρες) |
| `team` | όλες | Κωδικός ομάδας (η **τελευταία** ομάδα του παίκτη), όπως στα `/rankings` |
| `limit` | 20 | Μέγιστο πλήθος αποτελεσμάτων (1 έως 100) |

Η αναζήτηση **δεν κάνει διάκριση πεζών/κεφαλαίων και τόνων** και αγνοεί τα σημεία στίξης των ονομάτων (`'`, `.`, `-`, `,`). Κάθε λέξη του `search` πρέπει να εμφανίζεται ως υποσυμβολοσειρά στο όνομα, με οποιαδήποτε σειρά: τα `vezenkov`, `Vezenkóv`, `sasha vezenkov` και `zenk` βρίσκουν όλα τον `VEZENKOV, SASHA`· το `aj lawson` βρίσκει τον `LAWSON, A.J.`. Περιορισμός: γράμματα όπως το `đ` γίνονται `d` (όχι `dj`), άρα το `djordjevic` δεν βρίσκει τον `ĐORĐEVIĆ`. Επιστρέφονται και οι ανενεργοί παίκτες: πρώτα οι ενεργοί, μετά αλφαβητικά.

```bash
curl -s "http://127.0.0.1:8000/players?search=vezenkov"
```

```json
{
  "total": 1,
  "limit": 20,
  "items": [
    {"player_id": "P003469", "name": "VEZENKOV, SASHA", "team_code": "OLY", "team_name": "OLYMPIACOS PIRAEUS", "last_season": 2026, "is_active": true}
  ]
}
```

Το `total` είναι το πλήθος των παικτών που ταιριάζουν (πριν από το `limit`). Το κενό ή μόνο από κενά `search` ισοδυναμεί με καθόλου `search`. Άγνωστο `team`: 404· κακής μορφής `team`, `limit` εκτός ορίων ή `search` μεγαλύτερο από 100 χαρακτήρες: 422.

### 5.5 `GET /availability`

Δημόσια, μόνο για ανάγνωση. Παράμετρος `status` (`out`, `doubtful` ή `available`) για φίλτρο. Η λίστα είναι ταξινομημένη: πρώτα οι `out`, μετά οι `doubtful`, μετά οι `available`, και μέσα σε κάθε κατάσταση αλφαβητικά. Παίκτης χωρίς εγγραφή δεν εμφανίζεται (θεωρείται διαθέσιμος).

```json
{
  "total": 1,
  "items": [
    {"player_id": "P009846", "name": "BRYANT, ELIJAH", "status": "out", "source": "local check", "note": "temporary test record", "expected_return": "2026-10-20", "updated_at": "2026-10-03T15:08:04.372205Z"}
  ]
}
```

### 5.6 `POST /availability` (προστατευμένο)

Δημιουργεί ή **αντικαθιστά ολόκληρη** την εγγραφή του παίκτη (idempotent upsert). Τα πεδία που δεν δίνονται γίνονται `null`: δεν γίνεται συγχώνευση με την προηγούμενη εγγραφή. Η απάντηση είναι πάντα 200 με την εγγραφή όπως αποθηκεύτηκε.

| Πεδίο σώματος | Υποχρεωτικό | Κανόνες |
|---|---|---|
| `player_id` | ναι | Όπως στο `/predict`: strip, κεφαλαία, μορφή `^P[A-Z0-9]{3,6}$` (422 αν δεν ταιριάζει). Ο παίκτης πρέπει να υπάρχει στη βάση (404 αλλιώς) |
| `status` | ναι | Ακριβώς `out`, `doubtful` ή `available` (πεζά). Αλλιώς 422 |
| `source` | όχι | Προέλευση της πληροφορίας, έως 100 χαρακτήρες |
| `note` | όχι | Σημείωση, **έως 500 χαρακτήρες** (422 αν είναι μεγαλύτερη, δεν κόβεται) |
| `expected_return` | όχι | Ημερομηνία `YYYY-MM-DD`. Άκυρες ημερομηνίες, αριθμοί και datetime δίνουν 422 |

Τα `source` και `note` γίνονται strip και το κενό κείμενο γίνεται `null` (το όριο μήκους ελέγχεται μετά το strip). Άγνωστα πεδία απορρίπτονται με 422· ειδικά το `updated_at` δεν μπορεί να δοθεί από τον client, γιατί ορίζεται πάντα από τον server, σε UTC, από το ρολόι της εφαρμογής.

```bash
curl -s -X POST http://127.0.0.1:8000/availability \
  -H "X-API-Key: $ADMIN_API_KEY" -H "Content-Type: application/json" \
  -d '{"player_id": "P009846", "status": "out", "source": "local check", "note": "temporary test record", "expected_return": "2026-10-20"}'
```

```json
{"player_id": "P009846", "name": "BRYANT, ELIJAH", "status": "out", "source": "local check", "note": "temporary test record", "expected_return": "2026-10-20", "updated_at": "2026-10-03T15:08:04.372205Z"}
```

Σφάλματα: 401 (λείπει ή είναι λάθος το κλειδί), 503 (`admin API is disabled` όταν δεν έχει οριστεί κλειδί, ή η βάση δεν είναι διαθέσιμη), 404 (άγνωστος παίκτης), 422 (επικύρωση).

### 5.7 `DELETE /availability/{player_id}` (προστατευμένο)

Σβήνει την εγγραφή του παίκτη, ο οποίος θεωρείται πάλι διαθέσιμος. Επιτυχία: **204** χωρίς σώμα.

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X DELETE -H "X-API-Key: $ADMIN_API_KEY" http://127.0.0.1:8000/availability/P009846
# 204
```

Σφάλματα: 404 αν ο παίκτης είναι άγνωστος (`player not found; …`) ή δεν έχει εγγραφή (`{"detail": "no availability record for this player"}`), 422 για άκυρο `player_id`, 401/503 όπως παραπάνω. Ο έλεγχος κλειδιού γίνεται **πρώτος**: αίτημα χωρίς κλειδί παίρνει πάντα 401 και δεν αποκαλύπτει αν ο παίκτης υπάρχει.

### 5.8 `POST /admin/refresh` (προστατευμένο)

Ξαναδιαβάζει από τη βάση ιστορικό, αγώνες, πρόγραμμα, ονόματα ομάδων και παικτών, ξαναϋπολογίζει τις προβλέψεις της ημέρας και αναφέρει το νέο data cutoff. Βλ. ενότητα 11.

```bash
curl -s -X POST -H "X-API-Key: $ADMIN_API_KEY" http://127.0.0.1:8000/admin/refresh
```

```json
{
  "status": "refreshed",
  "model_version": "20261003T140043Z-7ae47948",
  "players": 1212,
  "previous_latest_played_game_date": "2026-10-02",
  "latest_played_game_date": "2026-10-02",
  "next_scheduled_game_date": "2026-10-07",
  "refreshed_at": "2026-10-03T15:08:06.333429Z",
  "duration_seconds": 1.723
}
```

Το `previous_latest_played_game_date` είναι το cutoff που έβλεπε η υπηρεσία πριν από την ανανέωση και το `latest_played_game_date` το νέο. Αν η ανανέωση αποτύχει επειδή η βάση δεν φτάνει, απαντά 503 και οι προηγούμενες προβλέψεις συνεχίζουν να εξυπηρετούνται.

## 6. Μορφή σφαλμάτων

Κάθε σφάλμα είναι JSON με πεδίο `detail`: **σύντομο μήνυμα στα αγγλικά** (για τους clients), και για τα σφάλματα επικύρωσης (422) η λίστα του FastAPI με `type`, `loc`, `msg`, `input`. Ποτέ stack trace, διαδρομές αρχείων, SQL ή credentials: όλες οι λεπτομέρειες γράφονται στο log του server.

| Κωδικός | Πότε |
|---|---|
| 401 | Λείπει ή είναι λάθος το `X-API-Key`: `{"detail": "invalid or missing API key"}` |
| 404 | Άγνωστος παίκτης ή ομάδα, εγγραφή διαθεσιμότητας που δεν υπάρχει, άγνωστη διαδρομή (`{"detail": "Not Found"}`) |
| 405 | Λάθος μέθοδος (`{"detail": "Method Not Allowed"}`) |
| 422 | Επικύρωση παραμέτρων ή σώματος |
| 500 | Απρόβλεπτο σφάλμα: `{"detail": "internal server error"}` (η λεπτομέρεια μόνο στο log) |
| 503 | Δεν υπάρχει μοντέλο (`prediction model is not available`), βάση (`database is not available`, `database unavailable` για σφάλμα βάσης σε αίτημα) ή κλειδί διαχειριστή (`admin API is disabled`) |

## 7. Διαθεσιμότητα παικτών (τραυματισμοί και απουσίες)

Το μοντέλο αγνοεί τραυματισμούς, απουσίες και αλλαγές ρόστερ. Ένας διαχειριστής μπορεί να καταχωρήσει χειροκίνητα τη διαθεσιμότητα ενός παίκτη στον πίνακα `player_availability` (`player_id` κλειδί και foreign key στον `players`, `status`, `source`, `note`, `expected_return`, `updated_at` σε UTC). Παίκτης χωρίς εγγραφή θεωρείται διαθέσιμος.

| Κατάσταση | `predicted_fantasy` | `/rankings` | `notes` |
|---|---|---|---|
| `out` | **`0.0`** (το `model_predicted_fantasy` μένει ακατέργαστο) | Εξαιρείται· με `include_unavailable=true` εμφανίζεται στο τέλος | `player is marked out: effective prediction is 0` |
| `doubtful` | Ίδια με το μοντέλο | Κανονικά, με `availability_status: doubtful` | `player is marked doubtful: prediction assumes the player plays` |
| `available` ή καμία εγγραφή | Ίδια με το μοντέλο | Κανονικά | — |

**Πώς εφαρμόζεται.** Το override εφαρμόζεται **ΜΕΤΑ** την πρόβλεψη, στο επίπεδο του API, και **δεν περνά ποτέ ως feature στο μοντέλο**: η διαθεσιμότητα δεν υπάρχει στο ιστορικό εκπαίδευσης, άρα το μοντέλο δεν θα μπορούσε να τη μάθει και θα εξαρτιόταν από δεδομένα που δεν υπάρχουν στο παρελθόν. Η εγγραφή διαβάζεται από τη βάση σε κάθε αίτημα, οπότε ισχύει **αμέσως** μετά το `POST` (χωρίς refresh) σε όλες τις διεργασίες.

**Το `predicted_pir` δεν αλλάζει** σε καμία κατάσταση: είναι πάντα η ακατέργαστη έξοδος του ανεξάρτητου μοντέλου PIR, ακόμη και για παίκτη `out`. Το override αφορά μόνο το αποτελεσματικό `predicted_fantasy`.

**Τι δεν καλύπτει.** Το override αφορά μόνο τον ίδιο τον παίκτη. Δεν αναπροσαρμόζει τα λεπτά και τη χρήση των συμπαικτών όταν ένας βασικός απουσιάζει και δεν μεταβάλλει το πλαίσιο της ομάδας. Οι εγγραφές είναι χειροκίνητες και **δεν λήγουν μόνες τους**: μετά την `expected_return` το `/predict` προσθέτει προειδοποίηση, αλλά ο παίκτης παραμένει `out` μέχρι να αλλάξει ή να αφαιρεθεί η εγγραφή.

**Σχήμα του πίνακα** (και στο `docs/INGESTION.md`, ενότητα 5): `player_id` (primary key, FK → `players`), `status` (`CHECK status IN ('out', 'doubtful', 'available')`, όνομα constraint `ck_player_availability_status`), `source`, `note`, `expected_return` (`DATE`), `updated_at` (`TIMESTAMP WITH TIME ZONE`, not null). Χρησιμοποιούνται μόνο γενικοί τύποι της SQLAlchemy, άρα το σχήμα δουλεύει αυτούσιο σε SQLite και Postgres (στο Postgres: `text`, `date`, `timestamptz`· το σχήμα ελέγχεται και σε πραγματικό Postgres, `docs/DATABASE.md`), και η εγγραφή γίνεται με το υπάρχον `upsert()` του `db/session.py`. **Στην SQLite** ο πίνακας δημιουργείται στο startup της εφαρμογής αν λείπει (`checkfirst`), και επίσης από το `create_all` του ingestion. **Στο Postgres** δημιουργείται από το migration `001_init.sql` (με Row Level Security από το `002`)· το startup ελέγχει μόνο ότι υπάρχει και δεν τον δημιουργεί ποτέ, ούτε χρειάζεται δικαίωμα `CREATE`. Το `updated_at` είναι `timestamptz`: γράφεται και διαβάζεται πάντα ως UTC ανεξάρτητα από τη ζώνη ώρας της συνεδρίας του server.

## 8. Ασφάλεια

- **Προστατευμένα endpoints:** `POST /availability`, `DELETE /availability/{player_id}`, `POST /admin/refresh`. Όλα τα υπόλοιπα είναι δημόσια και μόνο για ανάγνωση.
- **`X-API-Key`.** Το header συγκρίνεται με το `ADMIN_API_KEY` με `secrets.compare_digest` (σταθερός χρόνος, πάνω σε bytes UTF-8, άρα δουλεύει και με μη-ASCII τιμές). Λείπει ή είναι λάθος: 401. Το κλειδί δίνεται **μόνο** στο header (όχι ως query ή cookie).
- **Κλειστά όταν δεν υπάρχει κλειδί.** Αν το `ADMIN_API_KEY` είναι κενό (ή μόνο κενά), τα προστατευμένα endpoints απαντούν **503 `admin API is disabled`**, ανεξάρτητα από το header που στέλνεται: ποτέ ανοιχτά. Το `GET /availability` δουλεύει κανονικά.
- **Το κλειδί δεν εμφανίζεται ποτέ** σε logs, σε μηνύματα σφάλματος ή στο OpenAPI (ελέγχεται από tests και από πραγματικό τρέξιμο). Στο Swagger δηλώνεται ως security scheme `APIKeyHeader`: με το κουμπί **Authorize** του `/docs` βάζεις το κλειδί μία φορά και δουλεύουν όλα τα «Try it out».
- **Σύσταση:** μεγάλη τυχαία τιμή (π.χ. `python -c "import secrets; print(secrets.token_urlsafe(32))"`), μόνο μέσω περιβάλλοντος (Render secret), ποτέ σε αρχείο που γίνεται commit, και μόνο πάνω από HTTPS (το Render παρέχει TLS). Αλλαγή κλειδιού: νέα τιμή στο περιβάλλον και επανεκκίνηση.
- Η υπηρεσία δεν έχει rate limiting ούτε προστασία από brute force του κλειδιού (ενότητα 13).

## 9. Εκκίνηση, degraded κατάσταση και συμπεριφορά σε αποτυχίες

Η φόρτωση γίνεται στο lifespan της εφαρμογής: ανοίγει η κοινή engine, δημιουργείται (αν λείπει, **μόνο στην SQLite**) ή ελέγχεται (στο Postgres) ο πίνακας `player_availability`, φορτώνεται το μοντέλο, και υπολογίζονται εκ των προτέρων οι προβλέψεις της ημέρας (**warm-up**), ώστε το πρώτο αίτημα να μην πληρώσει τον υπολογισμό και ένα πρόβλημα να φανεί από την αρχή. Κάθε αποτυχία καταγράφεται στο log και γίνεται **degraded**, χωρίς να σταματά η εφαρμογή:

| Αποτυχία | `/health` | `/predict`, `/rankings`, `/players` | `/availability` |
|---|---|---|---|
| Λείπει, είναι κατεστραμμένο ή ασύμβατο το μοντέλο | 503, `model: null` | 503 `prediction model is not available` | δουλεύει |
| Η βάση δεν φτάνει ή δεν έχει αρχικοποιηθεί (στο Postgres: λείπουν πίνακες επειδή δεν εφαρμόστηκαν τα migrations) | 503, `database.ok: false` | 503 `database is not available` | 503 |
| Αποτυγχάνει ο πρώτος υπολογισμός προβλέψεων | 503, πρόβλημα `model` | 503 `prediction model is not available` | δουλεύει |
| Η βάση δεν έχει παίκτες | 503, `database.ok: false` | δουλεύουν (κενά αποτελέσματα στα `/rankings` και `/players`, 404 στο `/predict`) | δουλεύει |

Δεν γίνεται αυτόματη επανάληψη: μια degraded υπηρεσία γίνεται υγιής μόνο με επανεκκίνηση (το Render θα τη θεωρήσει unhealthy μέσω του `/health`).

## 10. Απόδοση

Μετρήσεις στο laptop (Windows 11, Python 3.13, πραγματική βάση με 1.212 παίκτες, `uvicorn` με μία διεργασία, loopback, 2026-10-03). **Δεν έχουν μετρηθεί στο Render free tier.**

| Μέτρηση | Αποτέλεσμα |
|---|---|
| Εκκίνηση μέχρι το πρώτο `200` του `/health` (δύο εκτελέσεις) | 6,4 s και 5,5 s (εισαγωγές, μοντέλο, βάση, warm-up) |
| Πρώτα αιτήματα μετά την εκκίνηση (`/health`, `/predict` Larkin, `/predict` Vezenkov) | 3,6 ms, 11,4 ms, 4,0 ms (το warm-up έχει ήδη υπολογίσει τις προβλέψεις) |
| «Ψυχρός» υπολογισμός όλων των προβλέψεων (`POST /admin/refresh`: ανάγνωση βάσης, features, μοντέλο) | 1,73 s (το `Predictor` δίνει 0,8 έως 0,9 s για τον υπολογισμό και 0,2 ms από cache, `docs/MODEL.md` ενότητα 12) |
| `GET /health` (median, 30 κλήσεις) | 3,5 ms |
| `GET /predict/P003469` | 3,6 ms |
| `GET /rankings` (50 στοιχεία) | 4,7 ms |
| `GET /rankings?limit=500&active_only=false` (και οι 1.212 παίκτες) | 11 ms |
| `GET /rankings?team=OLY` | 3,7 ms |
| `GET /players?search=vezenkov` | 4,1 ms |
| `GET /availability` | 3,2 ms |
| `POST` και `DELETE /availability` (εγγραφή στο SQLite) | περίπου 100 ms |
| 400 αιτήματα (μείγμα `/rankings`, `/predict`, `/players`, `/health`), 16 ταυτόχρονες συνδέσεις | 323 αιτήματα/s, median 47 ms, p95 61 ms, κανένα σφάλμα |

Ο υπολογισμός γίνεται **μία φορά ανά ημερομηνία `as_of`** (UTC) και κρατιέται σε cache του `Predictor` (έως 4 ημερομηνίες): τα αιτήματα δεν ξαναϋπολογίζουν ιστορικό. Ανά αίτημα διαβάζεται μόνο ο μικρός πίνακας διαθεσιμότητας. Το πρώτο αίτημα μετά τα μεσάνυχτα UTC πληρώνει έναν νέο υπολογισμό (περίπου 1 s τοπικά).

## 11. Ανανέωση δεδομένων

Η εφαρμογή δεν βλέπει νέα δεδομένα μετά από νέο ingestion μέχρι να κληθεί το `POST /admin/refresh` (ή να γίνει επανεκκίνηση). Τα ονόματα ομάδων και παικτών, το ιστορικό και το πρόγραμμα φορτώνονται στο startup και στο refresh. Κανονική ροή:

```bash
# 1. νέα δεδομένα στη βάση (όπου δείχνει το DATABASE_URL)
.venv/Scripts/python.exe -m elfantasy.ingest.pipeline --update
# 2. η υπηρεσία φορτώνει τα νέα δεδομένα (περίπου 1,7 s τοπικά)
curl -s -X POST -H "X-API-Key: $ADMIN_API_KEY" https://<host>/admin/refresh
# 3. έλεγχος: το data_loaded_through πρέπει να ισούται με το database.latest_played_game_date
curl -s https://<host>/health
```

- Το refresh **δεν αγγίζει τη διαθεσιμότητα** (δεν είναι στην cache).
- Αν η βάση δεν φτάνει κατά το refresh, η απάντηση είναι 503 και οι προηγούμενες προβλέψεις συνεχίζουν να εξυπηρετούνται.
- **Μία διεργασία:** με πολλούς workers κάθε διεργασία έχει δική της cache και το refresh ανανεώνει μόνο αυτήν που το δέχτηκε· εκεί χρειάζεται επανεκκίνηση όλων.

## 12. Tests

Όλα τα tests τρέχουν χωρίς δίκτυο, με `fastapi.testclient.TestClient` πάνω σε προσωρινή βάση SQLite και μικρό μοντέλο (το συνθετικό πρωτάθλημα και τα μικρά μοντέλα της Φάσης 3), με ρολόι που ορίζεται από το test (`create_app(clock=…)`) και, όπου χρειάζεται, με τη βάση των πραγματικών fixtures (παλαιά και νέα IDs, παίκτης χωρίς μελλοντικό αγώνα).

```bash
.venv/Scripts/python.exe -m pytest tests/unit/test_api_*.py tests/integration/test_api_*.py -q          # μόνο το API
.venv/Scripts/python.exe -m pytest --cov=elfantasy.api --cov-branch --cov-report=term-missing           # coverage των νέων modules
.venv/Scripts/python.exe -m pytest -q                                                                  # όλα
```

| Αρχείο | Τι ελέγχει |
|---|---|
| `tests/integration/test_api_health.py` | `ok` και κάθε degraded περίπτωση (λείπει ή είναι κατεστραμμένο το μοντέλο, λείπει ή είναι κατεστραμμένη η βάση, άκυρο URL, κενή βάση, βάση που πέφτει μετά το startup, αποτυχία του πρώτου υπολογισμού), καμία διαρροή διαδρομών ή credentials, μία κοινή engine στο `Predictor.load`, import χωρίς μοντέλο/βάση (σε υποδιεργασία) |
| `tests/integration/test_api_predict.py` | Έγκυρος, άγνωστος, άκυρου σχήματος και παλαιάς μορφής ID, πεζά και κενά, παίκτης χωρίς αγώνα, ανενεργός, χωρίς ιστορικό, `include_features`, `out`/`doubtful`/`available`, στρογγυλοποίηση |
| `tests/integration/test_api_rankings.py` | Προεπιλογές, `limit`/`offset`/`rank`, φίλτρο ομάδας, ταξινόμηση, `out`, `include_unavailable`, `active_only`, κενό αποτέλεσμα, όρια παραμέτρων (422), cache (καμία νέα πρόβλεψη ανά αίτημα), ακριβείς κανόνες με ψεύτικο `Predictor` |
| `tests/integration/test_api_players.py` | Τόνοι, πεζά/κεφαλαία, σημεία στίξης, σειρά λέξεων, όρια, φίλτρο ομάδας, ταξινόμηση |
| `tests/integration/test_api_availability.py` | 401 χωρίς/με λάθος κλειδί, 503 χωρίς κλειδί στις ρυθμίσεις, upsert και αντικατάσταση, λίστα και φίλτρο, DELETE, επικύρωση, άμεση ισχύς σε `/predict` και `/rankings`, παλαιά IDs |
| `tests/integration/test_api_admin.py` | Προστασία και ανανέωση: νέος αγώνας και αλλαγή ονόματος ομάδας γίνονται ορατά μόνο μετά το refresh, αναφορά του cutoff |
| `tests/integration/test_api_openapi.py` | `/openapi.json`, `/docs`, security scheme, τεκμηριωμένοι κωδικοί, παραδείγματα έγκυρα ως προς τα μοντέλα, κανένα μυστικό στο schema |
| `tests/integration/test_api_errors.py` | Generic 500 και 503 χωρίς stack trace (η λεπτομέρεια μένει στο log), μορφή `detail` |
| `tests/integration/test_api_predictions_table.py` | Κανένα endpoint δεν γράφει στον πίνακα `predictions` (τα GET δεν έχουν παρενέργειες) |
| `tests/integration/test_postgres.py` (marker `postgres`) | Το API, ο `Predictor` και η διαθεσιμότητα (`timestamptz`) σε πραγματικό τοπικό Postgres, με αποτελέσματα ίδια με της SQLite· startup χωρίς migrations: degraded και κανένας πίνακας δεν δημιουργείται |
| `tests/unit/test_api_availability.py`, `test_api_services.py`, `test_api_state.py`, `test_api_deps.py` | Λογική override, αποθήκευση, constraints, ταυτόχρονα upserts, κανονικοποίηση αναζήτησης, σημειώσεις, startup/shutdown και ταξινόμηση αποτυχιών (και startup σε Postgres με ψεύτικο engine: έλεγχος χωρίς δημιουργία πίνακα), έλεγχος κλειδιού και επικύρωση εισόδου, χωρίς HTTP |

Τα νούμερα (αριθμός tests, coverage) αναφέρονται στο report της φάσης. Σημείωση για το `pyproject.toml`: το Starlette 1.7 θεωρεί παρωχημένο το `httpx` στο `TestClient` και προτείνει το `httpx2`· η συγκεκριμένη προειδοποίηση αγνοείται με `filterwarnings`, ώστε να μη θάβει άλλες. Αν στο μέλλον το Starlette αφαιρέσει την υποστήριξη του `httpx`, αρκεί να αντικατασταθεί το `httpx` με το `httpx2` στο `requirements-dev.txt`.

## 13. Περιορισμοί και ό,τι δεν επιβεβαιώθηκε

- **Postgres (Φάση 5).** Το API δοκιμάστηκε σε **πραγματικό τοπικό Postgres** (PostgreSQL 18.4, `uvicorn` και `curl`: `/health`, `/rankings`, `/predict`, `POST`/`GET`/`DELETE /availability`, `POST /admin/refresh`, και τα ίδια πάνω σε `TestClient` στα tests με marker `postgres`)· οι προβλέψεις είναι ίδιες με της SQLite (μέγιστη διαφορά 0,0). **Δεν δοκιμάστηκε στο ίδιο το Supabase**: pooler, SSL, IPv4/IPv6, καθυστέρηση δικτύου (κάθε αίτημα `/predict` και `/rankings` διαβάζει τον πίνακα διαθεσιμότητας, δηλαδή ένα ταξίδι δικτύου προς τη βάση) και παύση του free tier. Για τον transaction pooler (πόρτα 6543) το `get_engine` ορίζει αυτόματα `prepare_threshold=None` (δοκιμάστηκε μόνο τοπικά, χωρίς pooler). Βλ. `docs/DATABASE.md`, ενότητες 12 και 13.
- **Δεν δοκιμάστηκε στο Render:** ούτε χρόνος εκκίνησης, ούτε μνήμη, ούτε ταχύτητα στο free tier (όλες οι μετρήσεις είναι τοπικές). Δεν έχει ελεγχθεί αν η εκκίνηση (περίπου 6 s τοπικά) χωράει στα χρονικά όρια του Render σε αργό instance.
- **Python.** Τα tests του API περνούν σε Python 3.13.2 και 3.12.7 (με τις ίδιες εκδόσεις βιβλιοθηκών)· η 3.11 δεν δοκιμάστηκε.
- **Ένα worker.** Ο `Predictor` και η cache του ζουν στη διεργασία· με πολλούς workers το `POST /admin/refresh` ανανεώνει μόνο έναν.
- **Δεν υπάρχει rate limiting, προστασία από brute force του κλειδιού, όριο μεγέθους σώματος αιτήματος ή CORS.** Ό,τι περιορισμό παρέχει το Render στο επίπεδο του proxy ισχύει, αλλά δεν δοκιμάστηκε. Αν χρειαστεί κλήση από browser σε άλλο domain, πρέπει να προστεθεί CORS.
- **Ο «σήμερα» είναι UTC.** Ένας αγώνας με ώρα έναρξης κοντά στα μεσάνυχτα UTC αλλάζει από «επόμενος» σε «δεν υπάρχει» στις 00:00 UTC, ακόμη κι αν δεν έχει ενημερωθεί η βάση με το αποτέλεσμά του.
- **Ανανέωση χειροκίνητη.** Δεν υπάρχει αυτόματο ingestion ή αυτόματο refresh: ο διαχειριστής τρέχει το ingestion και μετά το `/admin/refresh` (το `data_loaded_through` του `/health` δείχνει αν χρειάζεται).
- **Μόνο ο ίδιος ο παίκτης στο override** (όχι επιπτώσεις στους συμπαίκτες) και **χειροκίνητες εγγραφές που δεν λήγουν μόνες τους** (ενότητα 7). Δεν υπάρχει ιστορικό αλλαγών διαθεσιμότητας (μόνο η τελευταία εγγραφή ανά παίκτη).
- **Αναζήτηση παικτών:** τα γράμματα που δεν αναλύονται σε βασικό γράμμα και τόνο (`đ`, `ł`, `ø`, `æ`, `œ`, `ı`) αντιστοιχούν σε συγκεκριμένο λατινικό γράμμα (`d`, `l`, `o`, `ae`, `oe`, `i`), άρα π.χ. το `djordjevic` δεν βρίσκει τον `ĐORĐEVIĆ`. Η ομάδα ενός παίκτη είναι της τελευταίας του γραμμής: μετά από μεταγραφή το καλοκαίρι είναι η παλιά ομάδα μέχρι να εμφανιστεί σε αγώνα της νέας (`docs/MODEL.md`, ενότητα 14).
- Ό,τι αναφέρεται στο `docs/MODEL.md` (ενότητα 14) ισχύει και εδώ: το μοντέλο εκπαιδεύτηκε έως τη σεζόν 2024, το MAE του 2025 δεν είναι εγγυημένο για το 2026-27, ο ορισμός του «ενεργού» παίκτη δεν ταυτίζεται με τη λίστα του πραγματικού παιχνιδιού fantasy.

## 14. Αποφάσεις σχεδιασμού και αποκλίσεις από το πλάνο

- **`out` πάντα τελευταίοι** με `include_unavailable=true`, με ρητό κανόνα (και όχι μόνο λόγω της τιμής 0): αλλιώς ένας παίκτης με αρνητική πρόβλεψη θα εμφανιζόταν μετά από έναν `out`.
- **`POST /availability` αντικαθιστά ολόκληρη την εγγραφή** (όπως PUT) και απαντά πάντα 200· ο άγνωστος παίκτης στο σώμα δίνει 404, όχι 422. Το `DELETE` απαντά 204 και 404 όταν δεν υπάρχει εγγραφή.
- **Warm-up στο startup** και degraded κατάσταση αν αποτύχει. Έξτρα πεδία στο `/health` σε σχέση με το πλάνο: `problems` και `data_loaded_through`· στο `/admin/refresh`: `previous_latest_played_game_date`.
- **Σημείωση για ξεπερασμένη εγγραφή** (`expected return date has passed …`): οι εγγραφές είναι χειροκίνητες και αλλιώς ένας παίκτης θα έμενε `out` για πάντα χωρίς καμία ένδειξη.
- **`HEAD /health`** (εκτός OpenAPI) για monitors που χρησιμοποιούν HEAD, γιατί το FastAPI δίνει 405 στα HEAD των routes GET.
- **Ένα αρχείο SQLite που δεν υπάρχει δεν δημιουργείται:** η SQLite θα δημιουργούσε σιωπηλά κενή βάση στην πρώτη σύνδεση και το `/health` θα έδειχνε παραπλανητικά υγιές.
- **Ο πίνακας `player_availability` ανήκει στο κοινό `metadata`** (`db/models.py`), άρα στην SQLite το `create_all` του ingestion τον δημιουργεί κι αυτό (κενό). Προστέθηκε σύμβαση ονομασίας `ck` για το CHECK constraint. Στο υπάρχον `tests/unit/test_db.py` ενημερώθηκαν μόνο τα δύο σημεία που μετρούσαν τους πίνακες (5 → 6). **Φάση 5:** στο Postgres ο πίνακας δημιουργείται μόνο από τα migrations και το startup του API δεν τον δημιουργεί ποτέ (`ensure_schema`)· το API δεν γράφει στον πίνακα `predictions` (η καταγραφή γίνεται από το `python -m elfantasy.model.record_predictions`).
- **Φίλτρο `team`:** 3 χαρακτήρες (γράμματα ή ψηφία) αντί για αυστηρά 3 γράμματα, γιατί τα συνθετικά tests χρησιμοποιούν κωδικούς όπως `T03`· οι πραγματικοί κωδικοί είναι 3 γράμματα. Σχηματικά άκυρος κωδικός: 422, άγνωστος: 404.
- **`features` στο `/predict`:** απουσιάζει εντελώς από την απάντηση όταν δεν ζητηθεί (και όχι `null`), και στρογγυλεύεται σε 4 δεκαδικά.
