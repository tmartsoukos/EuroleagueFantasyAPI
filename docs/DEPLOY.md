# CI/CD και deploy στο Render (Φάση 6)

Το έγγραφο περιγράφει πώς ελέγχεται και δημοσιεύεται αυτόματα το API: το workflow του GitHub Actions, το Blueprint του Render, τις αναπαραγώγιμες εκδόσεις πακέτων, τις μετρήσεις μνήμης για το free plan, και **βήμα-βήμα τι πρέπει να κάνει ο χρήστης** για να στηθεί το Render, τα μυστικά και ο έλεγχος μετά το deploy. Στο τέλος αναφέρεται τι ελέγχθηκε πραγματικά και τι **δεν** μπορεί να επαληθευτεί πριν το πρώτο run στο GitHub και στο Render.

## 1. Σύνοψη

| Αρχείο | Ρόλος |
|---|---|
| `.github/workflows/ci.yml` | Το workflow: `lint`, `test`, `quality-gate`, `deploy`, `coverage-badge` |
| `render.yaml` | Blueprint του Render: μία web υπηρεσία στο free plan, χωρίς αυτόματο deploy |
| `constraints.txt` | Ακριβείς εκδόσεις όλων των πακέτων (ίδιες με της εκπαίδευσης του μοντέλου) |
| `scripts/smoke_check.py` | Έλεγχος μετά το deploy: περιμένει το `/health` της νέας έκδοσης |
| `scripts/make_coverage_badge.py` | Παράγει το SVG badge κάλυψης (χωρίς εξωτερική υπηρεσία) |
| `scripts/quality_summary.py` | Τα νούμερα του quality gate στη σελίδα του run |
| `tests/unit/test_ci_config.py` | Ελέγχει το workflow, το `render.yaml` και το `constraints.txt` |

Αποφάσεις του χρήστη που υλοποιούνται εδώ: deploy μέσω **Deploy Hook του Render** (GitHub secret `RENDER_DEPLOY_HOOK_URL`) με το αυτόματο deploy του Render **απενεργοποιημένο**, CI και Render σε **Python 3.12**, badges build και coverage στο README, και quality gate τον έλεγχο MAE του `tests/quality`.

## 2. Ροή

```
push (κάθε branch εκτός από το badges) ή pull request
   |
   +--> lint ----------+
   +--> test ----------+--> deploy ------------------> smoke check του /health
   +--> quality-gate --+    (μόνο push στο main,       (περιμένει το νέο commit)
   |                         environment: production)
   |
   +--> test --> coverage-badge   (μόνο push στο main: γράφει το coverage.svg στο branch badges)
```

Τα `lint`, `test` και `quality-gate` τρέχουν **παράλληλα**. Το `deploy` ξεκινά μόνο αν πέρασαν και τα τρία και μόνο σε push στο `main`. Στα pull requests και στα άλλα branches τρέχουν μόνο οι έλεγχοι.

## 3. Τι κάνει κάθε job

Όλα τα jobs τρέχουν σε `ubuntu-24.04` (σταθερή εικόνα, όχι `ubuntu-latest`) με Python 3.12 και έχουν timeout. Τα δικαιώματα του workflow είναι `contents: read`· μόνο το `coverage-badge` έχει `contents: write`. Οι actions είναι καρφωμένες σε major tag (`actions/checkout@v6`, `actions/setup-python@v6`, `actions/upload-artifact@v6`). Επιλέχθηκαν οι νεότερες εκδόσεις που τρέχουν σε Node 24 (επιβεβαιώθηκε από το `action.yml` της καθεμίας)· οι `actions/checkout@v4` και `actions/setup-python@v5` των παραδειγμάτων της εκφώνησης τρέχουν σε Node 20, το οποίο ο GitHub αποσύρει σταδιακά.

| Job | Τι κάνει |
|---|---|
| `lint` | Εγκαθιστά **μόνο το ruff**, στην έκδοση του `constraints.txt`, και τρέχει `ruff check .` και `ruff format --check .` |
| `test` | Εγκαθιστά με `pip install -r requirements-dev.txt -c constraints.txt` και `pip install --no-deps -e .`, ξεκινά service container `postgres:17` και τρέχει όλα τα tests **εκτός από το `tests/quality`** με `--cov=elfantasy --cov-fail-under=95`. Ανεβάζει το `coverage.xml` ως artifact και δίνει το ποσοστό ως output του job (`coverage`) |
| `quality-gate` | Τρέχει `pytest tests/quality`: φορτώνει το committed `models/model.joblib` και το `tests/fixtures/holdout_2025.parquet` και αποτυγχάνει αν το MAE δεν είναι μικρότερο από το `MAE_THRESHOLD` (6,00). Στη σελίδα του run γράφει πίνακα με MAE, threshold και baselines |
| `deploy` | Βλ. παρακάτω |
| `coverage-badge` | Φτιάχνει το SVG από το ποσοστό του `test` και το γράφει ως `coverage.svg` στο **orphan branch `badges`** (το δημιουργεί αν δεν υπάρχει) με ταυτότητα `github-actions[bot]`. Αν το SVG δεν άλλαξε, δεν γίνεται commit. Δεν αγγίζει το `main` |

### 3.1 Postgres στο job `test`

Τα 42 tests με marker `postgres` τρέχουν σε πραγματικό Postgres 17 (το Supabase τρέχει Postgres 17). Το service container ορίζεται με `TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres`: ο κωδικός `postgres` είναι ο κωδικός ενός container μιας χρήσεως, όχι μυστικό (τα tests δέχονται **μόνο τοπικό host**, ποτέ remote βάση, και το `tests/unit/test_repo_hygiene.py` επιτρέπει URL με κωδικό μόνο προς `localhost`). Δύο μεταβλητές κάνουν το CI αυστηρό:

* `ELFANTASY_REQUIRE_POSTGRES=1`: ένα test με marker `postgres` που **παραλείπεται** αποτυγχάνει ολόκληρη την εκτέλεση (plugin στο `tests/conftest.py`), ώστε η κάλυψη του Postgres να μη χάνεται σιωπηλά.
* `ELFANTASY_PG_DISPOSABLE=1`: δηλώνει ότι ο server είναι προσωρινός. Έτσι τρέχει και το test που μιμείται τους ρόλους `anon` και `authenticated` του Supabase (δημιουργεί και διαγράφει ρόλους σε επίπεδο server· πριν παραλειπόταν σε οποιονδήποτε server δεν ήταν ο ενσωματωμένος). **Μην το ορίσεις σε server που χρησιμοποιείς για άλλη δουλειά.**

### 3.2 Το job `deploy`

Τρέχει μόνο σε push στο `main`, με `needs: [lint, test, quality-gate]`, `environment: production` και `concurrency: deploy-production` (ένα deploy τη φορά, χωρίς ακύρωση). Το secret περνά στο περιβάλλον του job (`RENDER_DEPLOY_HOOK_URL: ${{ secrets.RENDER_DEPLOY_HOOK_URL }}`) και τα steps ελέγχουν `env.RENDER_DEPLOY_HOOK_URL != ''`: το `secrets` δεν επιτρέπεται μέσα σε `if`.

1. **Χωρίς secret:** το job τελειώνει **επιτυχώς** με μήνυμα στο summary «deploy skipped: secret not configured». Έτσι το CI δεν «κοκκινίζει» πριν στηθεί το Render.
2. **Με secret:** `curl --fail-with-body -sS -X POST` προς το hook, με έως 3 προσπάθειες (αναμονή 10 και 20 s). Τα σφάλματα του client (401 λάθος hook, 404 άγνωστη υπηρεσία, 409 ανεσταλμένη υπηρεσία) δεν επαναλαμβάνονται. Το URL **δεν τυπώνεται ποτέ**: κρύβεται ρητά με `::add-mask::` (και το URL και η μορφή του με το `ref`), δεν υπάρχει `set -x` και τα μηνύματα του curl δεν το περιέχουν.
3. Το hook καλείται με `ref=<SHA του commit>`: το Render κάνει deploy **ακριβώς του commit που πέρασε τους ελέγχους** και όχι «ό,τι είναι τελευταίο στο branch» (αν είχε προλάβει να γίνει νεότερο push, δεν θα έφευγε στην παραγωγή αδοκίμαστο).
4. **Smoke check** (αν υπάρχει το repository variable `RENDER_SERVICE_URL`, αλλιώς παραλείπεται με μήνυμα): `scripts/smoke_check.py` ρωτά το `<url>/health` με backoff (5 s, ×1,5, έως 30 s) μέχρι 15 λεπτά και απαιτεί HTTP 200, `"status": "ok"` **και** `commit` ίσο με το SHA του deploy. Αποτυχία = αποτυχία του job, με την τελευταία απάντηση στο log και στο summary.

**Γιατί ο smoke check απαιτεί το `commit`.** Το Render χτίζει τη νέα έκδοση ενώ η παλιά εξακολουθεί να εξυπηρετεί αιτήματα (αναβάθμιση χωρίς διακοπή), και αν η νέα δεν περάσει το health check, ακυρώνει το deploy και κρατά την παλιά. Ένας έλεγχος που περιμένει απλώς `status: ok` θα περνούσε αμέσως από την **παλιά** έκδοση, ακόμη κι αν το deploy αποτύχει. Γι' αυτό το `GET /health` δηλώνει πλέον το `commit` (από την `RENDER_GIT_COMMIT` που ορίζει το Render, `docs/API.md` ενότητα 5.1) και ο έλεγχος περιμένει να δει το νέο. Αν η υπηρεσία δεν δηλώνει commit (`null`), ο έλεγχος περνά με ρητή σημείωση ότι δεν μπορεί να επιβεβαιώσει την έκδοση.

## 4. Αναπαραγώγιμες εκδόσεις (`constraints.txt`)

Το `requirements.txt` δίνει ελάχιστες εκδόσεις· το `constraints.txt` καρφώνει **ακριβείς** εκδόσεις για όλα τα πακέτα, και το CI και το Render το χρησιμοποιούν με `pip install -r requirements.txt -c constraints.txt`. Το μοντέλο `models/model.joblib` φορτώνεται σωστά μόνο με τις εκδόσεις του xgboost και του scikit-learn της εκπαίδευσης, και το quality gate δίνει τα ίδια νούμερα μόνο με τις ίδιες εκδόσεις numpy, pandas και xgboost. Το `tests/unit/test_ci_config.py` ελέγχει ότι οι εκδόσεις του αρχείου ταυτίζονται με το `library_versions` του `models/metrics.json`.

**Πώς παράχθηκε.** Από το περιβάλλον όπου εκπαιδεύτηκε το μοντέλο (Windows 11, Python 3.13.2): `pip freeze --exclude-editable`, χωρίς το ίδιο το πακέτο (editable) και χωρίς το `colorama`, που εγκαθίσταται μόνο στα Windows. Οδηγίες ανανέωσης υπάρχουν στο header του αρχείου. Αν αλλάξουν τα numpy, scipy, scikit-learn, pandas ή xgboost, το μοντέλο πρέπει να ξαναεκπαιδευτεί και να γίνουν commit μαζί το νέο `constraints.txt` και το νέο artifact.

**Έλεγχος σε Linux με Python 3.12** (ό,τι θα βρει το CI και το Render): σε καθαρό venv με Python 3.12.3 σε Ubuntu 24.04, το `pip install -r requirements-dev.txt -c constraints.txt` εγκατέστησε ακριβώς τις ίδιες εκδόσεις με το περιβάλλον των Windows. Η μόνη διαφορά ήταν το `nvidia-nccl-cu13`, εξάρτηση του xgboost που εγκαθίσταται μόνο σε Linux (305 MB wheel)· καρφώθηκε στο `constraints.txt` με `; sys_platform == "linux"`. Επίσης `pip install --dry-run --python-version 3.13` για Linux βρήκε wheels για όλες τις εκδόσεις, ενώ για Python 3.11 η επίλυση αποτυγχάνει (`ResolutionImpossible`).

**Python ≥ 3.12.** Οι εκδόσεις numpy 2.5, scipy 1.18 και xgboost 3.4 του μοντέλου απαιτούν Python 3.12 ή νεότερη. Γι' αυτό το `requires-python` του `pyproject.toml` έγινε `>=3.12` (πριν ήταν `>=3.11`) και το `target-version` του ruff `py312`. Με Python 3.11 το pip θα διάλεγε παλαιότερες εκδόσεις και το μοντέλο θα φόρτωνε με `ModelVersionWarning`.

## 5. Οδηγίες για τον χρήστη, βήμα-βήμα

Ο κώδικας του workflow δεν χρειάζεται καμία αλλαγή. Πριν από τα παρακάτω, το CI τρέχει κανονικά και το `deploy` απλώς παραλείπεται με μήνυμα. Τα ονόματα των κουμπιών στα dashboards του Render, του Supabase και του GitHub μπορεί να αλλάξουν· τα παρακάτω βασίζονται στην τεκμηρίωσή τους (Οκτώβριος 2026) και δεν ελέγχθηκαν στα πραγματικά dashboards.

### Βήμα 1. Δημιουργία της υπηρεσίας στο Render

**Με Blueprint (προτεινόμενο).** Στο dashboard του Render: *New* → *Blueprint*, σύνδεσε τον λογαριασμό GitHub αν δεν είναι ήδη συνδεδεμένος, διάλεξε το repo `tmartsoukos/EuroleagueFantasyAPI` και το branch `main`. Το Render διαβάζει το `render.yaml` από τη ρίζα του repo και σου ζητά τιμές για τις μεταβλητές `DATABASE_URL` και `ADMIN_API_KEY` (βήματα 2 και 3). Η υπηρεσία δημιουργείται με όνομα `euroleague-fantasy-api`, region Frankfurt, free plan.

**Χειροκίνητα** (αν προτιμάς): *New* → *Web Service* → το ίδιο repo, και ίδιες ρυθμίσεις με το `render.yaml`:

| Ρύθμιση | Τιμή |
|---|---|
| Name | `euroleague-fantasy-api` |
| Region | Frankfurt |
| Branch | `main` |
| Runtime | Python 3 |
| Instance type | Free |
| Build Command | `pip install -r requirements.txt -c constraints.txt && pip install --no-deps .` |
| Start Command | `uvicorn elfantasy.api.main:app --host 0.0.0.0 --port $PORT --workers 1` |
| Health Check Path | `/health` |
| Auto-Deploy | **No** (off) |
| Environment | `PYTHON_VERSION=3.12.12`, `WEB_CONCURRENCY=1`, `MALLOC_ARENA_MAX=2`, `OMP_NUM_THREADS=1`, `MODEL_PATH=models/model.joblib`, και τα `DATABASE_URL`, `ADMIN_API_KEY` των βημάτων 2 και 3 |

Σημείωση: το Render ζητά τις τιμές των μεταβλητών με `sync: false` **μόνο κατά την αρχική δημιουργία** του Blueprint· αργότερα αλλάζουν από την καρτέλα *Environment* της υπηρεσίας (οι αλλαγές του `render.yaml` δεν τις αγγίζουν).

Το Render κάνει από μόνο του ένα πρώτο deploy όταν δημιουργηθεί η υπηρεσία. Αν η βάση δεν είναι προσβάσιμη (λάθος `DATABASE_URL`), η υπηρεσία δεν περνά το health check και το deploy δεν ολοκληρώνεται: διόρθωσε την τιμή και ξαναδοκίμασε (βήμα 6).

### Βήμα 2. `DATABASE_URL`: το session pooler του Supabase

Η τιμή είναι η συμβολοσειρά σύνδεσης του **session pooler** (πόρτα 5432), όχι του direct connection.

1. Στο dashboard του Supabase άνοιξε το project `euroleague-fantasy-api`.
2. Πάτα *Connect* (πάνω στο project) και διάλεξε *Session pooler* (*Project → Connect → Session pooler*). Εμφανίζεται συμβολοσειρά της μορφής `postgresql://postgres.PROJECT_REF:PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres` (στο dashboard ο κωδικός φαίνεται ως `[YOUR-PASSWORD]`). Ο χρήστης είναι `postgres.PROJECT_REF` (όχι απλώς `postgres`).
3. Αντικατάστησε το `[YOUR-PASSWORD]` με τον κωδικό της βάσης (αν τον ξέχασες: *Project Settings → Database → Reset database password*, και ενημέρωσε και το τοπικό σου `.env`).
4. Πρόσθεσε στο τέλος `?sslmode=require`.

**Γιατί session pooler και όχι direct.** Το direct connection του Supabase (`db.<project-ref>.supabase.co`) είναι **μόνο IPv6** στο free plan και το Render δεν είναι βέβαιο ότι φτάνει σε IPv6 προορισμούς. Το session pooler δουλεύει με **IPv4** και υποστηρίζει συνεδρίες, άρα και prepared statements (`docs/DATABASE.md`, ενότητα 6). Ο transaction pooler (πόρτα 6543) δεν χρειάζεται εδώ.

**Κωδικός με ειδικούς χαρακτήρες.** Οι `@ : / ? # %` και τα κενά πρέπει να γραφτούν με URL-encoding, αλλιώς το URL διαβάζεται λάθος:

```bash
python -c "from urllib.parse import quote; print(quote(input('κωδικός: '), safe=''))"
```

Η εντολή ζητά τον κωδικό διαδραστικά, ώστε να μη μείνει στο ιστορικό του shell. Η τελική τιμή είναι της μορφής `postgresql://postgres.PROJECT_REF:PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres?sslmode=require`, με τον κωδικό σε URL-encoding. Το URL γράφεται **μόνο** στο πεδίο του Render (και, αν θέλεις, στο τοπικό `.env`)· ποτέ σε αρχείο που γίνεται commit και ποτέ σε chat.

### Βήμα 3. `ADMIN_API_KEY`

Μεγάλη τυχαία τιμή, π.χ.:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Βάλ' την στο πεδίο `ADMIN_API_KEY` του Render και κράτησέ την σε password manager: την ίδια τιμή θα στέλνεις στο header `X-API-Key` για τα `POST /availability`, `DELETE /availability/{player_id}` και `POST /admin/refresh`. Αν μείνει κενή, τα endpoints αυτά απαντούν 503 (είναι κλειστά, ποτέ ανοιχτά).

### Βήμα 4. Το Deploy Hook ως GitHub secret

1. Στο Render: άνοιξε την υπηρεσία → καρτέλα **Settings** → ενότητα **Deploy Hook**. Αντίγραψε το URL (μοιάζει με `https://api.render.com/deploy/srv-…?key=…`). Το URL είναι **μυστικό**: όποιος το έχει μπορεί να ξεκινά deploys. Αν διαρρεύσει, πάτα *Regenerate Hook* και ξανακάνε το βήμα.
2. Βάλ' το στο GitHub **χωρίς να το επικολλήσεις σε chat**, με έναν από τους δύο τρόπους:
   * **Τερματικό** (`gh` συνδεδεμένο με τον λογαριασμό σου): `gh secret set RENDER_DEPLOY_HOOK_URL --repo tmartsoukos/EuroleagueFantasyAPI`. Η εντολή ζητά την τιμή διαδραστικά (δεν φαίνεται στην οθόνη και δεν μένει στο ιστορικό).
   * **Περιηγητής:** repo → *Settings* → *Secrets and variables* → *Actions* → *New repository secret*, όνομα `RENDER_DEPLOY_HOOK_URL`, τιμή το URL.

### Βήμα 5. Το repository variable `RENDER_SERVICE_URL`

Το δημόσιο URL της υπηρεσίας, που εμφανίζεται στο dashboard του Render (κάτω από το όνομα της υπηρεσίας), π.χ. `https://euroleague-fantasy-api.onrender.com`. Αν το όνομα ήταν πιασμένο, το Render προσθέτει επίθημα, γι' αυτό αντίγραψέ το από το dashboard. Δεν είναι μυστικό, άρα:

```bash
gh variable set RENDER_SERVICE_URL --repo tmartsoukos/EuroleagueFantasyAPI --body "https://euroleague-fantasy-api.onrender.com"
```

ή από τον περιηγητή: *Settings → Secrets and variables → Actions → Variables → New repository variable*. Χωρίς αυτή τη μεταβλητή το deploy γίνεται κανονικά αλλά ο smoke check παραλείπεται με μήνυμα.

### Βήμα 6. Πρώτο deploy και έλεγχος

1. Ξανατρέξε το τελευταίο run του workflow στο GitHub (*Actions* → το run του `main` → *Re-run all jobs*) ή κάνε ένα νέο push στο `main`. Τα secrets διαβάζονται όταν ξεκινά το job, άρα το re-run τα βλέπει.
2. Στο run: το job `deploy` πρέπει να δείξει «Render deploy triggered» (με το SHA και το id του deploy) και μετά «Post-deploy smoke check: passed». Αν το Render χτίζει ακόμη, ο smoke check περιμένει μέχρι 15 λεπτά.
3. Δοκίμασε χειροκίνητα (το πρώτο αίτημα μετά από αδράνεια παίρνει περίπου ένα λεπτό):

   ```bash
   curl -s https://euroleague-fantasy-api.onrender.com/health
   curl -s "https://euroleague-fantasy-api.onrender.com/rankings?limit=3"
   ```

   Το `/health` πρέπει να έχει `"status": "ok"` και `commit` ίσο με το SHA του `main`. Το Swagger είναι στο `/docs`.
4. Στο Render, καρτέλα *Logs*: δεν πρέπει να υπάρχουν σφάλματα βάσης ή μοντέλου.

### Μετά το πρώτο deploy: rollback

* **Στο Render:** υπηρεσία → **Deploys** → διάλεξε μια παλαιότερη επιτυχημένη έκδοση → **Rollback** → επιβεβαίωση. Η επαναφορά είναι άμεση (δεν ξαναχτίζει), αλλά το Render διατηρεί περιορισμένο αριθμό παλαιών builds και, όπως αναφέρει, η επαναφορά από το dashboard απενεργοποιεί τα αυτόματα deploys (εδώ είναι ήδη κλειστά). Δεν αλλάζει το `main`.
* **Στο git (συνιστάται για μόνιμη διόρθωση):** `git revert <κακό commit>` και push στο `main`. Το CI ξανατρέχει τους ελέγχους και κάνει deploy της διορθωμένης έκδοσης.

### Μετά το πρώτο deploy: επανεκπαίδευση και ξαναδημοσίευση του μοντέλου

1. Ενημέρωσε τα δεδομένα στην τοπική βάση και τρέξε `python -m elfantasy.model.train` (`docs/MODEL.md`, ενότητα 10). Γράφει `models/model.joblib` και `models/metrics.json` μόνο αν το honest MAE είναι κάτω από το threshold.
2. Αν άλλαξε η σεζόν test, ξαναδημιούργησε το fixture: `python -m elfantasy.model.holdout --out tests/fixtures/holdout_2025.parquet` (με το νέο όνομα).
3. Αν άλλαξαν εκδόσεις βιβλιοθηκών, ανανέωσε το `constraints.txt` (ενότητα 4).
4. Κάνε commit το νέο artifact, τα metrics, και ό,τι άλλαξε, και push στο `main`.
5. Το CI τρέχει το quality gate πάνω στο **νέο** committed μοντέλο· αν περάσει, το `deploy` ξεκινά και η υπηρεσία φορτώνει το νέο μοντέλο στην επανεκκίνηση.

### Μετά το πρώτο deploy: ανανέωση δεδομένων

Η υπηρεσία δεν βλέπει νέα δεδομένα μέχρι το `POST /admin/refresh` (`docs/API.md`, ενότητα 11). Από τη μηχανή σου, με το `DATABASE_URL` του Supabase μόνο στο περιβάλλον της διεργασίας:

```bash
# 1. νέα δεδομένα στη βάση του Supabase (η τρέχουσα σεζόν)
DATABASE_URL='postgresql://postgres.PROJECT_REF:PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres?sslmode=require' \
  .venv/Scripts/python.exe -m elfantasy.ingest.pipeline --update
# 2. η υπηρεσία ξαναδιαβάζει τη βάση
curl -s -X POST -H "X-API-Key: $ADMIN_API_KEY" https://euroleague-fantasy-api.onrender.com/admin/refresh
# 3. έλεγχος: το data_loaded_through πρέπει να ισούται με το database.latest_played_game_date
curl -s https://euroleague-fantasy-api.onrender.com/health
```

Ένα νέο deploy ή μια επανεκκίνηση επίσης διαβάζουν ξανά τη βάση.

## 6. Το free plan του Render

Από την τεκμηρίωση του Render (https://render.com/docs/free, https://render.com/docs/compute-plans, έλεγχος 2026-10-04):

| Θέμα | Τιμή |
|---|---|
| Πόροι | 512 MB RAM, 0,1 CPU, μία μόνο instance |
| Ώρες | 750 «Free instance hours» τον μήνα ανά workspace· αν εξαντληθούν, το Render αναστέλλει τις δωρεάν υπηρεσίες |
| Αδράνεια | Η υπηρεσία «κοιμάται» μετά από **15 λεπτά χωρίς εισερχόμενη κίνηση** |
| Εκκίνηση μετά από αδράνεια | «περίπου ένα λεπτό» |
| Αποθηκευτικός χώρος | Εφήμερο σύστημα αρχείων (χάνεται σε redeploy, restart, αδράνεια)· χωρίς persistent disk. Δεν επηρεάζει το project: η βάση είναι στο Supabase και το μοντέλο στο repo |
| Άλλα | Χωρίς SSH ή shell, χωρίς one-off jobs. Το Render γράφει ότι δεν προορίζεται για παραγωγικές εφαρμογές |

Ό,τι αφορά το project:

* **Cold start.** Το πρώτο αίτημα μετά την αδράνεια πληρώνει την εκκίνηση της Python: εισαγωγές βιβλιοθηκών, φόρτωση μοντέλου, ανάγνωση ~73.000 γραμμών από το Supabase και υπολογισμός των πρώτων προβλέψεων (τοπικά ~6 s με 12 πυρήνες). Με 0,1 CPU θα είναι πολύ μεγαλύτερο· **δεν έχει μετρηθεί**. Το port ανοίγει μόνο όταν τελειώσει η φόρτωση.
* **Health checks.** Το Render περιμένει έως 15 λεπτά να περάσει το health check μιας νέας έκδοσης (αλλιώς ακυρώνει το deploy και κρατά την παλιά) και κάθε απάντηση πρέπει να έρθει μέσα σε 5 s. Αν μια υπηρεσία αποτυγχάνει στα health checks για 60 s, το Render την επανεκκινεί. Το `/health` απαντά 503 όταν η βάση ή το μοντέλο δεν δουλεύουν, άρα μια υπηρεσία που έγινε degraded επανεκκινείται μόνη της (το API δεν ξαναδοκιμάζει σύνδεση χωρίς επανεκκίνηση, `docs/API.md` ενότητα 9).
* **Supabase.** Τα projects του free plan «παύουν» μετά από 1 εβδομάδα αδράνειας (`docs/DATABASE.md`, ενότητα 12). Αν παυθεί, το API ξεκινά degraded μέχρι να αποκατασταθεί το project από το dashboard του Supabase.
* **Δεν επιβεβαιώθηκε** από την τεκμηρίωση: αν τα health checks του Render μετρούν ως «εισερχόμενη κίνηση» που κρατά την υπηρεσία ξύπνια, και αν ένα αίτημα SQL από το Render αρκεί για να μη παυθεί το project του Supabase. Δεν επιβεβαιώθηκαν επίσης οι ώρες build που περιλαμβάνει το free plan (δεν αναφέρονται).

## 7. Μνήμη: μετρήσεις και βελτιώσεις

Το free plan έχει 512 MB. Μετρήθηκε το RSS της διεργασίας του API (ένας worker, πραγματική βάση 73.171 γραμμών, committed μοντέλο, SQLite) μετά το startup, μετά από `/rankings` και `/predict`, και μετά από 3 διαδοχικά `POST /admin/refresh`. Οι τιμές είναι σε MB· η κορυφή (peak) είναι το μέγιστο RSS με δειγματοληψία ανά 25 ms.

| Περιβάλλον | Μετά το startup | Μετά από refresh (σταθερή) | Κορυφή |
|---|---|---|---|
| Linux (Ubuntu 24.04 σε WSL1), Python 3.12, **πριν** | 247 | 319 έως 322 | **372** |
| Linux, Python 3.12, **μετά τον κώδικα** | 250 | 253 έως 257 | **299** |
| Linux, Python 3.12, μετά τον κώδικα και `MALLOC_ARENA_MAX=2` (όπως στο `render.yaml`) | **231** | **234 έως 241** | **298** |
| Windows 11, Python 3.13, πριν | 345 | 351 έως 355 | 481 (peak working set) |
| Windows 11, Python 3.13, μετά | 342 | 345 έως 350 | 416 |

Η μέτρηση σε Linux είναι η αντιπροσωπευτική για το Render. Στα Windows το RSS είναι υψηλότερο κατά ~100 MB λόγω του xgboost (η δημιουργία του πρώτου `Booster` προσθέτει 106 MB εκεί και 3 MB σε Linux). Η κορυφή στο Render αναμένεται κοντά στα 300 MB, με άνεση κάτω από τα 512 MB· **δεν μετρήθηκε στο ίδιο το Render** (το WSL1 δεν είναι πραγματικός πυρήνας Linux και το Render δεν ελέγχθηκε).

**Πού πήγαινε η μνήμη** (Linux, πριν): οι βιβλιοθήκες (numpy, pandas, xgboost, scipy, SQLAlchemy, FastAPI) ~170 MB· η ανάγνωση του ιστορικού με `pd.read_sql` δημιουργούσε προσωρινά **+101 MB** (όλες οι γραμμές γίνονται αντικείμενα Python πριν φτιαχτούν οι στήλες, ενώ το τελικό frame είναι 9 MB)· ο υπολογισμός των features δημιουργούσε προσωρινά **+110 έως +134 MB** (πολλαπλά αντίγραφα του frame των 73.000 γραμμών)· και η glibc κρατούσε στα arenas της τη μνήμη που ελευθερωνόταν (+72 MB μετά το πρώτο refresh).

**Τι άλλαξε** (καμία αλλαγή στα αποτελέσματα):

1. `features/load.py`: το `load_history` διαβάζει σε τμήματα των 5.000 γραμμών (`read_in_chunks`). Προσωρινή μνήμη: +101 → +28 MB, και ταχύτερα (0,93 → 0,71 s). Αν ένα ολόκληρο τμήμα είναι NULL σε μια στήλη (το pandas θα έδινε `object`), η στήλη επανέρχεται στον τύπο που θα είχε χωρίς τμήματα.
2. `features/build.py`: το τέλος της `build_features` ταξινομεί μόνο τις 4 στήλες-κλειδιά αντί ολόκληρου του frame, δημιουργεί το αποτέλεσμα με ένα μόνο αντίγραφο (αφαιρέθηκε το δεύτερο `.copy()` και το περιττό `astype`) και ελευθερώνει τα ενδιάμεσα (`del`). Προσωρινή μνήμη του υπολογισμού: +110 έως +134 → +60 έως +75 MB (το peak των αντικειμένων Python και numpy στο `tracemalloc`: 136 → 70 MB), και ταχύτερα (0,86 → 0,70 s).
3. `model/predict.py`: το `_compute` ελευθερώνει το frame των features μόλις πάρει τις γραμμές των επόμενων αγώνων.
4. `render.yaml`: `MALLOC_ARENA_MAX=2` (μόνιμο RSS μετά το refresh 253 → 234 MB) και `OMP_NUM_THREADS=1` (το free plan έχει 0,1 CPU· δεν άλλαξε η μνήμη ή η ταχύτητα στη μέτρηση, αποφεύγονται τα πολλά νήματα OpenMP στο όριο CPU).

**Απόδειξη ότι τα αποτελέσματα είναι ίδια.** Πριν και μετά τις αλλαγές αποτυπώθηκαν, με σταθερό `as_of`, τα frames του ιστορικού, των αγώνων, του προγράμματος και των παικτών (73.171 γραμμές), τα features (74.383 γραμμές, 54 στήλες) και οι προβλέψεις και των 1.212 παικτών (`predicted_fantasy`, `predicted_pir`, και τα 42 features, με `repr` των float). Σύγκριση με `assert_frame_equal(check_exact=True)` και ελέγχου των dtypes, και ψηφιακό αποτύπωμα sha256 ανά στήλη: **πανομοιότυπα**, και στα Windows και στο Linux (μέγιστη διαφορά 0). Στα tests: `tests/unit/test_load.py` συγκρίνει το `load_history` με την προηγούμενη υλοποίηση για κάθε μέγεθος τμήματος και στις οριακές περιπτώσεις (τμήμα όλο NULL, κενό αποτέλεσμα), και τα 65 tests των features και τα tests του `Predictor` περνούν αμετάβλητα.

## 8. Στάδια του CI που προσομοιώθηκαν

Δεν υπάρχει τρόπος να τρέξει το workflow τοπικά, γι' αυτό κάθε step εντολών προσομοιώθηκε σε καθαρό περιβάλλον Linux (Ubuntu 24.04, Python 3.12.3, νέο venv ανά job, WSL1 στο laptop της ανάπτυξης, 2026-10-04) με τις εντολές του `ci.yml`:

| Job | Αποτέλεσμα |
|---|---|
| `lint` | Εγκατάσταση του ruff 0.16.10 σε 3 s· `ruff check .` και `ruff format --check .` χωρίς σφάλματα (104 αρχεία) |
| `test` | Εγκατάσταση από το `constraints.txt` σε 91 s (με ζεστή cache του pip· 230 s με κρύα cache και αργή σύνδεση), 55 πακέτα ακριβώς όσα και οι καρφωμένες εκδόσεις, `pip check` χωρίς προβλήματα. `pytest --ignore=tests/quality` με Postgres στο `localhost:5432` και `ELFANTASY_REQUIRE_POSTGRES=1`, `ELFANTASY_PG_DISPOSABLE=1`: **1.949 passed, 1 skipped** σε 482 s (το WSL1 είναι αργό· οι runners του GitHub αναμένονται ταχύτεροι), και τα 42 tests του Postgres έτρεξαν. Η μόνη παράλειψη είναι το `test_clean.py` («no Greek locale», αναμενόμενη σε Linux). Coverage **99,04%** (όριο 95) |
| `quality-gate` | `pytest tests/quality`: 9 passed σε 2 s, MAE 5,9088 (ίδιο με τα Windows και με το `metrics.json`), και ο πίνακας της περίληψης |
| `coverage-badge` | Από το `99.04` προκύπτει badge «99%» |

Στα Windows (Python 3.13.2, το περιβάλλον ανάπτυξης) το πλήρες `pytest` έδωσε **1.974 passed** σε 7 λεπτά με coverage 99%. Σε καθαρό venv Python 3.12.7 στα Windows, με εγκατάσταση από το `constraints.txt`, πέρασαν 318 tests (quality gate, config, scripts, load, health και υγιεινή του repo). Επιπλέον:

* `actionlint` 1.7.12 (με shellcheck και pyflakes) και το επίσημο JSON schema των workflows του GitHub (`check-jsonschema --builtin-schema vendor.github-workflows`): κανένα σφάλμα.
* Το step του hook δοκιμάστηκε πάνω σε ψεύτικο Render (επιτυχία, αποτυχία 500 με 3 προσπάθειες, 500 δύο φορές και μετά επιτυχία, 401 χωρίς επανάληψη, hook με και χωρίς query string): το μυστικό δεν εμφανίζεται ποτέ στην έξοδο (εκτός από τις εντολές `::add-mask::`, που τις καταναλώνει ο runner).
* Το step του badge δοκιμάστηκε πάνω σε τοπικό bare git remote: δημιουργία του orphan branch, καθόλου commit όταν το SVG δεν αλλάζει, νέο commit όταν αλλάζει, ταυτότητα `github-actions[bot]`, το `main` αμετάβλητο.

## 9. Τι δεν έχει επαληθευτεί

Πριν το πρώτο πραγματικό run στο GitHub και το πρώτο deploy στο Render **δεν μπορούν** να επιβεβαιωθούν:

* **GitHub Actions:** η πραγματική συμπεριφορά των runners (χρόνος, cache του pip), το image `postgres:17` (η προσομοίωση χρησιμοποίησε PostgreSQL 18.4 ως τοπικό server), η σύνδεση των service containers, τα δικαιώματα `contents: write` του `GITHUB_TOKEN` για push στο `badges` (εξαρτάται από τη ρύθμιση *Settings → Actions → General → Workflow permissions* του repo· το job δηλώνει ρητά το δικαίωμα), η αυτόματη δημιουργία του environment `production`, η προσωρινή αποθήκευση (cache) του raw.githubusercontent.com για το badge (μέχρι λίγα λεπτά καθυστέρηση).
* **Render:** ότι το `render.yaml` γίνεται δεκτό από το dashboard (ελέγχθηκε μόνο με ανάγνωση της τεκμηρίωσης και του JSON schema που δημοσιεύει το Render, όχι με offline validator), ότι η `PYTHON_VERSION=3.12.12` είναι διαθέσιμη, ότι το Deploy Hook δέχεται την παράμετρο `ref` για υπηρεσία χωρίς αυτόματο deploy και ότι το `RENDER_GIT_COMMIT` ισούται με αυτό το `ref` (και οι δύο τεκμηριώνονται από το Render), η διάρκεια του build (η εγκατάσταση κατεβάζει ~490 MB wheels, από τα οποία τα 305 MB είναι το `nvidia-nccl-cu13`, εξάρτηση του xgboost για πολλαπλές GPU που η υπηρεσία δεν χρησιμοποιεί· το xgboost το ίδιο είναι 58 MB), ο χρόνος εκκίνησης με 0,1 CPU και η πραγματική κατανάλωση μνήμης.
* **Συνδυασμός Render και Supabase:** ότι το session pooler δουλεύει με το `sslmode=require` και τον ρόλο `postgres.<ref>` από το Render (δεν δοκιμάστηκε σε pooler, `docs/DATABASE.md`, ενότητα 13).

## 10. Πρόταση: προστασία του `main` (δεν εφαρμόζεται)

Αν θέλεις να μην μπαίνει στο `main` κώδικας που δεν πέρασε τους ελέγχους, στο repo: *Settings → Branches* (ή *Rules → Rulesets*) → κανόνας για το `main`:

* *Require status checks to pass before merging*, με required checks τα `lint`, `test` και `quality-gate` (τα ονόματα των jobs), και *Require branches to be up to date*.
* *Require a pull request before merging* (προαιρετικά με 0 εγκρίσεις, αν δουλεύεις μόνος).
* *Block force pushes* και *Restrict deletions*.

Προσοχή: με κανόνα που απαιτεί pull request, το `badges` δεν επηρεάζεται (είναι άλλο branch), αλλά ένα απευθείας push στο `main` θα απαιτεί bypass. Το `deploy` δεν πρέπει να είναι required check (τρέχει μόνο μετά το merge).

## 11. Συχνά προβλήματα

| Σύμπτωμα | Αιτία και λύση |
|---|---|
| Το `deploy` δείχνει «deploy skipped: secret not configured» | Δεν υπάρχει το secret `RENDER_DEPLOY_HOOK_URL` (βήμα 4). Δεν είναι σφάλμα |
| Το step του hook αποτυγχάνει με HTTP 401 | Λάθος ή αναγεννημένο hook: ξαναπάρε το URL από *Settings → Deploy Hook* και ξανάβαλ' το στο secret |
| 404 από το hook | Άγνωστη υπηρεσία ή το commit δεν υπάρχει στο repo που βλέπει το Render (πρέπει να έχει γίνει push) |
| 409 από το hook | Η υπηρεσία είναι ανεσταλμένη (π.χ. εξαντλήθηκαν οι ώρες του free plan) |
| Ο smoke check αποτυγχάνει με «runs commit …» | Το Render δεν ολοκλήρωσε το deploy μέσα σε 15 λεπτά. Δες το *Deploys* και τα *Logs* στο Render: συνήθως η νέα έκδοση δεν ξεκίνησε (π.χ. λάθος `DATABASE_URL`) και το Render κράτησε την παλιά |
| Ο smoke check αποτυγχάνει με `status is 'degraded'` | Τα `problems` της απάντησης λένε τι απέτυχε (βάση ή μοντέλο) |
| Το build στο Render αποτυγχάνει στο `pip install` | Έλεγξε ότι το `PYTHON_VERSION` είναι 3.12.x. Με 3.11 το `constraints.txt` δεν εγκαθίσταται |
| Το badge δεν εμφανίζεται | Το branch `badges` δημιουργείται στο πρώτο επιτυχημένο run στο `main`· το README το δείχνει μετά από λίγα λεπτά |
