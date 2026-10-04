# Βάση δεδομένων: SQLite τοπικά, Supabase Postgres στην παραγωγή (Φάση 5)

Το έγγραφο περιγράφει πώς αποθηκεύονται τα δεδομένα του project, πώς μεταφέρονται από την τοπική SQLite στο Supabase (Postgres), τι ισχύει για κάθε βάση και τι δοκιμάστηκε πραγματικά. Το σχήμα και ο ingestion βρίσκονται στο `docs/INGESTION.md`, το API στο `docs/API.md`.

## 1. Σύνοψη

| | SQLite (τοπικά, tests) | Postgres (Supabase, παραγωγή) |
|---|---|---|
| Ρύθμιση | `DATABASE_URL=sqlite:///data/elfantasy.db` (προεπιλογή) | `DATABASE_URL=postgresql://…` (η SQLAlchemy το μετατρέπει σε `postgresql+psycopg://…`) |
| Δημιουργία σχήματος | αυτόματα με `create_all` (ingestion, tests) | **μόνο** από τα migrations: `python -m elfantasy.db.migrate` |
| Ingestion (`--db` ή `DATABASE_URL`) | δημιουργεί τους πίνακες που λείπουν και γράφει με upsert | **δεν δημιουργεί ποτέ τίποτα**: ελέγχει ότι οι πίνακες υπάρχουν και γράφει με upsert |
| Startup του API | δημιουργεί τον πίνακα `player_availability` αν λείπει | ελέγχει ότι υπάρχει· αν λείπει, η υπηρεσία γίνεται degraded και **δεν τον δημιουργεί** |
| Row Level Security | δεν υπάρχει | ενεργό σε όλους τους πίνακες, χωρίς policies (migration `002`) |
| Tests και CI | **πάντα** (offline, γρήγορα) | μόνο σε ΤΟΠΙΚΟ server, με το marker `postgres` (ενότητα 9) |

Αρχεία της Φάσης 5:

| Αρχείο | Ρόλος |
|---|---|
| `src/elfantasy/db/models.py` | Σχήμα (SQLAlchemy Core): μία πηγή αλήθειας για SQLite και Postgres |
| `src/elfantasy/db/migrations/001_init.sql` | DDL του πλήρους σχήματος, **παραγόμενο** από το `models.py` |
| `src/elfantasy/db/migrations/002_row_level_security.sql` | RLS και αφαίρεση δικαιωμάτων των ρόλων του REST API |
| `src/elfantasy/db/migrate.py` | Runner των migrations (CLI και βιβλιοθήκη) |
| `src/elfantasy/db/transfer.py` | Μεταφορά δεδομένων SQLite → Postgres και επαλήθευση (CLI) |
| `src/elfantasy/db/session.py` | `get_engine` (ρυθμίσεις Postgres/pooler), `ensure_schema`, `upsert` |
| `src/elfantasy/db/urls.py` | Κανονικοποίηση URL, κρυμμένοι κωδικοί, σύγκριση βάσεων |
| `src/elfantasy/db/cli.py` | Κοινά βοηθητικά των εργαλείων (logging κονσόλας, σφάλματα χωρίς κωδικούς) |
| `src/elfantasy/model/record_predictions.py` | Καταγραφή των προβλέψεων στον πίνακα `predictions` (CLI) |
| `src/elfantasy/model/evaluate_recorded.py` | Σύγκριση των καταγεγραμμένων προβλέψεων με τα αποτελέσματα (CLI) |

## 2. Σχήμα

Έξι πίνακες της εφαρμογής (με τη σειρά των foreign keys: `models.TABLE_ORDER`) και ο πίνακας καταγραφής των migrations. Στο Postgres οι τύποι είναι `text`, `integer`, `double precision`, `boolean`, `date`, `timestamp` και `timestamptz`.

| Πίνακας | Κλειδί | Σημειώσεις |
|---|---|---|
| `teams` | `team_code` | 32 γραμμές στα τωρινά δεδομένα |
| `players` | `player_id` | `first_season`, `last_season` |
| `games` | `(season, gamecode)` | `game_date date`, `tipoff_utc timestamp` (UTC, χωρίς ζώνη), `played boolean`, FK `home_code`, `away_code`, `winner_code` → `teams` |
| `player_games` | `(season, gamecode, player_id)` | στατιστικά ανά αγώνα· `minutes`, `plus_minus`, `fantasy_score` είναι `double precision`· FK προς `games`, `players`, `teams` |
| `predictions` | `id serial`, και `UNIQUE (player_id, season, gamecode, model_version, as_of)` | καταγεγραμμένες προβλέψεις· FK προς `players` και `games(season, gamecode)`· `created_at timestamptz default now()` |
| `player_availability` | `player_id` | `CHECK (status IN ('out','doubtful','available'))`· `updated_at timestamptz` |
| `schema_migrations` | `version` | δημιουργείται από τον runner (όχι από το `001`): `version text primary key`, `applied_at timestamptz not null default now()`, `checksum text not null` |

**Αλλαγές σε σχέση με τη Φάση 2** (στο `models.py`): οι τύποι `String` και `Float` έγιναν `Text` και `Double` (στο Postgres: `text` αντί για `varchar` και ρητό `double precision` αντί για `float`), και ο πίνακας `predictions` απέκτησε τη στήλη `as_of`, υποχρεωτικά `season` και `gamecode` με foreign key προς το `games`, και το unique constraint. Οι παλιές τοπικές βάσεις SQLite που έχουν έναν **κενό** πίνακα `predictions` της παλιάς διάταξης (όλες οι βάσεις των Φάσεων 2 έως 4) τον αντικαθιστούν αυτόματα στην πρώτη εκτέλεση του ingestion ή του `record_predictions` (`ensure_schema`)· αν ο παλιός πίνακας έχει γραμμές, η εντολή σταματά με σαφές μήνυμα και δεν αγγίζει τίποτα. Οι υπόλοιποι πίνακες της παλιάς βάσης (με `VARCHAR`/`FLOAT`) είναι συμβατοί και μεταφέρονται κανονικά.

**Ονόματα** constraints και indexes: από τη naming convention του `models.py` (π.χ. `pk_games`, `fk_player_games_player_id_players`, `ix_games_played_game_date`, `ck_player_availability_status`). Όλα χωράνε στο όριο των 63 bytes του Postgres (το Postgres κόβει σιωπηλά μεγαλύτερα ονόματα· ένα test το ελέγχει).

**Ζώνες ώρας.** Το `tipoff_utc` είναι naive UTC (`timestamp`). Οι στήλες `created_at` και `updated_at` είναι `timestamptz`: ο κώδικας γράφει πάντα ώρες UTC με ρητή ζώνη και η μεταφορά δεδομένων κάνει ρητά UTC τις naive ώρες της SQLite, ώστε το αποτέλεσμα να μην εξαρτάται από τη ζώνη ώρας της συνεδρίας του server (δοκιμάστηκε με συνεδρία `Asia/Tokyo` και με τον server τοπικά σε `Europe/Bucharest`).

## 3. Migrations

```bash
python -m elfantasy.db.migrate --status      # ποια migrations έχουν εφαρμοστεί ή εκκρεμούν (δεν γράφει τίποτα)
python -m elfantasy.db.migrate --dry-run     # τι θα εφαρμοζόταν (δεν γράφει τίποτα, ούτε δημιουργεί πίνακες)
python -m elfantasy.db.migrate               # εφαρμογή των εκκρεμών migrations
python -m elfantasy.db.migrate --print-sql   # το DDL του 001_init.sql, παραγόμενο από το db/models.py
```

Το URL ορίζεται με `--db` ή (προτιμότερο, γιατί δεν μένει στο ιστορικό του shell) με το `DATABASE_URL`.

Κανόνες του runner:

* Τα αρχεία είναι `NNN_όνομα.sql` (UTF-8 χωρίς BOM, αλλαγές γραμμής LF) και εφαρμόζονται με αύξουσα σειρά του NNN. Άκυρο όνομα, BOM, διπλός αριθμός ή κενό αρχείο είναι σφάλμα.
* **Κάθε migration τρέχει σε μία συναλλαγή** μαζί με την εγγραφή του στο `schema_migrations`. Το Postgres έχει transactional DDL: ένα migration που αποτυγχάνει στη μέση δεν αφήνει τίποτα (δοκιμάστηκε: ούτε ο πίνακας που δημιουργήθηκε πριν από το σφάλμα). Το σφάλμα αναφέρει το migration και τον αριθμό του statement.
* **Idempotent:** ό,τι έχει εφαρμοστεί παραλείπεται.
* **Προστασία με checksum (SHA-256, ανεξάρτητο από CRLF/LF).** Αν ένα εφαρμοσμένο migration έχει αλλάξει ως αρχείο, αν λείπει το αρχείο ενός εφαρμοσμένου migration ή αν εκκρεμεί migration με αριθμό μικρότερο από ήδη εφαρμοσμένο, **δεν εφαρμόζεται τίποτα** και η εντολή αποτυγχάνει (κωδικός εξόδου 1). Ένα εφαρμοσμένο migration δεν επεξεργάζεται ποτέ· η διόρθωση γίνεται με νέο migration.
* Δύο ταυτόχρονα τρεξίματα δεν συγκρούονται: κάθε migration (και η δημιουργία του `schema_migrations`) παίρνει `pg_advisory_xact_lock`, ένα lock που ισχύει μέχρι το τέλος της συναλλαγής, άρα δεν εξαρτάται από τη συνεδρία και δεν υπάρχει λόγος να μη δουλεύει και στον transaction pooler (δεν δοκιμάστηκε σε pooler). Δοκιμάστηκε με δύο νήματα σε πραγματικό Postgres.
* Δουλεύει **μόνο σε Postgres**: με URL SQLite η εντολή τελειώνει με κωδικό 2 και σαφές μήνυμα (στην SQLite το σχήμα δημιουργείται με `create_all`).
* Ο πίνακας `schema_migrations` ενεργοποιεί RLS αμέσως μετά τη δημιουργία του (και το `002` το ξαναεπιβεβαιώνει).

**Κωδικοί εξόδου:** 0 επιτυχία, 1 αποτυχία ή ασυνέπεια, 2 άκυρο URL ή URL που δεν είναι Postgres.

### Πώς προστίθεται νέο migration

1. Άλλαξε το `models.py` (αν αλλάζει το σχήμα) και γράψε το νέο αρχείο `003_όνομα.sql` με το `ALTER TABLE` κ.λπ. Ο αριθμός είναι ο επόμενος, χωρίς κενά που να σπάνε τη σειρά.
2. Το test `test_001_equals_the_ddl_generated_from_the_models` (drift test) θα αποτύχει, γιατί το `001_init.sql` δεν ισούται πια με το σχήμα του `models.py`. Αυτό είναι σκόπιμο: **μην ξαναγράψεις το `001`** (έχει εφαρμοστεί και έχει checksum). Ενημέρωσε το test ώστε να συγκρίνει το `001` με το σχήμα της εποχής του, και βασίσου στο test `test_the_schema_matches_the_models` (πραγματικό Postgres), που ελέγχει ότι το αποτέλεσμα **όλων** των migrations ισούται με το `models.py`.
3. Ενεργοποίησε RLS στους νέους πίνακες (και `REVOKE`, όπως στο `002`) μέσα στο ίδιο migration.

## 4. Μεταφορά δεδομένων SQLite → Postgres

```bash
python -m elfantasy.db.transfer --from sqlite:///data/elfantasy.db [--to URL] [--tables …] [--batch-size 2000] [--dry-run | --verify-only] [--no-row-digest]
```

* Προορισμός: το `--to` ή το `DATABASE_URL`. Πρέπει να έχει ήδη το σχήμα (migrations), αλλιώς η εντολή σταματά πριν γράψει οτιδήποτε και δείχνει την εντολή `migrate`.
* Σειρά: `teams`, `players`, `games`, `player_games`, `predictions`, `player_availability` (foreign keys). Πίνακες που λείπουν ή είναι κενοί στην πηγή παραλείπονται (π.χ. το κενό `predictions` της παλιάς διάταξης).
* **Idempotent upsert σε παρτίδες** (μία συναλλαγή ανά παρτίδα, προεπιλογή 2.000 γραμμές): μια διακοπή ή επανάληψη δεν διπλασιάζει γραμμές και συνεχίζει με ασφάλεια. Το τεχνικό `predictions.id` δεν αντιγράφεται (ο προορισμός δίνει δικά του)· η σύγκρουση ορίζεται από το unique constraint των φυσικών στηλών.
* **Η πηγή ανοίγει μόνο για ανάγνωση** (`mode=ro`) και διαβάζεται σε streaming (δεν φορτώνεται ολόκληρη στη μνήμη). Η εντολή δεν μπορεί να την αλλάξει: στο πραγματικό τρέξιμο το md5 του `data/elfantasy.db` ήταν ίδιο πριν και μετά.
* **Προστασίες:** αρνείται να γράψει στην ίδια βάση που διαβάζει (σύγκριση host, πόρτας, βάσης και χρήστη· δεν μπορεί να εντοπίσει δύο διαφορετικά URL της ίδιας βάσης, π.χ. direct και pooler του Supabase) και προορισμό που δεν είναι Postgres, εκτός αν δοθεί `--allow-non-postgres` (για tests και για αντίγραφα ασφαλείας σε SQLite, ενότητα 11). Αρνείται πηγή SQLite που δεν υπάρχει (δεν τη δημιουργεί).
* **Τύποι:** το 0/1 της SQLite γίνεται boolean, οι ημερομηνίες `date`, οι ώρες UTC (ρητά, βλ. ενότητα 2), τα NULL μένουν NULL.

### Επαλήθευση

Στο τέλος (και μόνη της με `--verify-only`) τυπώνεται πίνακας συγκρίσεων πηγής και προορισμού, και ο κωδικός εξόδου είναι ≠ 0 σε οποιαδήποτε ασυμφωνία:

* πλήθος γραμμών ανά πίνακα·
* aggregates που υπολογίζονται από τη βάση: `sum(pir)`, `sum(valuation)`, `sum(fantasy_score)`, `sum(minutes)`, γραμμές DNP, `min`/`max(game_date)`, παιγμένοι αγώνες, `sum(home_score)`, `sum(away_score)`, **`sum(points)` ανά σεζόν** και, για τις προβλέψεις και τη διαθεσιμότητα, τα αντίστοιχα·
* **ψηφιακό αποτύπωμα του περιεχομένου** κάθε πίνακα (`πλήθος:hash`), υπολογισμένο στην Python και στις δύο βάσεις πάνω σε κανονικοποιημένες τιμές. Είναι ανεξάρτητο από τη σειρά των γραμμών και από το collation (που διαφέρουν ανάμεσα σε SQLite και Postgres) και πιάνει οποιαδήποτε διαφορά σε οποιαδήποτε στήλη, ακόμη και πολύ μικρότερη από την ανοχή των aggregates. Με `--no-row-digest` παραλείπεται (γρηγορότερο, ασθενέστερος έλεγχος).

Τα ποσά `double precision` συγκρίνονται με ανοχή (σχετική 1e-9, απόλυτη 1e-6), γιατί η σειρά άθροισης διαφέρει ανάμεσα στις βάσεις.

**Κωδικοί εξόδου:** 0 επιτυχία (και επαλήθευση ΟΚ), 1 αποτυχία ή ασυμφωνία, 2 άρνηση (άκυρο URL, ίδια βάση, προορισμός που δεν είναι Postgres, πηγή που δεν υπάρχει, άκυρες επιλογές).

## 5. Μετάβαση βήμα-βήμα

Οι εντολές τρέχουν από τη ρίζα του repo. Στα ελληνικά Windows χρειάζονται `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`. Το URL μπαίνει στο **περιβάλλον της διεργασίας** (ή σε αρχείο `.env`, που αγνοεί το git)· **ποτέ** σε αρχείο που γίνεται commit.

```bash
# Git Bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export DATABASE_URL='postgresql://postgres.PROJECT_REF:PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres?sslmode=require'
PY=.venv/Scripts/python.exe

$PY -m elfantasy.db.migrate --status        # 1. σύνδεση και κατάσταση: 001 και 002 σε εκκρεμότητα
$PY -m elfantasy.db.migrate --dry-run       # 2. τι θα εφαρμοστεί
$PY -m elfantasy.db.migrate                 # 3. εφαρμογή (δεύτερη εκτέλεση: «nothing (up to date)»)
$PY -m elfantasy.db.transfer --from sqlite:///data/elfantasy.db --dry-run   # 4. σχέδιο μεταφοράς
$PY -m elfantasy.db.transfer --from sqlite:///data/elfantasy.db             # 5. μεταφορά + επαλήθευση
$PY -m elfantasy.db.transfer --from sqlite:///data/elfantasy.db --verify-only   # 6. (προαιρετικά) νέα επαλήθευση
$PY -m elfantasy.ingest.verify --accept-missing 2018/21                     # 7. έλεγχοι ποιότητας πάνω στο Postgres
```

```powershell
# PowerShell
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
$env:DATABASE_URL = 'postgresql://postgres.PROJECT_REF:PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres?sslmode=require'
.venv\Scripts\python.exe -m elfantasy.db.migrate --status
```

Μετά τη μεταφορά, το API και το ingestion δουλεύουν με το ίδιο `DATABASE_URL` (ενότητα 8). Η τοπική `data/elfantasy.db` **δεν αλλάζει ποτέ**: η επιστροφή στην SQLite είναι απλώς `DATABASE_URL=sqlite:///data/elfantasy.db` (ή αφαίρεση της μεταβλητής).

Αν η μεταφορά διακοπεί (δίκτυο, παύση της βάσης), ξανατρέξε την ίδια εντολή: συνεχίζει με ασφάλεια.

Αν χρησιμοποιείς αρχείο ρυθμίσεων αντί για μεταβλητή περιβάλλοντος, πρέπει να λέγεται ακριβώς **`.env`** (ο Notepad προσθέτει συχνά `.txt`): το `Settings` δεν διαβάζει άλλο όνομα. Το `.gitignore` αγνοεί κάθε `.env*` εκτός από το `.env.example`.

## 6. Connection strings του Supabase

Από την τεκμηρίωση του Supabase (https://supabase.com/docs/guides/database/connecting-to-postgres, ελέγχθηκε στις 2026-10-04):

| Μέθοδος | Host | Πόρτα | Χρήστης | IPv4 / IPv6 |
|---|---|---|---|---|
| Direct connection | `db.<project-ref>.supabase.co` | 5432 | `postgres` | IPv6 (IPv4 μόνο με πρόσθετο add-on) |
| **Session pooler** (Supavisor) | `aws-<n>-<region>.pooler.supabase.com` | **5432** | `postgres.<project-ref>` | IPv4 (όλα τα plans) |
| Transaction pooler (Supavisor) | `aws-<n>-<region>.pooler.supabase.com` | 6543 | `postgres.<project-ref>` | IPv4 (όλα τα plans) |

**Επιλογή για αυτό το project: session pooler, πόρτα 5432.** Δουλεύει με IPv4 (το direct connection χρειάζεται IPv6) και υποστηρίζει συνεδρίες, άρα και prepared statements. Το URL το δίνει το dashboard: *Project → Connect → Session pooler*. Ο transaction pooler (6543) προορίζεται για serverless περιβάλλοντα με πολλές μικρές συνδέσεις και **δεν υποστηρίζει prepared statements**: το `get_engine` ορίζει αυτόματα `prepare_threshold=None` όταν η πόρτα είναι 6543, ή όταν το URL έχει την παράμετρο `prepare_threshold=none` (ή ακέραιο, π.χ. `prepare_threshold=0`)· η παράμετρος αφαιρείται από το URL πριν φτάσει στον driver. Το direct connection είναι κατάλληλο για migrations και αντίγραφα όταν υπάρχει IPv6.

**SSL.** Το Supabase υποστηρίζει SSL και η τεκμηρίωσή του συστήνει να ορίζεται `sslmode=require`, ώστε ο driver να αρνείται σύνδεση χωρίς κρυπτογράφηση (https://supabase.com/docs/guides/platform/ssl-enforcement). Βάλε `?sslmode=require` στο URL: το `get_engine` το περνά αυτούσιο στο libpq (δοκιμάστηκε ότι σε server χωρίς SSL η σύνδεση αποτυγχάνει με «server does not support SSL, but SSL was required»). Το project έχει και ρύθμιση **«Enforce SSL on incoming connections»** (Database Settings): όταν είναι ενεργή, οι συνδέσεις χωρίς SSL απορρίπτονται· η προεπιλογή της δεν αναφέρεται στην τεκμηρίωση και η αλλαγή της προκαλεί επανεκκίνηση της βάσης με σύντομη διακοπή. Η `require` κρυπτογραφεί χωρίς να επαληθεύει το πιστοποιητικό του server· η `verify-full` το επαληθεύει και συστήνεται από το Supabase, αλλά χρειάζεται το πιστοποιητικό CA του project (από το dashboard, ενότητα SSL Configuration) και δεν χρησιμοποιείται εδώ. Ο **τοπικός** server των tests δεν έχει SSL: μην βάλεις `require` σε τοπικό URL.

**Κωδικός με ειδικούς χαρακτήρες.** Οι χαρακτήρες `@ : / ? # %` και τα κενά πρέπει να γραφτούν με URL-encoding, αλλιώς το URL διαβάζεται λάθος:

```bash
.venv/Scripts/python.exe -c "from urllib.parse import quote; print(quote('p@ss/w:rd#1', safe=''))"
# p%40ss%2Fw%3Ard%231
```

**Driver.** Τα URL της μορφής `postgresql://…` και `postgres://…` (όπως τα δίνει το Supabase) μετατρέπονται σε `postgresql+psycopg://…` (driver `psycopg` 3, εγκατεστημένο μέσω `psycopg[binary]`). Μπορείς να γράψεις και ρητά `postgresql+psycopg://…`.

**Ρυθμίσεις του engine** (`get_engine`, μόνο για Postgres· η SQLite δεν αλλάζει): `pool_pre_ping=True` (κάθε σύνδεση ελέγχεται πριν τη χρήση, ώστε μια σύνδεση που έκλεισε ο pooler ή ο firewall να αντικαθίσταται αντί να αποτύχει το αίτημα· δοκιμάστηκε σκοτώνοντας τη σύνδεση από τον server), `pool_size=5`, `max_overflow=5` (έως 10 συνδέσεις ανά διεργασία), `pool_recycle=300` s και `connect_timeout=10` s (αν το URL δεν ορίζει δικό του: μια παυμένη βάση δεν κρατά το startup «κολλημένο»). Σε **κάθε νέα σύνδεση** εκτελείται `SET extra_float_digits = 3` (με commit): ο server του Supabase έχει προεπιλογή 0 και στέλνει τα `double precision` με 15 σημαντικά ψηφία, οπότε χωρίς το SET τα floats διαβάζονται αλλοιωμένα (19.983333333333334 → 19.9833333333333). Η εγγραφή ήταν πάντα σωστή· το ζήτημα φάνηκε στην επαλήθευση της μεταφοράς (ενότητα 13). Στον transaction pooler η ρύθμιση συνεδρίας δεν είναι εγγυημένη. Η δημιουργία του engine δεν συνδέεται στη βάση.

**Κωδικοί σε logs.** Κάθε URL που εμφανίζεται σε log, μήνυμα ή σφάλμα περνά από `safe_url` (κωδικός `***`, και στις παραμέτρους του query `password`, `sslpassword` και `passphrase`, π.χ. `?password=...`) και κάθε μήνυμα εξαίρεσης από `redact_secrets` (ο κωδικός αντικαθίσταται σε οποιοδήποτε κείμενο, με και χωρίς percent-encoding). Τα σφάλματα ανάγνωσης URL δεν περιέχουν ποτέ το URL. Το ingestion γράφει στο `data/logs/ingest.log` τα ορίσματα της γραμμής εντολών με το `--db` κρυμμένο. Τα tests ελέγχουν όλα τα παραπάνω (και ότι κανένα αρχείο του repo δεν περιέχει πραγματικό κωδικό ή κλειδί).

## 7. Row Level Security και ρόλος `postgres`

Το Supabase εκθέτει τους πίνακες του schema `public` στο REST API του (PostgREST) με τους ρόλους `anon` και `authenticated`. Σύμφωνα με την τεκμηρίωση, ένας πίνακας χωρίς RLS είναι αναγνώσιμος και εγγράψιμος από κάθε ρόλο που έχει δικαίωμα πάνω του, και ένας νέος πίνακας στο `public` μπορεί να έχει ήδη δικαιώματα για τους τρεις ρόλους (εξαρτάται από τις default privileges του project). Η εφαρμογή δεν χρησιμοποιεί ποτέ το REST API: συνδέεται απευθείας στη βάση. Άρα κανένας πίνακας δεν πρέπει να φαίνεται από εκεί.

Το migration `002_row_level_security.sql`:

1. `ALTER TABLE … ENABLE ROW LEVEL SECURITY` σε **όλους** τους πίνακες (και στον `schema_migrations`), **χωρίς policies**: χωρίς policy ο ρόλος που υπόκειται σε RLS δεν διαβάζει και δεν αλλάζει καμία γραμμή (deny-all). Δεν χρησιμοποιείται `FORCE ROW LEVEL SECURITY` (θα έκλεινε και τον ιδιοκτήτη).
2. Defense in depth: `REVOKE ALL` στους πίνακες και στην ακολουθία `predictions_id_seq` για τους `anon` και `authenticated`, **μόνο αν οι ρόλοι υπάρχουν** (έλεγχος στο `pg_roles`), ώστε το ίδιο migration να τρέχει και σε απλό Postgres χωρίς Supabase.

Ο ρόλος `postgres` της εφαρμογής είναι ο ιδιοκτήτης των πινάκων και, σύμφωνα με το Supabase (https://supabase.com/docs/guides/database/postgres/row-level-security), έχει το χαρακτηριστικό `bypassrls`: παρακάμπτει το RLS και η εφαρμογή (API, ingestion) δουλεύει κανονικά. Το ίδιο ισχύει για τον `service_role`. Αν συνδεθείς με άλλον ρόλο, χρειάζεται `BYPASSRLS` ή ρητές policies.

**Τι δοκιμάστηκε** (σε πραγματικό Postgres, με ρόλους `anon`/`authenticated` και default privileges όπως του Supabase): μετά τα migrations κανένας από τους δύο ρόλους δεν έχει κανένα δικαίωμα σε κανέναν πίνακα ούτε στην ακολουθία· αν κάποιος ξαναδώσει `SELECT`, ο ρόλος βλέπει 0 γραμμές (RLS)· χωρίς το `SELECT` παίρνει «permission denied»· ο ιδιοκτήτης βλέπει κανονικά τα δεδομένα.

**Στο dashboard του Supabase** (δεν ελέγχθηκε εδώ): το Security Advisor δεν πρέπει να δείχνει «RLS disabled in public». Η πληροφοριακή ένδειξη «RLS enabled, no policy» είναι **αναμενόμενη και σκόπιμη** (deny-all). Αν δεν χρησιμοποιείς καθόλου το Data API, μπορείς επιπλέον να το απενεργοποιήσεις από τις ρυθμίσεις του project.

## 8. Ingestion, API και `/admin/refresh` προς Postgres

**Ingestion.** `python -m elfantasy.ingest.pipeline` γράφει όπου δείχνει το `DATABASE_URL` (ή το `--db`). Σε Postgres:

* **δεν εκτελεί `create_all`** και δεν δημιουργεί ποτέ πίνακες: καλεί το `ensure_schema`, που ελέγχει μόνο (από τους καταλόγους του συστήματος) ότι υπάρχουν όλοι οι πίνακες. Αν λείπει κάποιος, η εντολή αποτυγχάνει πριν από κάθε εγγραφή με μήνυμα «database tables are missing: …; apply the migrations with: python -m elfantasy.db.migrate» (κωδικός εξόδου 1). Ο λόγος είναι ασφάλεια: ένας πίνακας που δημιουργείται εκτός migrations δεν έχει RLS. Το ίδιο ισχύει αν η βάση έχει πίνακες αλλά όχι `schema_migrations` (αποδεκτό· δεν γίνεται καμία δημιουργία).
* γράφει με τα ίδια upserts και κανόνες ενημέρωσης (ενότητα 5 του `docs/INGESTION.md`), σε μία συναλλαγή.
* το `--update` (μόνο η τρέχουσα σεζόν) είναι ο κανονικός τρόπος ανανέωσης: `DATABASE_URL=… python -m elfantasy.ingest.pipeline --update`, και μετά `POST /admin/refresh` στο API (ενότητα 11 του `docs/API.md`).
* το `python -m elfantasy.ingest.verify` δουλεύει αυτούσιο σε Postgres και δίνει την ίδια αναφορά με την SQLite (δοκιμάστηκε: ίδιο `season_summary`, 100% ταύτιση `pir == valuation`).

**API.** Το startup ελέγχει ότι ο πίνακας `player_availability` υπάρχει, χωρίς να δημιουργεί τίποτα και χωρίς να χρειάζεται δικαίωμα `CREATE` (μόνο ανάγνωση των καταλόγων). Αν η βάση δεν είναι προσβάσιμη ή δεν έχει σχήμα, η υπηρεσία ξεκινά **degraded** (`/health` 503, τα `/predict` και `/rankings` 503) και δεν δημιουργεί πίνακες· στο log γράφεται ο λόγος (χωρίς κωδικούς). Στην SQLite ο πίνακας δημιουργείται όπως πριν. Το API **δεν γράφει ποτέ** στον πίνακα `predictions` (τα GET αιτήματα δεν έχουν παρενέργειες: ένα test το ελέγχει).

Κάθε αίτημα `/predict` και `/rankings` διαβάζει τον μικρό πίνακα διαθεσιμότητας από τη βάση. Με το Supabase αυτό σημαίνει ένα ταξίδι δικτύου ανά αίτημα: ο χρόνος απόκρισης εξαρτάται από την απόσταση ανάμεσα στο Render και στη βάση. Η ανάγνωση του ιστορικού (startup και `/admin/refresh`) είναι ~73.000 γραμμές.

## 9. Tests και CI

Τα tests και το CI τρέχουν **πάντα σε SQLite**, offline. Επιπλέον:

* **Offline (SQLite):** drift test migration ↔ models, runner (σειρά, checksum, idempotency, ατομικότητα, εκτός σειράς, ταυτόχρονη εφαρμογή), μεταφορά SQLite → SQLite με `--allow-non-postgres` (πλήθη, checksums, επαναλήψεις, ασυμφωνίες, προστασίες, τύποι), `record_predictions`, `evaluate_recorded`, ρυθμίσεις `get_engine` (pooler, `prepare_threshold`, κρυμμένοι κωδικοί), CLI ως `python -m` (υποδιεργασίες), υγιεινή του repo.
* **Σε πραγματικό ΤΟΠΙΚΟ Postgres** (marker `postgres`, αρχείο `tests/integration/test_postgres.py`, 42 tests): migrations, σχήμα έναντι `models.py`, RLS και δικαιώματα, μεταφορά, ingestion, `Predictor`, API, καταγραφή και αξιολόγηση προβλέψεων, engine. Ο server είναι, με σειρά:
  1. το `TEST_DATABASE_URL`, αν οριστεί και δείχνει σε **τοπικό** host (`localhost`, `127.0.0.1`, `::1`, socket unix)· URL προς άλλον host **απορρίπτεται** (τα tests παραλείπονται): κανένα test δεν συνδέεται ποτέ σε remote βάση·
  2. ο ενσωματωμένος Postgres του πακέτου **`pixeltable-pgserver`** (`requirements-dev.txt`, ~30 MB, binaries μέσα στο wheel), που ξεκινά σε προσωρινό φάκελο και σταματά στο τέλος (τοπικά: PostgreSQL 18.4 σε Windows 11, εκκίνηση ~8 s στην πρώτη φορά).

  Αν δεν υπάρχει καμία πηγή, τα tests παραλείπονται· με `ELFANTASY_REQUIRE_POSTGRES=1` αποτυγχάνουν (χρήσιμο στο CI, για να μη χάνεται σιωπηλά η κάλυψη). Με την ίδια μεταβλητή αποτυγχάνει και κάθε test με marker `postgres` που παραλείπεται για οποιονδήποτε άλλον λόγο (plugin `ForbidSilentPostgresSkips` στο `tests/pg_support.py`). Κάθε test παίρνει δική του κενή βάση (`CREATE DATABASE` και διαγραφή στο τέλος). Εκτός: `pytest -m "not postgres"`.

  **Ρόλοι σε επίπεδο server.** Ένα test (`test_the_supabase_roles_lose_all_access`) μιμείται τους ρόλους `anon` και `authenticated` του Supabase και τους δημιουργεί και τους διαγράφει στο cluster. Τρέχει μόνο σε **προσωρινό** server: τον ενσωματωμένο, ή τον server του `TEST_DATABASE_URL` όταν δηλωθεί ρητά με `ELFANTASY_PG_DISPOSABLE=1`. Το ορίζει το CI (service container `postgres:17`, `docs/DEPLOY.md`), ώστε να τρέχουν και τα 42 tests· σε οποιονδήποτε άλλον server το test παραλείπεται, και μην το ορίσεις ποτέ σε server που χρησιμοποιείς για άλλη δουλειά.

## 10. Καταγραφή και αξιολόγηση προβλέψεων

```bash
python -m elfantasy.model.record_predictions [--db URL] [--as-of YYYY-MM-DD] [--active-only | --all] [--model PATH]
python -m elfantasy.model.evaluate_recorded  [--db URL] [--model-version V] [--since YYYY-MM-DD] [--until YYYY-MM-DD]
```

**`record_predictions`** καλεί το `Predictor.predict_all` και γράφει στον πίνακα `predictions` μία γραμμή ανά παίκτη με προγραμματισμένο επόμενο αγώνα: `predicted_fantasy`, `predicted_pir`, `model_version` και `as_of`.

* Κλειδί της γραμμής: `(player_id, season, gamecode, model_version, as_of)`. Η εγγραφή είναι upsert: **το ξανατρέξιμο της ίδιας ημέρας δεν διπλασιάζει γραμμές** (ενημερώνει τις τιμές και το `created_at`). Γραμμές δεν διαγράφονται ποτέ. Αν ο επόμενος αγώνας ενός παίκτη αλλάξει την ίδια ημέρα (αναβολή), προστίθεται και δεύτερη γραμμή για τον νέο αγώνα.
* Παίκτες χωρίς επόμενο αγώνα (offseason, ομάδα χωρίς άλλους αγώνες) **δεν καταγράφονται**: δεν υπάρχει αγώνας με τον οποίο να συγκριθεί η πρόβλεψη. Προβλέψεις με μη πεπερασμένη τιμή παραλείπονται.
* Το `as_of` είναι η ημερομηνία από την οποία ο `Predictor` ψάχνει τον επόμενο αγώνα κάθε ομάδας (προεπιλογή: σήμερα, UTC). **Δεν είναι «ταξίδι στο παρελθόν»**: το ιστορικό που χρησιμοποιεί το μοντέλο είναι πάντα ό,τι υπάρχει τώρα στη βάση.
* Προεπιλογή `--active-only` (οι ενεργοί παίκτες του `Predictor`)· το `--all` γράφει όλους τους γνωστούς παίκτες με επόμενο αγώνα.
* Το `INSERT … ON CONFLICT` του Postgres καταναλώνει τιμές της ακολουθίας `predictions_id_seq` ακόμη και για γραμμές που ενημερώνονται: τα `id` έχουν κενά (στο πραγματικό τρέξιμο: 992 γραμμές με `id` έως 1.516), δεν επαναλαμβάνονται όμως ποτέ και δεν επηρεάζουν τίποτα.

**`evaluate_recorded`** συγκρίνει τις καταγεγραμμένες προβλέψεις με τα πραγματικά αποτελέσματα όταν οι αγώνες παιχτούν και τυπώνει MAE ανά ημέρα αγώνα και συνολικά, χωριστά ανά έκδοση μοντέλου. Κάθε πρόβλεψη ανήκει σε μία κατηγορία:

| Κατηγορία | Σημασία | Μετρά στο MAE; |
|---|---|---|
| `pending` | ο αγώνας δεν έχει παιχτεί ακόμη | όχι |
| `appeared` | ο παίκτης αγωνίστηκε (γραμμή boxscore με `dnp = false` και λεπτά > 0) | **ναι** |
| `dnp` | ο αγώνας παίχτηκε και ο παίκτης έχει γραμμή DNP (πραγματική τιμή 0) | όχι (αναφέρεται) |
| `not_in_boxscore` | ο αγώνας παίχτηκε αλλά ο παίκτης δεν υπάρχει στο boxscore | όχι (αναφέρεται) |

Το MAE και το `bias` (μέσος όρος του `predicted − actual`, θετικό = υπερεκτίμηση) υπολογίζονται **μόνο** για όσους αγωνίστηκαν, γιατί το μοντέλο προβλέπει υπό την προϋπόθεση ότι ο παίκτης αγωνίζεται (`docs/MODEL.md`)· οι κατηγορίες `dnp` και `not_in_boxscore` φαίνονται πάντα ρητά. Το `mae_incl_dnp` είναι ενημερωτική μετρική (οι γραμμές DNP μετρούν με πραγματική τιμή 0). Αν κανένας καταγεγραμμένος αγώνας δεν έχει παιχτεί ακόμη, η εντολή το λέει και τελειώνει με κωδικό 0· αν ο πίνακας `predictions` δεν υπάρχει, με κωδικό 1.

## 11. Rollback και ανάκτηση

| Περίπτωση | Τι γίνεται |
|---|---|
| Αποτυχία migration | Η συναλλαγή του ακυρώνεται: δεν μένει τίποτα και δεν καταγράφεται. Διόρθωσε το αρχείο (ή τη βάση) και ξανατρέξε |
| Άλλαξε αρχείο εφαρμοσμένου migration | Η εντολή αρνείται (checksum). Επανάφερε το αρχείο όπως ήταν (`git checkout -- <αρχείο>`) και πρόσθεσε νέο migration. Μην επεξεργαστείς χειροκίνητα τον `schema_migrations` |
| Λάθος μεταφορά ή αμφίβολα δεδομένα | Η μεταφορά είναι idempotent: ξανατρέξε την. Για τους τέσσερις βασικούς πίνακες το ingestion από το cache (`data/raw`, `--no-fetch`, ~14 s τοπικά) ξαναφτιάχνει τα ίδια δεδομένα |
| Επιστροφή στην SQLite | Η `data/elfantasy.db` δεν αλλάζει ποτέ από τη μεταφορά: όρισε `DATABASE_URL=sqlite:///data/elfantasy.db` |
| Νέα, κενή βάση (π.χ. νέο project) | `migrate` και μετά `transfer` |
| Αντίγραφο ασφαλείας των `predictions` και της διαθεσιμότητας (υπάρχουν μόνο στο Postgres) | δες παρακάτω |

**Αντίγραφο ασφαλείας του Postgres σε SQLite** (το `transfer` δουλεύει και αντίστροφα, με `--allow-non-postgres`· δοκιμάστηκε σε test):

```bash
.venv/Scripts/python.exe -c "from elfantasy.db.session import get_engine, ensure_schema; ensure_schema(get_engine('sqlite:///data/backup.db'))"
.venv/Scripts/python.exe -m elfantasy.db.transfer --from "$DATABASE_URL" --to sqlite:///data/backup.db --allow-non-postgres
```

Το Supabase έχει δικό του μηχανισμό αντιγράφων ασφαλείας ανά plan: για το free plan **δεν επιβεβαιώθηκε** από την τεκμηρίωση, γι' αυτό μην βασίζεσαι σε αυτόν.

## 12. Περιορισμοί του free tier

Από την τεκμηρίωση του Supabase (ελέγχθηκε στις 2026-10-04, **δεν δοκιμάστηκαν σε πραγματικό project**):

* **Παύση λόγω αδράνειας:** τα projects του free plan «παύουν» μετά από 1 εβδομάδα αδράνειας (https://supabase.com/pricing). Ο ακριβής ορισμός της «αδράνειας» (αν μετρά ένα αίτημα SQL από τον pooler) **δεν επιβεβαιώθηκε**. Ένα παυμένο project πρέπει να αποκατασταθεί από το dashboard· στο μεταξύ το API ξεκινά degraded (και δεν επαναλαμβάνει μόνο του την προσπάθεια: μετά την αποκατάσταση χρειάζεται επανεκκίνηση, `docs/API.md` ενότητα 9). Το `/health` εκτελεί μικρά queries στη βάση, άρα ένας περιοδικός έλεγχος υγείας (π.χ. του Render) παράγει δραστηριότητα· αν αυτό αρκεί για να μην παυθεί το project δεν έχει επιβεβαιωθεί.
* **Μέγεθος:** 500 MB βάση (read-only όταν ξεπεραστεί). Τα τωρινά δεδομένα καταλαμβάνουν ~57 MB σε τοπικό Postgres (όλοι οι πίνακες με indexes, 73.171 γραμμές `player_games` = 47 MB)· ~11% του ορίου.
* **Συνδέσεις:** για το μικρότερο compute (Nano, το προεπιλεγμένο του free plan) το όριο είναι 60 απευθείας συνδέσεις και 200 clients στον pooler (https://supabase.com/docs/guides/platform/compute-and-disk). Το project χρησιμοποιεί έως 10 συνδέσεις ανά διεργασία (pool 5 + 5), άρα το API (μία διεργασία) και ένα ingestion χωράνε άνετα.
* **Statement timeout:** για τον ρόλο `postgres` δεν ορίζεται δικό του όριο, αλλά υπάρχει καθολικό όριο 2 λεπτών ανά statement (https://supabase.com/docs/guides/database/postgres/timeouts). Κάθε παρτίδα της μεταφοράς (2.000 γραμμές) είναι πολύ μικρότερη.
* **Δύο ενεργά projects** ανά λογαριασμό στο free plan.
* **Compute:** κοινή CPU και έως 0,5 GB RAM (Nano)· ο χρόνος μεταφοράς και απόκρισης από το Render **δεν μετρήθηκε**.

## 13. Τι δοκιμάστηκε και τι όχι

**Σε πραγματικό τοπικό Postgres** (PostgreSQL 18.4 μέσω `pixeltable-pgserver`, Windows 11, Python 3.13, loopback):

| Βήμα | Αποτέλεσμα |
|---|---|
| `migrate` (001 και 002) | 0,16 s μέσα στη βάση (1,2 s η εντολή με την εκκίνηση της Python)· δεύτερη εκτέλεση: τίποτα· RLS σε 7 από 7 πίνακες |
| `transfer` της πραγματικής `data/elfantasy.db` | 77.926 γραμμές (teams 32, players 1.212, games 3.511, player_games 73.171) σε 11,3 s (~6.900 γραμμές/s)· επαλήθευση 39 έλεγχοι ΟΚ σε ~4 s· σύνολο 15,2 s |
| `transfer` ξανά (idempotent) | 9,7 s, επαλήθευση ΟΚ, ίδια πλήθη |
| `ingest.pipeline --no-fetch --db` (upsert πάνω στα δεδομένα) | 10 s, ίδια πλήθη (32 / 1.212 / 3.511 / 73.171), και η SQLite πηγή εξακολουθεί να ισούται με το Postgres |
| `ingest.verify` στο Postgres | `pir == valuation` σε 73.171 από 73.171 γραμμές, RESULT: OK |
| `Predictor` σε SQLite και Postgres | 1.212 παίκτες, **μέγιστη διαφορά 0,0** στα `predicted_fantasy` και `predicted_pir`· ίδιοι επόμενοι αγώνες |
| API (`uvicorn`, `DATABASE_URL` προς το τοπικό Postgres) | `/health` ok (6,6 s μέχρι το πρώτο 200), `/rankings`, `/predict`, `POST`/`GET`/`DELETE /availability` (timestamptz), `POST /admin/refresh` 1,58 s, 401 χωρίς κλειδί· κανένα σφάλμα στο log |
| `record_predictions` | 262 προβλέψεις (ενεργοί), ξανά: 262 (καμία διπλή)· με `--all`: 992 συνολικά· `evaluate_recorded`: «nothing to evaluate yet» |

**Στο πραγματικό Supabase** (project `euroleague-fantasy-api`, eu-central-1, free plan, PostgreSQL 17.11, 2026-10-04). Η σύνδεση έγινε με **direct connection** (`db.<ref>.supabase.co:5432`, χρήστης `postgres`) από το laptop (το δίκτυο έχει IPv6)· η σύνδεση είναι κρυπτογραφημένη (TLS 1.3, `TLS_AES_256_GCM_SHA384`, επιβεβαιώθηκε από το `pg_stat_ssl`).

| Βήμα | Αποτέλεσμα |
|---|---|
| `migrate` (001 και 002) | 1,32 s και 0,88 s· δεύτερη εκτέλεση: «nothing (up to date)»· το `--status` δείχνει και τα δύο ως εφαρμοσμένα |
| `transfer` της πραγματικής `data/elfantasy.db` | 77.926 γραμμές σε 69,3 s (1.125 γραμμές/s από την Ελλάδα προς τη Φρανκφούρτη)· σύνολο 90 s με την επαλήθευση. Η SQLite πηγή έμεινε αναλλοίωτη (ίδιο md5) |
| Επαλήθευση | 38 από 39 έλεγχοι ΟΚ την πρώτη φορά. Ο έλεγχος **content digest του `player_games` απέτυχε**: ο server του Supabase έχει `extra_float_digits = 0` και στέλνει τα `double precision` ως κείμενο με 15 σημαντικά ψηφία (19.983333333333334 → 19.9833333333333, σε 44.031 γραμμές της στήλης `minutes`). Η αποθηκευμένη τιμή ήταν σωστή (με `SET extra_float_digits = 3` διαβάζεται ακριβής)· η ανάγνωση την αλλοίωνε. Διορθώθηκε στο `get_engine` (ενότητα 6, `set_full_precision_floats`) και η νέα επαλήθευση έδωσε **39 από 39 ΟΚ**, μαζί με τα ψηφιακά αποτυπώματα όλων των πινάκων |
| Security Advisor του Supabase | Κανένα σφάλμα ή προειδοποίηση. Μόνο 7 πληροφοριακές ενδείξεις «RLS Enabled No Policy» (ένα για κάθε πίνακα), που είναι αναμενόμενες και σκόπιμες (deny-all). RLS ενεργό σε 7 από 7 πίνακες |
| `ingest.verify` στο Supabase | `pir == valuation` σε 73.171 από 73.171 γραμμές, RESULT: OK, ίδια αναφορά με την SQLite |
| `ingest.pipeline --no-fetch` προς το Supabase (upsert) | 65 s, ίδια πλήθη (32 / 1.212 / 3.511 / 73.171) και ο έλεγχος των τεσσάρων βασικών πινάκων παραμένει ΟΚ |
| API τοπικά πάνω στο Supabase (`uvicorn`) | Startup περίπου 10 s, `/health` ok, περίπου 255 ms ανά αίτημα (κυρίως το ταξίδι δικτύου για την ανάγνωση της διαθεσιμότητας), `POST /availability` 0,59 s, `POST /admin/refresh` 8,1 s (ανάγνωση 73.171 γραμμών από το Internet). Οι προβλέψεις ταυτίζονται με της SQLite (Vezenkov 20,13, Bryant 20,9). Ροή `out`: προσωρινή εγγραφή, ο παίκτης έφυγε από το `/rankings`, μετά `DELETE`, και ο πίνακας έμεινε άδειος· κανένα σφάλμα στο log |
| `record_predictions` | 262 προβλέψεις (as_of 2026-10-03), ξανά την ίδια ημέρα: 262 (καμία διπλή)· `evaluate_recorded`: «nothing to evaluate yet» (κανένας αγώνας δεν έχει παιχτεί ακόμη) |
| Μέγεθος | 45 MB στο Supabase (πίνακες και indexes), περίπου 9% του ορίου των 500 MB |

**Επανασυγχρονισμός μετά τη διόρθωση του κανόνα νίκης (2026-10-04).** Ο κανόνας του fantasy score για αρνητικό PIR σε νίκη διορθώθηκε (`FANTASY_RULES.md`, ενότητα 3.4) και η στήλη `player_games.fantasy_score` ξαναϋπολογίστηκε στο Supabase με `python -m elfantasy.ingest.pipeline --seasons 2016-2026 --no-fetch` (65 s, upsert: το ίδιο εργαλείο που φορτώνει τα δεδομένα ενημερώνει και τις τιμές που άλλαξαν). Άλλαξαν 3.432 γραμμές (οι νίκες με αρνητικό PIR), το άθροισμα του `fantasy_score` πήγε από 564.701,9 σε 566.209,9 και η επαλήθευση `transfer --verify-only` για τους τέσσερις βασικούς πίνακες έδωσε **31 από 31 ελέγχους ΟΚ** (ψηφιακό αποτύπωμα του `player_games`: `73171:a5c46750bfae25c0` και στις δύο βάσεις). Στη συνέχεια η `record_predictions` έγραψε 262 προβλέψεις του νέου μοντέλου (`20261004T045157Z-a87106fb`, `as_of` 2026-10-04). Ο πίνακας `predictions` περιέχει πλέον και τις 262 προβλέψεις του προηγούμενου μοντέλου (`20261003T140043Z-7ae47948`, `as_of` 2026-10-03), που έγιναν με τον παλιό κανόνα· διακρίνονται από το `model_version`. Αν η επαλήθευση `transfer --verify-only` τρέξει για ολόκληρο το σχήμα (χωρίς `--tables`), ο πίνακας `predictions` θα εμφανίσει διαφορά από την τοπική SQLite, γιατί οι προβλέψεις γράφονται μόνο στο Supabase.

**Δεν δοκιμάστηκαν** (χρειάζονται άλλο δίκτυο ή το Render):

* **session και transaction pooler** του Supabase και ονόματα χρήστη `postgres.<ref>`. Ο transaction pooler έχει επιπλέον ζήτημα με το `SET extra_float_digits`: η ρύθμιση συνεδρίας δεν είναι εγγυημένη ανάμεσα σε συναλλαγές, άρα εκεί οι ανάγνωση floats μπορεί να ξαναγίνει με 15 ψηφία. Για το API και το ingestion προτιμάται το session pooler ή το direct connection·
* **SSL `require` και `verify-full`:** η σύνδεση χρησιμοποίησε TLS, αλλά δεν δοκιμάστηκε ρητά το `sslmode=require` ούτε η επαλήθευση πιστοποιητικού·
* η συμπεριφορά του transaction pooler (6543) με prepared statements: η απενεργοποίησή τους (`prepare_threshold=None`) δοκιμάστηκε μόνο τοπικά, χωρίς pooler·
* το `INSERT` πολλών γραμμών του psycopg 3 (`executemany`) χρησιμοποιεί εσωτερικά **pipeline mode** του πρωτοκόλλου Postgres· δούλεψε στο direct connection του Supabase, αλλά η συμβατότητά του με τους poolers δεν επιβεβαιώθηκε·
* η **παύση λόγω αδράνειας**, τα όρια συνδέσεων και ο χρόνος του `/health` σε παυμένη βάση·
* **Render:** χρόνος εκκίνησης, μνήμη, πολλαπλές διεργασίες. Σημαντικό: το direct connection του Supabase είναι μόνο IPv6 στο free plan και δεν είναι βέβαιο ότι το Render το φτάνει· στο Render πρέπει να χρησιμοποιηθεί το **session pooler** (IPv4).

## 14. Αποφάσεις σχεδιασμού και αποκλίσεις από το πλάνο

* **Τα migrations ζουν στο πακέτο** (`src/elfantasy/db/migrations/`, όπως όλο το `db/` του πλάνου) και δηλώνονται ως package data στο `pyproject.toml`, ώστε να ακολουθούν το `pip install`.
* **Το `001_init.sql` παράγεται από το `models.py`** και ένα test ελέγχει ότι ισούται με την έξοδο του `migrate --print-sql` (byte προς byte· η έξοδος είναι ντετερμινιστική ανεξάρτητα από το hash seed της Python). Για να μη σπάσει το test σε ένα μελλοντικό migration βλ. την ενότητα 3.
* **Το ingestion σε Postgres δεν κάνει ποτέ `create_all`** (και όχι μόνο όταν υπάρχει `schema_migrations`, όπως έλεγε το πλάνο): ένας πίνακας που δημιουργείται εκτός migrations δεν έχει RLS και θα ήταν εκτεθειμένος στο REST API. Για τον ίδιο λόγο η `create_all` αρνείται Postgres (σηκώνει `RuntimeError`).
* **Ο `schema_migrations` δεν ανήκει στο `models.metadata`** (δεν τον δημιουργεί ούτε το `create_all` ούτε το `001`)· τον δημιουργεί ο runner, με RLS αμέσως μετά τη δημιουργία.
* **Η μεταφορά δεν αντιγράφει το `predictions.id`** και συγκρούεται στο φυσικό κλειδί, γιατί το τεχνικό `id` δεν έχει νόημα ανάμεσα σε βάσεις και η αντιγραφή του θα άφηνε την ακολουθία πίσω.
* **Ψηφιακό αποτύπωμα αντί για απλά aggregates** στην επαλήθευση: τα aggregates δεν πιάνουν ανταλλαγές τιμών ή μικρές αλλαγές κειμένου.
* **Οι logger των εργαλείων έχουν ρητό όνομα** (`elfantasy.db.migrate` κ.λπ.): με `python -m module` το `__name__` είναι `__main__` και τα μηνύματα (ακόμη και τα σφάλματα) δεν θα έφταναν στην κονσόλα. Ένα test με πραγματικές υποδιεργασίες το καλύπτει.
* **Ρητή προστασία των αρχείων μυστικών:** το `.gitignore` αγνοεί κάθε `.env.*` (π.χ. `.env.txt` του Notepad) εκτός από το `.env.example`, και τα tests δεν ανοίγουν ποτέ αρχείο `.env*` εκτός από το `.env.example`.

## 15. Συχνά σφάλματα

| Μήνυμα | Αιτία και λύση |
|---|---|
| `invalid database URL (details are hidden because a URL may contain a password)` | Το URL δεν διαβάζεται (π.χ. ειδικός χαρακτήρας στον κωδικό χωρίς URL-encoding, ενότητα 6). Το μήνυμα είναι σκόπιμα γενικό: δεν επαναλαμβάνει το URL |
| `the target is not PostgreSQL (sqlite): refusing` ή `refusing to copy a database onto itself` | Δεν έχει οριστεί το `DATABASE_URL` του προορισμού: η προεπιλογή του είναι η τοπική SQLite, δηλαδή η ίδια βάση με την πηγή |
| `database tables are missing: …; apply the migrations with: python -m elfantasy.db.migrate` | Η βάση Postgres δεν έχει σχήμα: τρέξε `migrate` (το ingestion, το API και το `transfer` δεν δημιουργούν ποτέ πίνακες σε Postgres) |
| `connection timeout expired` | Λάθος host ή πόρτα, firewall, παυμένο project (free tier, ενότητα 12) ή host μόνο IPv6 (direct connection) από δίκτυο IPv4: χρησιμοποίησε το session pooler |
| `server does not support SSL, but SSL was required` | Ο server δεν έχει SSL (π.χ. τοπικός): αφαίρεσε το `sslmode=require`. Το Supabase υποστηρίζει SSL |
| `password authentication failed for user …` | Λάθος κωδικός, ή (στον pooler) όνομα χρήστη χωρίς το `.<project-ref>`, ή κωδικός χωρίς URL-encoding (μήνυμα του Postgres, δεν παρατηρήθηκε εδώ) |
| `Tenant or user not found` | Μήνυμα του Supavisor όταν το όνομα χρήστη του pooler δεν έχει τη μορφή `postgres.<project-ref>` (αναφέρεται από τη βιβλιογραφία της κοινότητας, **δεν επιβεβαιώθηκε** εδώ) |
| `prepared statement "…" does not exist` ή `already exists` | Χρήση transaction pooler με prepared statements. Στην πόρτα 6543 απενεργοποιούνται αυτόματα· σε άλλη πόρτα πρόσθεσε `?prepare_threshold=none` στο URL (δεν επιβεβαιώθηκε σε pooler) |
| `001_init: the file changed after it was applied (recorded checksum …)` | Άλλαξε αρχείο εφαρμοσμένου migration: επανάφερέ το με `git checkout -- <αρχείο>` και γράψε νέο migration (ενότητα 3) |
