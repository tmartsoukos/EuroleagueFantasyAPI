"""Ρυθμίσεις της εφαρμογής: μεταβλητές περιβάλλοντος και αρχείο `.env`.

Οι τιμές διαβάζονται με σειρά προτεραιότητας: μεταβλητές περιβάλλοντος, αρχείο `.env`
(στον τρέχοντα φάκελο), προεπιλογές. Οι σχετικές διαδρομές (π.χ. `data`) υπολογίζονται
ως προς τον τρέχοντα φάκελο εργασίας, οπότε οι εντολές τρέχουν από τη ρίζα του repo.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from elfantasy.db.urls import safe_url

# Προεπιλεγμένο όριο MAE (σε μονάδες fantasy score) του quality gate (docs/MODEL.md, ενότητα 8).
# Το honest test MAE του committed μοντέλου στη σεζόν 2025 είναι 5,909 και το καλύτερο naive
# baseline (μέσος όρος σεζόν του παίκτη) έχει 6,148. Το 6,00 αφήνει περιθώριο 1,5% πάνω από το
# μοντέλο και απαιτεί να κερδίζεται το naive baseline τουλάχιστον κατά 2,4%. Το «×1,05» (6,21)
# θα ήταν πάνω από το naive baseline και δεν χρησιμοποιείται.
DEFAULT_MAE_THRESHOLD = 6.00


class Settings(BaseSettings):
    """Ρυθμίσεις του project.

    Το όνομα κάθε μεταβλητής περιβάλλοντος είναι το όνομα του πεδίου με κεφαλαία γράμματα.
    """

    # protected_namespaces=(): το πεδίο `model_path` ξεκινά με `model_` και αλλιώς το pydantic
    # θα έβγαζε προειδοποίηση σε παλαιότερες εκδόσεις.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
        # Το `git_commit` διαβάζεται από άλλα ονόματα μεταβλητών (βλ. παρακάτω)· τα tests το δίνουν
        # και με το όνομα του πεδίου.
        validate_by_name=True,
        validate_by_alias=True,
    )

    # Σύνδεση στη βάση: SQLite τοπικά, Postgres (Supabase, session pooler) στην παραγωγή
    # (docs/DATABASE.md). Περιέχει τον κωδικό της βάσης: δεν γράφεται ποτέ σε log ή σε αρχείο.
    database_url: str = "sqlite:///data/elfantasy.db"
    # Διαδρομή του αποθηκευμένου μοντέλου (Φάσεις 3 και 4), σχετική με τον φάκελο εργασίας.
    model_path: str = "models/model.joblib"
    # Όριο MAE του quality gate: το train.py αποθηκεύει μοντέλο μόνο αν το test MAE είναι
    # μικρότερο, και το tests/quality το ελέγχει στο committed μοντέλο. Τιμή ≤ 0 σημαίνει ότι
    # δεν έχει οριστεί όριο (η εκπαίδευση αποτυγχάνει με σαφές μήνυμα).
    mae_threshold: float = DEFAULT_MAE_THRESHOLD
    # Κλειδί για τα προστατευμένα endpoints (header X-API-Key). Κενό = τα endpoints είναι κλειστά.
    admin_api_key: str = ""
    # Φάκελος δεδομένων: raw cache, αναφορές και logs.
    data_dir: str = "data"
    # Το commit (SHA) του κώδικα που τρέχει. Το Render το ορίζει μόνο του ως `RENDER_GIT_COMMIT`·
    # τοπικά μένει κενό. Εμφανίζεται στο `GET /health`, ώστε το CI να επιβεβαιώνει μετά από ένα
    # deploy ότι απαντά η ΝΕΑ έκδοση και όχι η παλιά (scripts/smoke_check.py, docs/DEPLOY.md).
    git_commit: str = Field(
        default="", validation_alias=AliasChoices("RENDER_GIT_COMMIT", "GIT_COMMIT")
    )

    def __repr_args__(self):
        """Η αναπαράσταση (`repr`, `str`) κρύβει τον κωδικό της βάσης και το κλειδί διαχειριστή:
        οι ρυθμίσεις μπορεί να τυπωθούν κατά λάθος σε log ή σε μήνυμα σφάλματος."""
        for name, value in super().__repr_args__():
            if name == "database_url" and isinstance(value, str):
                value = safe_url(value)
            elif name == "admin_api_key" and value:
                value = "***"
            yield name, value

    @property
    def raw_dir(self) -> Path:
        """Φάκελος του raw cache (parquet ανά σεζόν)."""
        return Path(self.data_dir) / "raw"

    @property
    def reports_dir(self) -> Path:
        """Φάκελος των αναφορών ποιότητας (CSV)."""
        return Path(self.data_dir) / "reports"

    @property
    def logs_dir(self) -> Path:
        """Φάκελος των αρχείων log του ingestion."""
        return Path(self.data_dir) / "logs"


@lru_cache
def get_settings() -> Settings:
    """Επιστρέφει τις ρυθμίσεις (μία φορά ανά διεργασία).

    Στα tests χρησιμοποιείται το `get_settings.cache_clear()` για να ξαναδιαβαστούν.
    """
    return Settings()
