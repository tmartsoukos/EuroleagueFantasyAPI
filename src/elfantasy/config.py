"""Ρυθμίσεις της εφαρμογής: μεταβλητές περιβάλλοντος και αρχείο `.env`.

Οι τιμές διαβάζονται με σειρά προτεραιότητας: μεταβλητές περιβάλλοντος, αρχείο `.env`
(στον τρέχοντα φάκελο), προεπιλογές. Οι σχετικές διαδρομές (π.χ. `data`) υπολογίζονται
ως προς τον τρέχοντα φάκελο εργασίας, οπότε οι εντολές τρέχουν από τη ρίζα του repo.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    )

    # Σύνδεση στη βάση: SQLite τοπικά, Postgres (Supabase) στη Φάση 5.
    database_url: str = "sqlite:///data/elfantasy.db"
    # Διαδρομή του αποθηκευμένου μοντέλου (Φάσεις 3 και 4).
    model_path: str = "models/model.joblib"
    # Όριο MAE του quality gate. Το 0 σημαίνει ότι δεν έχει οριστεί ακόμη πραγματικό όριο.
    mae_threshold: float = 0.0
    # Κλειδί για τα προστατευμένα endpoints (header X-API-Key). Κενό = τα endpoints είναι κλειστά.
    admin_api_key: str = ""
    # Φάκελος δεδομένων: raw cache, αναφορές και logs.
    data_dir: str = "data"

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
