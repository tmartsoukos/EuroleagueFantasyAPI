"""Κανόνες για το κλειδί διαχειριστή (`ADMIN_API_KEY`): ελάχιστο μήκος και σύγκριση σε bytes.

Ζει σε ξεχωριστό module χωρίς εξαρτήσεις από την υπόλοιπη εφαρμογή, ώστε να το χρησιμοποιούν και το
`deps` (έλεγχος αιτήματος) και το `state` (προειδοποίηση στο startup) χωρίς κυκλικά imports.
"""

from __future__ import annotations

# Ελάχιστο μήκος του ADMIN_API_KEY (βλ. `admin_key_is_usable`).
MIN_ADMIN_KEY_LENGTH = 24


def admin_key_is_usable(key: str) -> bool:
    """True αν το `ADMIN_API_KEY` επιτρέπεται να ανοίξει τα προστατευμένα endpoints.

    Κενό ή πολύ σύντομο κλειδί (π.χ. `1234`) δεν δέχεται κανέναν: τα endpoints μένουν κλειστά
    (503), γιατί χωρίς όριο ρυθμού ένα σύντομο κλειδί σπάει με brute force (το review της Φάσης 7
    μέτρησε περίπου 600 προσπάθειες το δευτερόλεπτο). Το `secrets.token_urlsafe(32)` των docs δίνει
    43 χαρακτήρες.
    """
    return len(key.strip()) >= MIN_ADMIN_KEY_LENGTH


def header_bytes(value: str) -> bytes:
    """Τα bytes της επικεφαλίδας όπως τα έστειλε ο client.

    Το Starlette αποκωδικοποιεί τις επικεφαλίδες ως latin-1· η αντίστροφη κωδικοποίηση ανακτά τα
    αρχικά bytes. Έτσι ένα κλειδί με μη-ASCII χαρακτήρες που στέλνεται σε UTF-8 (όπως κάνουν οι
    περισσότεροι clients) ταιριάζει με το `ADMIN_API_KEY` (που κωδικοποιείται σε UTF-8).
    """
    try:
        return value.encode("latin-1")
    except UnicodeEncodeError:  # δεν συμβαίνει για επικεφαλίδες HTTP, μόνο για τιμές από tests
        return value.encode("utf-8")
