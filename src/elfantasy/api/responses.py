"""Κοινοί ορισμοί των τεκμηριωμένων απαντήσεων σφάλματος (OpenAPI) των endpoints."""

from elfantasy.api.schemas import ErrorOut


def error_responses(code: int, description: str) -> dict[int | str, dict]:
    """Τεκμηρίωση μίας απάντησης σφάλματος με τη μορφή `{"detail": "..."}`."""
    return {code: {"model": ErrorOut, "description": description}}


ERROR_503 = error_responses(
    503, "Το μοντέλο ή η βάση δεν είναι διαθέσιμα (η υπηρεσία είναι degraded, δες το `/health`)."
)

# Προστατευμένα endpoints: λάθος κλειδί → 401, δεν έχει οριστεί κλειδί → 503.
ADMIN_RESPONSES = {
    **error_responses(401, "Λείπει ή είναι λάθος το κλειδί στο header `X-API-Key`."),
    **error_responses(
        503,
        "Τα endpoints διαχείρισης είναι κλειστά (`admin API is disabled`: δεν έχει οριστεί "
        "`ADMIN_API_KEY`) ή η βάση/το μοντέλο δεν είναι διαθέσιμα.",
    ),
}
