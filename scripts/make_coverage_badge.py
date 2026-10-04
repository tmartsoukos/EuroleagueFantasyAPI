#!/usr/bin/env python3
"""Δημιουργεί αυτοτελές SVG badge κάλυψης (coverage), χωρίς εξωτερική υπηρεσία ή εξάρτηση.

Το CI (job `coverage-badge`) καλεί το script μετά από κάθε push στο `main` και γράφει το
αποτέλεσμα (`coverage.svg`) στο orphan branch `badges`, από όπου το διαβάζει το README. Χρησιμοποιεί
μόνο την τυπική βιβλιοθήκη της Python.

Είσοδος (ακριβώς μία από τις δύο):

* `--xml coverage.xml`: το αρχείο που παράγει το `pytest --cov-report=xml`. Το ποσοστό είναι το ίδιο
  με αυτό του `coverage report` (και του `--cov-fail-under`): (γραμμές + κλάδοι που εκτελέστηκαν)
  προς (γραμμές + κλάδους που μετρήθηκαν).
* `--percent 98.7`: έτοιμο ποσοστό (το αποτέλεσμα του `--print-percent` από προηγούμενο βήμα).

Έξοδος: `--output coverage.svg` (SVG) και/ή `--print-percent` (το ποσοστό με 2 δεκαδικά στην
κονσόλα, κατάλληλο για το `$GITHUB_OUTPUT`). Το SVG είναι ντετερμινιστικό: το ίδιο ποσοστό (στον
ίδιο ακέραιο και χρώμα) δίνει πάντα τα ίδια bytes, ώστε το CI να μην κάνει περιττά commits.

Κωδικοί εξόδου: 0 επιτυχία, 2 άκυρη είσοδος ή αρχείο (μήνυμα στο stderr, χωρίς traceback).
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

# Ελάχιστο ποσοστό -> χρώμα (τα χρώματα των badges του shields.io: brightgreen, green,
# yellowgreen, yellow, orange). Κάτω από το χαμηλότερο όριο το χρώμα είναι κόκκινο.
COLOR_STEPS = (
    (95.0, "#4c1"),
    (90.0, "#97ca00"),
    (80.0, "#a4a61d"),
    (70.0, "#dfb317"),
    (60.0, "#fe7d37"),
)
COLOR_LOW = "#e05d44"
LABEL_COLOR = "#555"

# Προσεγγιστικά πλάτη χαρακτήρων της γραμματοσειράς Verdana στα 11 px (μονάδες: px). Το SVG
# ορίζει και `textLength`, άρα μια μικρή απόκλιση δεν αλλάζει τη διάταξη.
CHAR_WIDTHS = {
    **dict.fromkeys("0123456789", 7.0),
    "%": 10.5,
    ".": 3.7,
    " ": 3.8,
    "a": 6.1,
    "c": 5.9,
    "e": 6.2,
    "g": 6.8,
    "o": 6.8,
    "r": 4.6,
    "v": 6.2,
}
DEFAULT_CHAR_WIDTH = 6.6
PADDING = 5.0


class BadgeError(ValueError):
    """Άκυρη είσοδος (ποσοστό ή αρχείο coverage)."""


def parse_percent(value: str | float) -> float:
    """Μετατρέπει σε ποσοστό 0..100. Δέχεται και κείμενο με κενά ή με `%` στο τέλος."""
    text = str(value).strip().rstrip("%").strip()
    try:
        percent = float(text)
    except ValueError:
        raise BadgeError(f"not a number: {str(value)!r}") from None
    if not math.isfinite(percent) or not 0.0 <= percent <= 100.0:
        raise BadgeError(f"the percentage must be between 0 and 100, got {str(value)!r}")
    return percent


def read_coverage_xml(path: str | Path) -> float:
    """Το συνολικό ποσοστό κάλυψης από ένα `coverage.xml` (μορφή Cobertura του coverage.py)."""
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except FileNotFoundError:
        raise BadgeError(f"{path}: file not found") from None
    except (ET.ParseError, OSError) as error:
        raise BadgeError(f"{path}: cannot be read as XML ({error})") from None
    if root.tag != "coverage":
        raise BadgeError(f"{path}: not a coverage report (root element <{root.tag}>)")
    try:
        lines_valid = int(root.attrib["lines-valid"])
        lines_covered = int(root.attrib["lines-covered"])
        branches_valid = int(root.attrib.get("branches-valid", "0"))
        branches_covered = int(root.attrib.get("branches-covered", "0"))
    except (KeyError, ValueError):
        # Παλαιότερα ή απλοποιημένα αρχεία έχουν μόνο το line-rate (κλάσμα 0..1).
        try:
            return parse_percent(float(root.attrib["line-rate"]) * 100.0)
        except (KeyError, ValueError, BadgeError):
            raise BadgeError(f"{path}: no coverage totals in the report") from None
    measured = lines_valid + branches_valid
    if measured <= 0:
        raise BadgeError(f"{path}: the report contains no measured lines")
    return parse_percent((lines_covered + branches_covered) / measured * 100.0)


def color_for(percent: float) -> str:
    """Το χρώμα του μέρους του μηνύματος ανάλογα με το (ακριβές) ποσοστό."""
    for threshold, color in COLOR_STEPS:
        if percent >= threshold:
            return color
    return COLOR_LOW


def format_percent(percent: float) -> str:
    """Το ποσοστό ως ακέραιος με `%`: στρογγυλοποίηση στον πλησιέστερο (τα μισά προς τα πάνω)·
    τιμή κάτω από 100 δεν εμφανίζεται ποτέ ως 100%, όπως και στο `coverage report`."""
    rounded = math.floor(percent + 0.5)
    if rounded >= 100 and percent < 100.0:
        rounded = 99
    return f"{rounded}%"


def text_width(text: str) -> float:
    return sum(CHAR_WIDTHS.get(char, DEFAULT_CHAR_WIDTH) for char in text)


def _number(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def render_svg(label: str, message: str, color: str) -> str:
    """Το badge ως SVG (στυλ «flat»): γκρι τμήμα με την ετικέτα και χρωματιστό με το μήνυμα."""
    label_text, message_text = text_width(label), text_width(message)
    label_box = label_text + 2 * PADDING
    message_box = message_text + 2 * PADDING
    total = label_box + message_box
    title = f"{label}: {message}"
    shared = 'fill="#010101" fill-opacity=".3"'
    label_x, message_x = label_box / 2, label_box + message_box / 2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_number(total)}" height="20" '
        f'role="img" aria-label={quoteattr(title)}>',
        f"<title>{escape(title)}</title>",
        '<linearGradient id="s" x2="0" y2="100%">'
        '<stop offset="0" stop-color="#bbb" stop-opacity=".1"/>'
        '<stop offset="1" stop-opacity=".1"/></linearGradient>',
        f'<clipPath id="r"><rect width="{_number(total)}" height="20" rx="3" fill="#fff"/>'
        "</clipPath>",
        '<g clip-path="url(#r)">',
        f'<rect width="{_number(label_box)}" height="20" fill="{LABEL_COLOR}"/>',
        f'<rect x="{_number(label_box)}" width="{_number(message_box)}" height="20" '
        f'fill="{color}"/>',
        f'<rect width="{_number(total)}" height="20" fill="url(#s)"/>',
        "</g>",
        '<g fill="#fff" text-anchor="middle" '
        'font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" '
        'font-size="11">',
        f'<text aria-hidden="true" x="{_number(label_x)}" y="15" {shared} '
        f'textLength="{_number(label_text)}">{escape(label)}</text>',
        f'<text x="{_number(label_x)}" y="14" textLength="{_number(label_text)}">'
        f"{escape(label)}</text>",
        f'<text aria-hidden="true" x="{_number(message_x)}" y="15" {shared} '
        f'textLength="{_number(message_text)}">{escape(message)}</text>',
        f'<text x="{_number(message_x)}" y="14" textLength="{_number(message_text)}">'
        f"{escape(message)}</text>",
        "</g>",
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Δημιουργεί SVG badge κάλυψης από coverage.xml ή από έτοιμο ποσοστό."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--xml", metavar="FILE", help="αρχείο coverage.xml (Cobertura)")
    source.add_argument("--percent", metavar="VALUE", help="έτοιμο ποσοστό, 0 έως 100")
    parser.add_argument("--output", metavar="FILE", help="αρχείο SVG εξόδου")
    parser.add_argument("--label", default="coverage", help="ετικέτα του badge (coverage)")
    parser.add_argument(
        "--print-percent",
        action="store_true",
        help="τυπώνει το ποσοστό με 2 δεκαδικά (για το $GITHUB_OUTPUT)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output is None and not args.print_percent:
        parser.error("nothing to do: give --output and/or --print-percent")
    try:
        percent = read_coverage_xml(args.xml) if args.xml else parse_percent(args.percent)
    except BadgeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.output is not None:
        svg = render_svg(args.label, format_percent(percent), color_for(percent))
        # newline="\n": ίδια bytes σε Windows και Linux (το .gitattributes ορίζει eol=lf).
        with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(svg)
    if args.print_percent:
        print(f"{percent:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
