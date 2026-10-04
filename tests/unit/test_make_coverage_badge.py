"""Tests του `scripts/make_coverage_badge.py`: ποσοστό, χρώματα, στρογγυλοποίηση, SVG, CLI.

Το script τρέχει στο CI (job `coverage-badge`) και παράγει το `coverage.svg` του branch `badges`,
που εμφανίζεται στο README. Εδώ ελέγχεται ότι το ποσοστό ταυτίζεται με αυτό του coverage.py (το ίδιο
νούμερο που ελέγχει το `--cov-fail-under`), ότι το SVG είναι έγκυρο και ντετερμινιστικό και ότι οι
λάθος είσοδοι δίνουν καθαρό σφάλμα.
"""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import make_coverage_badge as badge
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "make_coverage_badge.py"


def write_report(path: Path, **attributes: object) -> Path:
    """Γράφει ένα ελάχιστο `coverage.xml` (μορφή Cobertura του coverage.py)."""
    values = {
        "version": "7.0",
        "timestamp": "1",
        "lines-valid": "100",
        "lines-covered": "99",
        "line-rate": "0.99",
        "branches-covered": "0",
        "branches-valid": "0",
        "branch-rate": "0",
        "complexity": "0",
    }
    values.update({key: str(value) for key, value in attributes.items()})
    attrs = " ".join(f'{key}="{value}"' for key, value in values.items())
    path.write_text(
        f'<?xml version="1.0" ?>\n<coverage {attrs}>\n<sources/>\n<packages/>\n</coverage>\n',
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------------------
# Ποσοστό
# --------------------------------------------------------------------------------------


class TestParsePercent:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("0", 0.0), ("100", 100.0), ("98.7", 98.7), (" 95 ", 95.0), ("93.4%", 93.4), (99, 99.0)],
    )
    def test_valid_values(self, text, expected):
        assert badge.parse_percent(text) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "text", ["", "abc", "98,7", "-0.1", "100.01", "101", "nan", "inf", "-inf", "%", "9 9"]
    )
    def test_invalid_values_are_rejected(self, text):
        with pytest.raises(badge.BadgeError):
            badge.parse_percent(text)

    def test_the_error_does_not_include_a_traceback_chain(self):
        with pytest.raises(badge.BadgeError) as caught:
            badge.parse_percent("abc")
        assert caught.value.__cause__ is None  # `from None`: μήνυμα χωρίς εσωτερικά του float()


class TestReadCoverageXml:
    def test_lines_only(self, tmp_path):
        path = write_report(tmp_path / "coverage.xml", **{"lines-valid": 200, "lines-covered": 198})
        assert badge.read_coverage_xml(path) == pytest.approx(99.0)

    def test_branches_are_part_of_the_total_like_coverage_py(self, tmp_path):
        path = write_report(
            tmp_path / "coverage.xml",
            **{
                "lines-valid": 100,
                "lines-covered": 90,
                "branches-valid": 20,
                "branches-covered": 10,
            },
        )
        assert badge.read_coverage_xml(path) == pytest.approx((90 + 10) / (100 + 20) * 100)

    def test_a_report_with_only_a_line_rate_is_accepted(self, tmp_path):
        path = tmp_path / "coverage.xml"
        path.write_text('<coverage line-rate="0.875"/>', encoding="utf-8")
        assert badge.read_coverage_xml(path) == pytest.approx(87.5)

    def test_a_report_without_measured_lines_is_an_error_not_100_percent(self, tmp_path):
        path = write_report(tmp_path / "coverage.xml", **{"lines-valid": 0, "lines-covered": 0})
        with pytest.raises(badge.BadgeError, match="no measured lines"):
            badge.read_coverage_xml(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(badge.BadgeError, match="file not found"):
            badge.read_coverage_xml(tmp_path / "nope.xml")

    def test_a_directory_is_not_a_report(self, tmp_path):
        with pytest.raises(badge.BadgeError):
            badge.read_coverage_xml(tmp_path)

    def test_malformed_xml(self, tmp_path):
        path = tmp_path / "coverage.xml"
        path.write_text("<coverage", encoding="utf-8")
        with pytest.raises(badge.BadgeError, match="cannot be read as XML"):
            badge.read_coverage_xml(path)

    def test_wrong_root_element(self, tmp_path):
        path = tmp_path / "coverage.xml"
        path.write_text("<report/>", encoding="utf-8")
        with pytest.raises(badge.BadgeError, match="not a coverage report"):
            badge.read_coverage_xml(path)

    def test_no_totals_at_all(self, tmp_path):
        path = tmp_path / "coverage.xml"
        path.write_text("<coverage/>", encoding="utf-8")
        with pytest.raises(badge.BadgeError, match="no coverage totals"):
            badge.read_coverage_xml(path)

    def test_the_percentage_matches_coverage_py_for_lines_and_branches(self, tmp_path):
        """Πραγματικό `coverage.xml` του coverage.py: το νούμερο του badge είναι το `totals` του."""
        module = tmp_path / "sample.py"
        module.write_text(
            "def classify(value):\n"
            "    if value > 0:\n"
            "        return 'positive'\n"
            "    elif value < 0:\n"
            "        return 'negative'\n"
            "    return 'zero'\n"
            "\n"
            "\n"
            "def never_called():\n"
            "    return 1\n"
            "\n"
            "\n"
            "classify(1)\n"
            "classify(0)\n",
            encoding="utf-8",
        )
        for branch in (False, True):
            data = tmp_path / f".coverage{int(branch)}"
            run = [sys.executable, "-m", "coverage", "run", f"--data-file={data}"]
            if branch:
                run.append("--branch")
            subprocess.run([*run, str(module)], cwd=tmp_path, check=True, capture_output=True)
            xml_path = tmp_path / f"coverage{int(branch)}.xml"
            json_path = tmp_path / f"coverage{int(branch)}.json"
            for command, target in (("xml", xml_path), ("json", json_path)):
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "coverage",
                        command,
                        f"--data-file={data}",
                        "-o",
                        target,
                    ],
                    cwd=tmp_path,
                    check=True,
                    capture_output=True,
                )
            expected = json.loads(json_path.read_text(encoding="utf-8"))["totals"]
            assert badge.read_coverage_xml(xml_path) == pytest.approx(
                expected["percent_covered"], abs=1e-9
            )
            assert 0 < expected["percent_covered"] < 100


# --------------------------------------------------------------------------------------
# Χρώμα και στρογγυλοποίηση
# --------------------------------------------------------------------------------------


class TestColor:
    @pytest.mark.parametrize(
        ("percent", "color"),
        [
            (100, "#4c1"),
            (95, "#4c1"),
            (94.99, "#97ca00"),
            (90, "#97ca00"),
            (89.99, "#a4a61d"),
            (80, "#a4a61d"),
            (79.99, "#dfb317"),
            (70, "#dfb317"),
            (69.99, "#fe7d37"),
            (60, "#fe7d37"),
            (59.99, "#e05d44"),
            (0, "#e05d44"),
        ],
    )
    def test_color_by_percentage(self, percent, color):
        assert badge.color_for(percent) == color

    def test_the_colors_are_ordered_from_green_to_red(self):
        thresholds = [threshold for threshold, _ in badge.COLOR_STEPS]
        assert thresholds == sorted(thresholds, reverse=True)
        assert len({color for _, color in badge.COLOR_STEPS} | {badge.COLOR_LOW}) == 6


class TestFormatPercent:
    @pytest.mark.parametrize(
        ("percent", "text"),
        [
            (0, "0%"),
            (0.4, "0%"),
            (0.5, "1%"),
            (94.5, "95%"),  # τα μισά προς τα πάνω (όχι «στρογγυλοποίηση του τραπεζίτη»)
            (95.49, "95%"),
            (98.7, "99%"),
            (99.4, "99%"),
            (99.5, "99%"),  # ποτέ 100% αν δεν είναι πραγματικά 100
            (99.99, "99%"),
            (100, "100%"),
            (100.0, "100%"),
        ],
    )
    def test_rounding(self, percent, text):
        assert badge.format_percent(percent) == text


# --------------------------------------------------------------------------------------
# SVG
# --------------------------------------------------------------------------------------

SVG = "{http://www.w3.org/2000/svg}"


class TestRenderSvg:
    def test_the_svg_is_well_formed_and_self_contained(self):
        svg = badge.render_svg("coverage", "99%", "#4c1")
        root = ET.fromstring(svg)
        assert root.tag == f"{SVG}svg"
        assert root.get("role") == "img"
        assert root.get("aria-label") == "coverage: 99%"
        assert root.findtext(f"{SVG}title") == "coverage: 99%"
        assert "http://" not in svg.replace("http://www.w3.org/2000/svg", "")
        assert "<script" not in svg and "href=" not in svg  # καμία εξωτερική αναφορά ή κώδικας

    def test_the_value_box_has_the_color_and_the_widths_add_up(self):
        root = ET.fromstring(badge.render_svg("coverage", "87%", "#dfb317"))
        total = float(root.get("width"))
        rects = list(root.iter(f"{SVG}rect"))
        label_box = next(r for r in rects if r.get("fill") == badge.LABEL_COLOR)
        value_box = next(r for r in rects if r.get("fill") == "#dfb317")
        assert float(label_box.get("width")) + float(value_box.get("width")) == pytest.approx(total)
        assert float(value_box.get("x")) == pytest.approx(float(label_box.get("width")))
        texts = [t.text for t in root.iter(f"{SVG}text")]
        assert texts.count("coverage") == 2 and texts.count("87%") == 2  # κείμενο και σκιά

    def test_a_longer_message_makes_a_wider_badge(self):
        short = ET.fromstring(badge.render_svg("coverage", "9%", "#e05d44")).get("width")
        long = ET.fromstring(badge.render_svg("coverage", "100%", "#4c1")).get("width")
        assert float(long) > float(short)

    def test_special_characters_are_escaped(self):
        svg = badge.render_svg('a<b&"c', '9"%', "#4c1")
        root = ET.fromstring(svg)  # θα έσκαγε με ανεπίτρεπτο χαρακτήρα
        assert root.get("aria-label") == 'a<b&"c: 9"%'

    def test_the_output_is_deterministic(self):
        assert badge.render_svg("coverage", "99%", "#4c1") == badge.render_svg(
            "coverage", "99%", "#4c1"
        )

    def test_ends_with_a_single_newline_and_has_no_carriage_returns(self):
        svg = badge.render_svg("coverage", "99%", "#4c1")
        assert svg.endswith("</svg>\n") and "\r" not in svg


# --------------------------------------------------------------------------------------
# Γραμμή εντολών
# --------------------------------------------------------------------------------------


class TestCommandLine:
    def test_writes_the_svg_from_a_percentage(self, tmp_path):
        output = tmp_path / "coverage.svg"
        assert badge.main(["--percent", "98.6", "--output", str(output)]) == 0
        root = ET.fromstring(output.read_text(encoding="utf-8"))
        assert root.get("aria-label") == "coverage: 99%"

    def test_the_file_has_lf_line_endings_on_every_platform(self, tmp_path):
        output = tmp_path / "coverage.svg"
        badge.main(["--percent", "50", "--output", str(output)])
        assert b"\r" not in output.read_bytes()

    def test_reads_the_percentage_from_a_report_and_prints_it(self, tmp_path, capsys):
        report = write_report(tmp_path / "coverage.xml", **{"lines-valid": 3, "lines-covered": 2})
        output = tmp_path / "coverage.svg"
        assert badge.main(["--xml", str(report), "--output", str(output), "--print-percent"]) == 0
        assert capsys.readouterr().out == "66.67\n"
        assert "67%" in output.read_text(encoding="utf-8")

    def test_print_percent_alone_does_not_need_an_output_file(self, tmp_path, capsys):
        report = write_report(tmp_path / "coverage.xml")
        assert badge.main(["--xml", str(report), "--print-percent"]) == 0
        assert capsys.readouterr().out == "99.00\n"
        assert list(tmp_path.glob("*.svg")) == []

    def test_the_same_display_gives_identical_bytes_so_ci_makes_no_new_commit(self, tmp_path):
        first, second = tmp_path / "a.svg", tmp_path / "b.svg"
        badge.main(["--percent", "99.31", "--output", str(first)])
        badge.main(["--percent", "99.04", "--output", str(second)])
        assert first.read_bytes() == second.read_bytes()
        badge.main(["--percent", "98.0", "--output", str(second)])
        assert first.read_bytes() != second.read_bytes()

    def test_a_custom_label(self, tmp_path):
        output = tmp_path / "x.svg"
        badge.main(["--percent", "80", "--label", "tests", "--output", str(output)])
        assert "tests: 80%" in output.read_text(encoding="utf-8")

    @pytest.mark.parametrize("value", ["abc", "101", "-5", ""])
    def test_a_bad_percentage_is_exit_code_2_and_writes_nothing(self, tmp_path, capsys, value):
        output = tmp_path / "coverage.svg"
        assert badge.main(["--percent", value, "--output", str(output)]) == 2
        assert not output.exists()
        error = capsys.readouterr().err
        assert error.startswith("error: ") and "Traceback" not in error

    def test_a_missing_report_is_exit_code_2(self, tmp_path, capsys):
        output = tmp_path / "coverage.svg"
        code = badge.main(["--xml", str(tmp_path / "missing.xml"), "--output", str(output)])
        assert code == 2 and not output.exists()
        assert "file not found" in capsys.readouterr().err

    def test_the_two_sources_are_mutually_exclusive_and_one_is_required(self, tmp_path):
        with pytest.raises(SystemExit) as both:
            badge.main(["--xml", "a.xml", "--percent", "5", "--output", str(tmp_path / "x.svg")])
        assert both.value.code == 2
        with pytest.raises(SystemExit) as neither:
            badge.main(["--output", str(tmp_path / "x.svg")])
        assert neither.value.code == 2

    def test_something_to_do_is_required(self):
        with pytest.raises(SystemExit) as nothing:
            badge.main(["--percent", "50"])
        assert nothing.value.code == 2

    def test_runs_as_a_script_without_the_package_installed(self, tmp_path):
        """Όπως στο CI: `python scripts/make_coverage_badge.py ...` (μόνο τυπική βιβλιοθήκη)."""
        output = tmp_path / "coverage.svg"
        result = subprocess.run(
            [sys.executable, "-I", str(SCRIPT), "--percent", "97.2", "--output", str(output)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        assert "97%" in output.read_text(encoding="utf-8")
