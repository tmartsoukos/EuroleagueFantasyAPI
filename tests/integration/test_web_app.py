"""Tests της web εφαρμογής (`src/elfantasy/web`, σερβίρεται από το API στο `/app/`): σερβίρισμα,
τύποι αρχείων, πληρότητα των αναφορών και στατικοί κανόνες ασφαλείας του κώδικα της.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from api_support import ADMIN_KEY

from elfantasy.api.main import WEB_DIR

ROOT = Path(__file__).resolve().parents[2]
WEB_FILES = sorted(path for path in WEB_DIR.iterdir() if path.is_file())
JS_FILES = [path for path in WEB_FILES if path.suffix == ".js"]

MEDIA_TYPES = {
    "app.js": "text/javascript",
    "app.css": "text/css",
    "manifest.webmanifest": "application/manifest+json",
    "icon.svg": "image/svg+xml",
    "icon-192.png": "image/png",
    "icon-512.png": "image/png",
    "sw.js": "text/javascript",
}


class TestServing:
    def test_the_app_page_is_served_at_app_slash(self, client):
        response = client.get("/app/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert '<html lang="el">' in response.text
        assert "Euroleague Fantasy" in response.text

    def test_the_app_without_the_trailing_slash_redirects(self, client):
        response = client.get("/app", follow_redirects=False)
        assert response.status_code in (307, 308)
        assert response.headers["location"].endswith("/app/")

    @pytest.mark.parametrize(("name", "media_type"), sorted(MEDIA_TYPES.items()))
    def test_assets_have_the_right_media_type(self, client, name, media_type):
        response = client.get(f"/app/{name}")
        assert response.status_code == 200
        assert response.headers["content-type"].split(";")[0] == media_type

    def test_assets_are_revalidated_on_every_load(self, client):
        response = client.get("/app/app.js")
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["etag"]
        revalidated = client.get("/app/app.js", headers={"If-None-Match": response.headers["etag"]})
        assert revalidated.status_code == 304

    def test_every_web_file_is_served(self, client):
        for path in WEB_FILES:
            assert client.get(f"/app/{path.name}").status_code == 200, path.name

    def test_unknown_app_paths_are_json_404(self, client):
        response = client.get("/app/nope.js")
        assert response.status_code == 404
        assert response.json() == {"detail": "Not Found"}

    def test_the_static_mount_does_not_shadow_the_api(self, client):
        assert client.get("/health").status_code in (200, 503)
        assert client.get("/rankings?limit=1").status_code in (200, 503)
        assert client.post("/app/").status_code == 405


class TestReferences:
    def test_the_page_references_only_existing_files(self):
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        references = re.findall(r'(?:href|src)="([^"#][^"]*)"', html)
        assert references
        for reference in references:
            assert (WEB_DIR / reference).is_file(), reference

    def test_the_manifest_references_existing_icons(self):
        import json

        manifest = json.loads((WEB_DIR / "manifest.webmanifest").read_text(encoding="utf-8"))
        for icon in manifest["icons"]:
            assert (WEB_DIR / icon["src"]).is_file(), icon["src"]

    @pytest.mark.parametrize("path", JS_FILES, ids=lambda p: p.name)
    def test_js_imports_resolve(self, path):
        for target in re.findall(r"from '(\./[^']+)'", path.read_text(encoding="utf-8")):
            assert (WEB_DIR / target).is_file(), f"{path.name} imports {target}"

    def test_the_service_worker_precaches_existing_files(self):
        source = (WEB_DIR / "sw.js").read_text(encoding="utf-8")
        block = re.search(r"const SHELL = \[(.*?)\];", source, re.DOTALL)
        assert block
        for name in re.findall(r"'([^']+)'", block.group(1)):
            assert name == "./" or (WEB_DIR / name).is_file(), name

    def test_the_service_worker_never_caches_api_responses(self):
        source = (WEB_DIR / "sw.js").read_text(encoding="utf-8")
        assert "url.pathname.startsWith('/app/')" in source


class TestSecurity:
    def test_the_page_has_a_strict_content_security_policy(self):
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        policy = re.search(r'Content-Security-Policy" content="([^"]+)"', html)
        assert policy
        assert "script-src 'self'" in policy.group(1)
        assert "unsafe-inline" not in policy.group(1)
        assert "unsafe-eval" not in policy.group(1)
        assert "<script>" not in html  # κανένα inline script

    @pytest.mark.parametrize("path", JS_FILES, ids=lambda p: p.name)
    def test_js_never_writes_html_or_evaluates_strings(self, path):
        source = path.read_text(encoding="utf-8")
        forbidden_calls = (
            "innerHTML",
            "outerHTML",
            "insertAdjacentHTML",
            "document.write",
            "eval(",
            "new Function",
        )
        for forbidden in forbidden_calls:
            assert forbidden not in source, f"{path.name}: {forbidden}"

    def test_the_admin_key_is_not_in_the_web_files_and_is_session_only(self):
        for path in WEB_FILES:
            if path.suffix in {".png"}:
                continue
            assert ADMIN_KEY not in path.read_text(encoding="utf-8"), path.name
        admin = (WEB_DIR / "view-admin.js").read_text(encoding="utf-8")
        assert "sessionStorage" in admin
        assert "localStorage" not in admin

    def test_no_external_hosts_are_referenced(self):
        for path in WEB_FILES:
            if path.suffix in {".png"}:
                continue
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"https?://(?!www\.w3\.org/2000/svg)", text), path.name


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_lineup_logic_passes_its_node_tests():
    result = subprocess.run(
        ["node", "--test", str(ROOT / "tests" / "web" / "lineup.test.mjs")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
