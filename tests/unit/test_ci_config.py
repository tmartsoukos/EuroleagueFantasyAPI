"""Έλεγχοι του CI/CD (Φάση 6): `.github/workflows/ci.yml`, `render.yaml` και `constraints.txt`.

Το workflow δεν μπορεί να τρέξει τοπικά, γι' αυτό η δομή του ελέγχεται εδώ: ποια jobs υπάρχουν, ποιο
περιμένει ποιο, ότι το deploy γίνεται μόνο στο main, ότι τα δικαιώματα είναι τα ελάχιστα, ότι
δεν υπάρχει κανένα μυστικό μέσα στα αρχεία και ότι οι εκδόσεις Python και πακέτων είναι συνεπείς
ανάμεσα στο CI, στο Render και στο μοντέλο.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

from elfantasy.config import Settings

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ci.yml"
RENDER_PATH = ROOT / "render.yaml"
CONSTRAINTS_PATH = ROOT / "constraints.txt"

JOBS = {"lint", "test", "quality-gate", "deploy", "coverage-badge"}
#: `actions/…@v6` (major tag) ή `…@<sha-40>`. Ποτέ branch (`@main`) ή `@latest`.
PINNED_ACTION = re.compile(r"^[\w.-]+/[\w.-]+(?:/[\w./-]+)?@(?:v\d+(?:\.\d+){0,2}|[0-9a-f]{40})$")
SECRET_PATTERNS = {
    "Deploy Hook του Render": re.compile(r"api\.render\.com/deploy/srv-", re.I),
    "παράμετρος key= σε URL": re.compile(r"[?&]key=[A-Za-z0-9_-]{6,}"),
    "token του GitHub": re.compile(
        r"\b(?:ghp|gho|ghs|ghu|ghr)_[A-Za-z0-9]{20,}|github_pat_\w{20,}"
    ),
    "JWT": re.compile(r"eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}"),
    "Supabase": re.compile(r"supabase\.(?:co|com)", re.I),
    "Supabase secret key": re.compile(r"sb_secret_[A-Za-z0-9_-]{10,}"),
    "υπηρεσία onrender.com": re.compile(r"[\w-]+\.onrender\.com", re.I),
}
URL_WITH_PASSWORD = re.compile(r"\w+(?:\+\w+)?://[^:/@\s'\"]+:[^@\s'\"]+@([^:/?#\s'\"]+)")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def workflow() -> dict:
    return load_yaml(WORKFLOW_PATH)


@pytest.fixture(scope="module")
def jobs(workflow) -> dict:
    return workflow["jobs"]


@pytest.fixture(scope="module")
def render() -> dict:
    return load_yaml(RENDER_PATH)


@pytest.fixture(scope="module")
def service(render) -> dict:
    (only,) = render["services"]
    return only


def steps_of(job: dict) -> list[dict]:
    return job["steps"]


def run_scripts(job: dict) -> list[str]:
    return [step["run"] for step in steps_of(job) if "run" in step]


def all_uses(jobs: dict) -> list[str]:
    return [step["uses"] for job in jobs.values() for step in steps_of(job) if "uses" in step]


def find_step(job: dict, needle: str) -> dict:
    """Το πρώτο step που περιέχει το κείμενο `needle` στο όνομα ή στην εντολή του."""
    for step in steps_of(job):
        if needle in step.get("name", "") or needle in step.get("run", ""):
            return step
    raise AssertionError(f"no step contains {needle!r}")


def expressions(text: str) -> set[str]:
    return set(re.findall(r"\$\{\{\s*(.*?)\s*\}\}", text))


# --------------------------------------------------------------------------------------
# Δομή του workflow
# --------------------------------------------------------------------------------------


class TestWorkflowStructure:
    def test_the_file_is_valid_yaml_with_a_name(self, workflow):
        assert workflow["name"] == "CI"

    def test_it_runs_on_every_branch_push_and_on_pull_requests(self, workflow):
        triggers = workflow.get("on", workflow.get(True))  # το YAML 1.1 διαβάζει το `on` ως True
        assert set(triggers) == {"push", "pull_request"}
        # Όλα τα branches, εκτός από το `badges` (μόνο το coverage.svg, χωρίς κώδικα).
        assert triggers["push"] == {"branches-ignore": ["badges"]}
        assert "branches" not in triggers["push"]

    def test_the_jobs_are_exactly_the_documented_ones(self, jobs):
        assert set(jobs) == JOBS

    def test_deploy_waits_for_lint_test_and_the_quality_gate(self, jobs):
        assert set(jobs["deploy"]["needs"]) == {"lint", "test", "quality-gate"}

    def test_the_badge_waits_for_the_tests_only(self, jobs):
        assert jobs["coverage-badge"]["needs"] == "test"

    def test_lint_test_and_the_quality_gate_run_in_parallel(self, jobs):
        for name in ("lint", "test", "quality-gate"):
            assert "needs" not in jobs[name], name

    @pytest.mark.parametrize("name", ["deploy", "coverage-badge"])
    def test_deploy_and_the_badge_run_only_on_a_push_to_main(self, jobs, name):
        condition = jobs[name]["if"]
        assert "github.event_name == 'push'" in condition
        assert "github.ref == 'refs/heads/main'" in condition
        assert "always()" not in condition  # αλλιώς θα έτρεχε και μετά από αποτυχία των `needs`

    @pytest.mark.parametrize("name", ["lint", "test", "quality-gate"])
    def test_the_checks_have_no_condition(self, jobs, name):
        assert "if" not in jobs[name]

    def test_every_job_has_a_timeout(self, jobs):
        for name, job in jobs.items():
            assert isinstance(job["timeout-minutes"], int), name
            assert 1 <= job["timeout-minutes"] <= 60, name

    def test_every_job_runs_on_a_pinned_ubuntu_image(self, jobs):
        assert {job["runs-on"] for job in jobs.values()} == {"ubuntu-24.04"}

    def test_older_runs_of_the_same_ref_are_cancelled_but_never_on_main(self, workflow):
        concurrency = workflow["concurrency"]
        assert "github.ref" in concurrency["group"]
        assert concurrency["cancel-in-progress"] == "${{ github.ref != 'refs/heads/main' }}"

    def test_deploys_are_serialized_and_never_cancelled(self, jobs):
        assert jobs["deploy"]["concurrency"] == {
            "group": "deploy-production",
            "cancel-in-progress": False,
        }
        assert jobs["coverage-badge"]["concurrency"]["cancel-in-progress"] is False

    def test_bash_with_pipefail_is_the_default_shell(self, workflow):
        assert workflow["defaults"]["run"]["shell"] == "bash"


class TestPermissionsAndActions:
    def test_the_default_permissions_are_read_only(self, workflow):
        assert workflow["permissions"] == {"contents": "read"}

    def test_only_the_badge_job_may_write(self, jobs):
        assert jobs["coverage-badge"]["permissions"] == {"contents": "write"}
        for name, job in jobs.items():
            if name != "coverage-badge":
                assert "permissions" not in job, f"{name} would override the read-only default"

    def test_the_badge_job_is_the_only_one_that_pushes(self, jobs):
        for name, job in jobs.items():
            pushes = any("git push" in script for script in run_scripts(job))
            assert pushes == (name == "coverage-badge"), name

    def test_every_action_is_pinned_to_a_tag_or_a_commit(self, jobs):
        used = all_uses(jobs)
        assert used, "no actions found"
        for reference in used:
            assert PINNED_ACTION.match(reference), f"{reference} is not pinned to a tag or sha"
            assert not reference.endswith(("@main", "@master", "@latest"))

    def test_only_official_github_actions_are_used(self, jobs):
        assert {reference.split("/")[0] for reference in all_uses(jobs)} == {"actions"}

    def test_run_steps_do_not_interpolate_event_data(self, jobs):
        """Προστασία από script injection: κανένα `${{ github.event… }}` ή branch στα scripts."""
        for name, job in jobs.items():
            for script in run_scripts(job):
                for expression in expressions(script):
                    assert not expression.startswith(("github.event", "github.head_ref")), (
                        name,
                        expression,
                    )


class TestSecretsAreNeverInTheFile:
    def test_no_secret_or_service_url_is_written_in_the_workflow(self):
        text = WORKFLOW_PATH.read_text(encoding="utf-8")
        for label, pattern in SECRET_PATTERNS.items():
            assert not pattern.search(text), f"{label} found in ci.yml"

    def test_the_only_database_url_with_a_password_points_to_localhost(self):
        text = WORKFLOW_PATH.read_text(encoding="utf-8")
        hosts = URL_WITH_PASSWORD.findall(text)
        assert hosts == ["localhost"]  # το TEST_DATABASE_URL του service container

    def test_the_only_secret_and_variable_used_are_the_documented_ones(self):
        text = WORKFLOW_PATH.read_text(encoding="utf-8")
        assert set(re.findall(r"secrets\.(\w+)", text)) == {"RENDER_DEPLOY_HOOK_URL"}
        assert set(re.findall(r"\bvars\.(\w+)", text)) == {"RENDER_SERVICE_URL"}

    def test_secrets_are_never_used_inside_a_condition(self, jobs):
        for name, job in jobs.items():
            assert "secrets" not in str(job.get("if", "")), name
            for step in steps_of(job):
                assert "secrets" not in str(step.get("if", "")), (name, step.get("name"))

    def test_the_secret_reaches_the_steps_only_through_the_job_environment(self, jobs):
        deploy = jobs["deploy"]
        assert deploy["env"]["RENDER_DEPLOY_HOOK_URL"] == "${{ secrets.RENDER_DEPLOY_HOOK_URL }}"
        assert deploy["env"]["RENDER_SERVICE_URL"] == "${{ vars.RENDER_SERVICE_URL }}"
        for name, job in jobs.items():
            if name != "deploy":
                assert "RENDER_DEPLOY_HOOK_URL" not in json.dumps(job), name

    def test_the_hook_is_never_printed(self, jobs):
        """Κάθε γραμμή που αγγίζει το hook είναι ανάθεση, έλεγχος, mask ή η κλήση του curl."""
        script = find_step(jobs["deploy"], "Trigger the Render deploy hook")["run"]
        mention = re.compile(r"RENDER_DEPLOY_HOOK_URL|\$\{?hook\}?(?![A-Za-z_])")
        for line in script.splitlines():
            if not mention.search(line):
                continue
            assert not re.search(r"\b(?:echo|printf|cat|tee|env|set -x)\b", line) or (
                "::add-mask::" in line
            ), line
        assert script.count("::add-mask::") >= 2  # και το URL και το URL με το ref
        assert "set -x" not in script and "curl -v" not in script and " --verbose" not in script


# --------------------------------------------------------------------------------------
# Τα jobs
# --------------------------------------------------------------------------------------


class TestPythonAndDependencies:
    def test_every_python_job_uses_the_same_version_from_one_place(self, workflow, jobs):
        assert workflow["env"]["PYTHON_VERSION"] == "3.12"
        for job in jobs.values():
            for step in steps_of(job):
                if step.get("uses", "").startswith("actions/setup-python@"):
                    assert step["with"]["python-version"] == "${{ env.PYTHON_VERSION }}"

    @pytest.mark.parametrize("name", ["lint", "test", "quality-gate"])
    def test_pip_is_cached_by_the_dependency_files(self, jobs, name):
        setup = next(s for s in steps_of(jobs[name]) if "setup-python" in s.get("uses", ""))
        assert setup["with"]["cache"] == "pip"
        assert "constraints.txt" in setup["with"]["cache-dependency-path"]

    @pytest.mark.parametrize("name", ["test", "quality-gate"])
    def test_dependencies_are_installed_with_the_constraints_and_the_package_without_deps(
        self, jobs, name
    ):
        script = find_step(jobs[name], "Install dependencies")["run"]
        assert "pip install -r requirements-dev.txt -c constraints.txt" in script
        assert "pip install --no-deps -e ." in script

    def test_lint_installs_only_ruff_with_the_pinned_version(self, jobs):
        script = find_step(jobs["lint"], "Install ruff")["run"]
        assert "pip install -c constraints.txt ruff" in script
        assert "requirements" not in script


class TestLintJob:
    def test_it_runs_ruff_check_and_ruff_format_check(self, jobs):
        scripts = run_scripts(jobs["lint"])
        assert "ruff check ." in scripts
        assert "ruff format --check ." in scripts


class TestTestJob:
    def test_a_real_postgres_service_container_is_started_and_awaited(self, jobs):
        postgres = jobs["test"]["services"]["postgres"]
        assert postgres["image"] == "postgres:17"
        assert postgres["ports"] == ["5432:5432"]
        assert "--health-cmd" in postgres["options"] and "pg_isready" in postgres["options"]

    def test_the_postgres_tests_cannot_be_lost_silently(self, jobs):
        env = jobs["test"]["env"]
        assert env["TEST_DATABASE_URL"] == "postgresql://postgres:postgres@localhost:5432/postgres"
        assert env["ELFANTASY_REQUIRE_POSTGRES"] == "1"
        assert env["ELFANTASY_PG_DISPOSABLE"] == "1"  # ώστε να τρέχουν και και τα 42 tests

    def test_it_runs_unit_and_integration_tests_with_a_coverage_gate(self, jobs):
        script = find_step(jobs["test"], "Run the tests")["run"]
        for option in (
            "--ignore=tests/quality",
            "--cov=elfantasy",
            "--cov-report=xml",
            "--cov-report=term-missing",
            "--cov-fail-under=95",
        ):
            assert option in script
        assert script.lstrip().startswith("python -m pytest")

    def test_the_coverage_percentage_is_a_job_output_and_the_xml_an_artifact(self, jobs):
        job = jobs["test"]
        assert job["outputs"] == {"coverage": "${{ steps.coverage.outputs.percent }}"}
        step = find_step(job, "Compute the coverage percentage")
        assert step["id"] == "coverage"
        assert "make_coverage_badge.py --xml coverage.xml --print-percent" in step["run"]
        assert "GITHUB_OUTPUT" in step["run"]
        upload = next(s for s in steps_of(job) if "upload-artifact" in s.get("uses", ""))
        assert upload["with"]["name"] == "coverage-xml"
        assert upload["with"]["path"] == "coverage.xml"
        assert upload["if"] == "${{ !cancelled() }}"


class TestQualityGateJob:
    def test_it_runs_the_quality_tests_and_prints_the_numbers_to_the_summary(self, jobs):
        job = jobs["quality-gate"]
        assert find_step(job, "Model quality gate")["run"] == "python -m pytest tests/quality -q"
        summary = find_step(job, "Quality numbers")
        assert summary["run"] == 'python scripts/quality_summary.py >> "$GITHUB_STEP_SUMMARY"'
        assert summary["if"] == "${{ !cancelled() }}"  # φαίνεται και όταν το gate αποτυγχάνει

    def test_it_needs_no_secrets_database_or_dotenv(self, jobs):
        text = json.dumps(jobs["quality-gate"])
        for forbidden in ("secrets.", "DATABASE_URL", ".env", "services"):
            assert forbidden not in text


class TestDeployJob:
    def test_it_uses_the_production_environment(self, jobs):
        assert jobs["deploy"]["environment"] == "production"

    def test_a_missing_secret_is_a_clear_message_and_not_a_failure(self, jobs):
        step = find_step(jobs["deploy"], "Deploy skipped")
        assert step["if"] == "env.RENDER_DEPLOY_HOOK_URL == ''"
        assert "deploy skipped: secret not configured" in step["run"]
        assert "GITHUB_STEP_SUMMARY" in step["run"]
        assert "exit 1" not in step["run"]

    def test_the_other_steps_run_only_when_the_secret_exists(self, jobs):
        steps = steps_of(jobs["deploy"])
        skip, *rest = steps
        assert skip["if"] == "env.RENDER_DEPLOY_HOOK_URL == ''"
        for step in rest:
            assert "env.RENDER_DEPLOY_HOOK_URL != ''" in step["if"], step.get("name")

    def test_the_hook_is_posted_with_curl_and_retries(self, jobs):
        script = find_step(jobs["deploy"], "Trigger the Render deploy hook")["run"]
        assert "curl --fail-with-body -sS -X POST" in script
        assert "attempts=3" in script
        assert "ref=${GITHUB_SHA}" in script  # deploy του commit που πέρασε τους ελέγχους
        assert "exit 1" in script  # αποτυχία μετά τις προσπάθειες
        assert "4??) break" in script  # τα σφάλματα του client (401, 404, …) δεν επαναλαμβάνονται

    def test_the_smoke_check_waits_for_the_new_commit_and_is_skipped_without_the_variable(
        self, jobs
    ):
        deploy = jobs["deploy"]
        check = find_step(deploy, "Post-deploy smoke check")
        assert check["if"] == (
            "env.RENDER_DEPLOY_HOOK_URL != '' && env.RENDER_SERVICE_URL != '' "
            "&& steps.guard.outputs.superseded != 'true'"
        )
        command = " ".join(check["run"].split())
        assert 'python scripts/smoke_check.py --url "${RENDER_SERVICE_URL}"' in command
        assert '--expect-commit "${GITHUB_SHA}"' in command
        assert "--timeout-seconds 900" in command
        skipped = find_step(deploy, "Smoke check skipped")
        assert skipped["if"] == (
            "env.RENDER_DEPLOY_HOOK_URL != '' && env.RENDER_SERVICE_URL == '' "
            "&& steps.guard.outputs.superseded != 'true'"
        )
        assert "exit 1" not in skipped["run"]

    def test_a_superseded_commit_does_not_deploy(self, jobs):
        """Δύο διαδοχικά pushes στο main: αν το παλαιότερο τελειώσει τους ελέγχους μετά το νεότερο,
        δεν πρέπει να κάνει deploy πάνω από αυτό (εύρημα m6 του review της Φάσης 7)."""
        deploy = jobs["deploy"]
        names = [step.get("name", "") for step in steps_of(deploy)]
        guard = find_step(deploy, "Check that this commit is still the head of main")
        assert guard["id"] == "guard"
        assert guard["if"] == "env.RENDER_DEPLOY_HOOK_URL != ''"
        script = guard["run"]
        assert "git ls-remote" in script and "refs/heads/main" in script
        assert '"${GITHUB_SHA}"' in script and "superseded=true" in script
        # Αν το main δεν διαβάζεται, το deploy συνεχίζει (ο φύλακας δεν είναι δικλείδα ασφαλείας).
        assert "superseded=false" in script and "exit 1" not in script
        # Ο φύλακας έρχεται πριν από το Trigger και κρατά και τα δύο επόμενα steps εκτός.
        assert names.index(guard["name"]) < names.index("Trigger the Render deploy hook")
        trigger = find_step(deploy, "Trigger the Render deploy hook")
        assert trigger["if"] == (
            "env.RENDER_DEPLOY_HOOK_URL != '' && steps.guard.outputs.superseded != 'true'"
        )
        notice = find_step(deploy, "Deploy skipped (superseded")
        assert notice["if"] == (
            "env.RENDER_DEPLOY_HOOK_URL != '' && steps.guard.outputs.superseded == 'true'"
        )
        assert "exit 1" not in notice["run"]  # ένα ξεπερασμένο commit δεν είναι αποτυχία

    def test_the_job_timeout_covers_the_retries_and_the_smoke_check(self, jobs):
        assert jobs["deploy"]["timeout-minutes"] >= 25  # 3 × 60 s + αναμονές + 15 λεπτά smoke check


class TestCoverageBadgeJob:
    def test_it_publishes_to_the_badges_branch_with_the_bot_identity(self, jobs):
        publish = find_step(jobs["coverage-badge"], "Publish the badge")["run"]
        assert 'git config user.name "github-actions[bot]"' in publish
        assert "users.noreply.github.com" in publish
        assert "refs/heads/badges" in publish and "git push origin badges" in publish
        assert "--orphan badges" in publish  # δημιουργείται αν δεν υπάρχει
        assert "git diff --cached --quiet" in publish  # idempotent: κανένα commit αν δεν άλλαξε
        assert "git push origin main" not in publish and "origin HEAD" not in publish

    def test_it_generates_the_svg_from_the_output_of_the_test_job(self, jobs):
        step = find_step(jobs["coverage-badge"], "Generate the badge")
        assert step["env"]["COVERAGE"] == "${{ needs.test.outputs.coverage }}"
        assert "scripts/make_coverage_badge.py" in step["run"]
        assert "coverage.svg" in step["run"]


# --------------------------------------------------------------------------------------
# render.yaml
# --------------------------------------------------------------------------------------


class TestRenderBlueprint:
    def test_one_python_web_service_in_frankfurt_on_the_free_plan(self, render, service):
        assert list(render) == ["services"]
        assert service["type"] == "web"
        assert service["name"] == "euroleague-fantasy-api"
        assert service["runtime"] == "python"
        assert service["region"] == "frankfurt"
        assert service["plan"] == "free"
        assert service["branch"] == "main"

    def test_auto_deploy_is_disabled_because_the_ci_deploys(self, service):
        # Το `autoDeploy: false` (παλαιότερο) ή το `autoDeployTrigger: off` (σημερινό)· το «off»
        # σε εισαγωγικά, γιατί το YAML 1.1 θα το διάβαζε ως false.
        assert service.get("autoDeploy") is False or service.get("autoDeployTrigger") == "off"
        assert service.get("autoDeployTrigger", "off") == "off"
        assert service.get("autoDeploy", False) is False

    def test_the_health_check_is_the_health_endpoint(self, service):
        assert service["healthCheckPath"] == "/health"

    def test_the_build_installs_with_the_constraints_and_then_the_package(self, service):
        command = service["buildCommand"]
        assert command == (
            "pip install -r requirements.txt -c constraints.txt && pip install --no-deps ."
        )
        for name in ("requirements.txt", "constraints.txt", "pyproject.toml"):
            assert (ROOT / name).is_file(), name

    def test_the_start_command_runs_one_uvicorn_worker_on_the_render_port(self, service):
        command = service["startCommand"].split()
        assert command[:2] == ["uvicorn", "elfantasy.api.main:app"]
        assert command[command.index("--host") + 1] == "0.0.0.0"
        assert command[command.index("--port") + 1] == "$PORT"
        assert command[command.index("--workers") + 1] == "1"

    def test_the_secrets_are_asked_in_the_dashboard_and_never_stored(self, service):
        env = {item["key"]: item for item in service["envVars"]}
        for name in ("DATABASE_URL", "ADMIN_API_KEY"):
            assert env[name] == {"key": name, "sync": False}, name

    def test_nothing_that_looks_like_a_secret_has_a_value(self, service):
        for item in service["envVars"]:
            if re.search(r"KEY|SECRET|PASSWORD|TOKEN|URL|HOOK", item["key"]):
                assert "value" not in item, item["key"]
                assert item.get("sync") is False, item["key"]

    def test_every_value_is_a_string(self, service):
        for item in service["envVars"]:
            assert isinstance(item.get("value", ""), str), item["key"]

    def test_the_model_path_is_the_default_of_the_settings(self, service):
        env = {item["key"]: item.get("value") for item in service["envVars"]}
        assert env["MODEL_PATH"] == Settings.model_fields["model_path"].default
        assert (ROOT / env["MODEL_PATH"]).is_file()

    def test_a_single_worker_is_requested_twice(self, service):
        env = {item["key"]: item.get("value") for item in service["envVars"]}
        assert env["WEB_CONCURRENCY"] == "1"

    def test_the_memory_settings_for_the_512_mb_plan(self, service):
        env = {item["key"]: item.get("value") for item in service["envVars"]}
        assert env["MALLOC_ARENA_MAX"] == "2"
        assert env["OMP_NUM_THREADS"] == "1"

    def test_the_file_contains_no_secret(self):
        text = RENDER_PATH.read_text(encoding="utf-8")
        for label, pattern in SECRET_PATTERNS.items():
            assert not pattern.search(text), f"{label} found in render.yaml"
        assert not URL_WITH_PASSWORD.search(text)


class TestPythonVersionsAgree:
    def test_render_and_ci_use_the_same_python_minor_version(self, workflow, service):
        env = {item["key"]: item.get("value") for item in service["envVars"]}
        full = env["PYTHON_VERSION"]
        assert re.fullmatch(r"\d+\.\d+\.\d+", full), "Render needs a fully qualified version"
        assert ".".join(full.split(".")[:2]) == workflow["env"]["PYTHON_VERSION"] == "3.12"

    def test_the_project_requires_python_312_or_newer(self):
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        requires = SpecifierSet(pyproject["project"]["requires-python"])
        assert "3.12.0" in requires and "3.13.2" in requires
        assert "3.11.9" not in requires
        assert pyproject["tool"]["ruff"]["target-version"] == "py312"


# --------------------------------------------------------------------------------------
# constraints.txt
# --------------------------------------------------------------------------------------


def read_pins() -> dict[str, tuple[str, str | None]]:
    """Όνομα -> (έκδοση, marker) από το constraints.txt."""
    pins: dict[str, tuple[str, str | None]] = {}
    for line in CONSTRAINTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        specifiers = list(requirement.specifier)
        assert len(specifiers) == 1 and specifiers[0].operator == "==", line
        name = canonicalize_name(requirement.name)
        assert name not in pins, f"{name} is pinned twice"
        marker = str(requirement.marker) if requirement.marker else None
        pins[name] = (specifiers[0].version, marker)
    return pins


def requirement_names(filename: str) -> set[str]:
    names = set()
    for line in (ROOT / filename).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        requirement = Requirement(line)
        names.add(canonicalize_name(requirement.name))
        if requirement.name == "psycopg":  # το extra [binary] φέρνει το psycopg-binary
            names.add("psycopg-binary")
    return names


class TestConstraints:
    def test_every_line_is_an_exact_pin(self):
        assert len(read_pins()) > 40

    def test_every_direct_requirement_is_pinned(self):
        pins = read_pins()
        for filename in ("requirements.txt", "requirements-dev.txt"):
            missing = requirement_names(filename) - set(pins)
            assert not missing, f"{filename}: no pin for {sorted(missing)}"

    def test_the_pins_are_the_versions_the_committed_model_was_trained_with(self):
        metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
        pins = read_pins()
        for name, version in metrics["library_versions"].items():
            if name == "python":
                continue
            assert pins[canonicalize_name(name)][0] == version, name

    def test_the_pins_satisfy_the_minimum_versions_of_the_requirements(self):
        pins = read_pins()
        for filename in ("requirements.txt", "requirements-dev.txt"):
            for line in (ROOT / filename).read_text(encoding="utf-8").splitlines():
                line = line.split("#")[0].strip()
                if not line or line.startswith("-"):
                    continue
                requirement = Requirement(line)
                version = pins[canonicalize_name(requirement.name)][0]
                assert version in requirement.specifier, f"{line} vs pinned {version}"

    def test_ruff_is_pinned_for_the_lint_job(self):
        assert "ruff" in read_pins()

    def test_only_linux_only_packages_have_a_marker(self):
        markers = {name: marker for name, (_, marker) in read_pins().items() if marker}
        assert markers == {"nvidia-nccl-cu13": 'sys_platform == "linux"'}

    def test_windows_only_packages_are_left_out(self):
        assert "colorama" not in read_pins()
        assert "elfantasy" not in read_pins()  # το ίδιο το πακέτο (editable)


# --------------------------------------------------------------------------------------
# README και docs/DEPLOY.md
# --------------------------------------------------------------------------------------


class TestDocumentation:
    def test_the_readme_has_the_build_and_coverage_badges_and_the_full_instructions(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        repo = "tmartsoukos/EuroleagueFantasyAPI"
        assert f"https://github.com/{repo}/actions/workflows/ci.yml/badge.svg?branch=main" in text
        assert f"https://raw.githubusercontent.com/{repo}/badges/coverage.svg" in text
        assert text.lstrip().startswith("# Euroleague Fantasy Points Predictor API")
        # Το τελικό README (Φάση 7) έχει οδηγίες setup, run, test και deploy· όχι πια το stub.
        assert "Το πλήρες README" not in text
        for heading in (
            "## Γρήγορη εκκίνηση (setup)",
            "## Χρήση του API",
            "## Tests και ποιότητα",
            "## CI/CD και deploy στο Render",
        ):
            assert heading in text, heading
        for command in (
            "pip install -r requirements-dev.txt -c constraints.txt",
            "pytest -q",
            "uvicorn elfantasy.api.main:app",
            "python -m elfantasy.ingest.pipeline",
        ):
            assert command in text, command

    def test_the_deploy_guide_covers_every_step_the_user_must_take(self):
        text = (ROOT / "docs" / "DEPLOY.md").read_text(encoding="utf-8")
        for needle in (
            "render.yaml",
            "Session pooler",
            "sslmode=require",
            "IPv6",
            "ADMIN_API_KEY",
            "secrets.token_urlsafe(32)",
            "gh secret set RENDER_DEPLOY_HOOK_URL --repo tmartsoukos/EuroleagueFantasyAPI",
            "RENDER_SERVICE_URL",
            "Rollback",
            "elfantasy.model.train",
            "elfantasy.ingest.pipeline --update",
            "/admin/refresh",
            "15 λεπτά",
            "512 MB",
        ):
            assert needle in text, needle
        assert "προστασία του `main` (δεν εφαρμόζεται)" in text  # μόνο πρόταση

    def test_the_deploy_guide_contains_no_real_hook_or_service_secret(self):
        text = (ROOT / "docs" / "DEPLOY.md").read_text(encoding="utf-8")
        assert not re.search(r"[?&]key=[A-Za-z0-9_-]{6,}", text)  # κανένα πραγματικό hook
        assert not re.search(r"srv-[a-z0-9]{10,}", text)

    def test_no_text_file_still_claims_python_311_support(self):
        for path in (ROOT / "pyproject.toml", ROOT / "requirements.txt", ROOT / "README.md"):
            assert 'requires-python = ">=3.11"' not in path.read_text(encoding="utf-8")
        for path in (ROOT / "docs").glob("*.md"):
            text = path.read_text(encoding="utf-8")
            assert "ικανοποιεί το «3.11+»" not in text, path.name
