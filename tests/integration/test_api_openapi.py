"""Tests της τεκμηρίωσης: `/openapi.json`, `/docs`, security scheme, τεκμηριωμένοι κωδικοί
απάντησης, παραδείγματα και απουσία μυστικών από το schema.
"""

from __future__ import annotations

import json
import re

import pytest
from api_support import ADMIN_KEY

from elfantasy.api import schemas

# Για κάθε λειτουργία: οι κωδικοί HTTP που πρέπει να είναι τεκμηριωμένοι.
EXPECTED = {
    ("get", "/health"): {"200", "503"},
    ("get", "/predict/{player_id}"): {"200", "404", "422", "503"},
    ("get", "/rankings"): {"200", "404", "422", "503"},
    ("get", "/players"): {"200", "404", "422", "503"},
    ("get", "/availability"): {"200", "422", "503"},
    ("post", "/availability"): {"200", "401", "404", "422", "503"},
    ("delete", "/availability/{player_id}"): {"204", "401", "404", "422", "503"},
    ("post", "/admin/refresh"): {"200", "401", "503"},
}
PROTECTED = {
    ("post", "/availability"),
    ("delete", "/availability/{player_id}"),
    ("post", "/admin/refresh"),
}
GREEK = re.compile("[Α-Ωα-ωά-ώ]")


@pytest.fixture
def spec(client) -> dict:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


def operations(spec):
    for path, item in spec["paths"].items():
        for method, operation in item.items():
            yield (method, path), operation


def resolve(spec: dict, ref: str) -> dict:
    assert ref.startswith("#/"), ref
    node = spec
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def collect_refs(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref":
                yield value
            else:
                yield from collect_refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from collect_refs(value)


class TestOpenApiDocument:
    def test_the_document_is_served_and_well_formed(self, client, spec):
        assert client.get("/openapi.json").headers["content-type"].startswith("application/json")
        assert spec["openapi"].startswith("3.")
        assert spec["info"]["title"] == "Euroleague Fantasy Points Predictor API"
        assert spec["info"]["version"]
        assert set(spec) >= {"openapi", "info", "paths", "components", "tags"}
        for ref in set(collect_refs(spec)):
            resolve(spec, ref)  # κάθε $ref δείχνει σε υπαρκτό στοιχείο

    def test_the_description_explains_how_to_read_the_predictions(self, spec):
        description = spec["info"]["description"]
        assert GREEK.search(description)
        for phrase in ("τυπική", "captain", "πάγκο", "τραυματισμούς", "X-API-Key", "predicted_pir"):
            assert phrase in description, phrase
        assert spec["info"]["summary"]

    def test_exactly_the_expected_operations_exist(self, spec):
        assert {key for key, _ in operations(spec)} == set(EXPECTED)

    @pytest.mark.parametrize(("method", "path"), sorted(EXPECTED))
    def test_every_operation_is_documented_in_greek(self, spec, method, path):
        operation = spec["paths"][path][method]
        assert operation["summary"] and GREEK.search(operation["summary"])
        assert operation["description"] and GREEK.search(operation["description"])
        assert operation["tags"]
        assert operation["operationId"]

    @pytest.mark.parametrize(("method", "path"), sorted(EXPECTED))
    def test_the_status_codes_are_documented(self, spec, method, path):
        operation = spec["paths"][path][method]
        assert set(operation["responses"]) == EXPECTED[(method, path)]
        for code, response in operation["responses"].items():
            assert response["description"]
            if code != "422":  # το 422 είναι η προεπιλεγμένη τεκμηρίωση του FastAPI
                assert GREEK.search(response["description"]), (code, response["description"])
            if code in {"401", "404", "503"} and (method, path) != ("get", "/health"):
                schema = response["content"]["application/json"]["schema"]
                assert schema["$ref"].endswith("/ErrorOut"), (code, schema)

    def test_the_health_503_documents_the_degraded_body(self, spec):
        responses = spec["paths"]["/health"]["get"]["responses"]
        assert responses["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
            "/HealthResponse"
        )
        assert responses["503"]["content"]["application/json"]["schema"]["$ref"].endswith(
            "/HealthResponse"
        )

    def test_the_tags_have_descriptions(self, spec):
        tags = {tag["name"]: tag["description"] for tag in spec["tags"]}
        assert set(tags) == {"health", "predictions", "players", "availability", "admin"}
        assert all(tags.values())
        used = {tag for _, operation in operations(spec) for tag in operation["tags"]}
        assert used == set(tags)

    def test_the_delete_response_has_no_body(self, spec):
        assert (
            "content"
            not in spec["paths"]["/availability/{player_id}"]["delete"]["responses"]["204"]
        )


class TestSecurity:
    def test_the_api_key_scheme_is_declared(self, spec):
        scheme = spec["components"]["securitySchemes"]["APIKeyHeader"]
        assert scheme["type"] == "apiKey"
        assert scheme["in"] == "header"
        assert scheme["name"] == "X-API-Key"
        assert GREEK.search(scheme["description"])
        assert set(spec["components"]["securitySchemes"]) == {"APIKeyHeader"}

    @pytest.mark.parametrize(("method", "path"), sorted(EXPECTED))
    def test_only_the_admin_operations_require_the_key(self, spec, method, path):
        operation = spec["paths"][path][method]
        if (method, path) in PROTECTED:
            assert operation["security"] == [{"APIKeyHeader": []}]
        else:
            assert "security" not in operation

    def test_no_secret_appears_anywhere_in_the_schema(self, spec, client):
        text = json.dumps(spec)
        assert ADMIN_KEY not in text
        assert "change-me" not in text.lower()
        for example in re.findall(r'"X-API-Key": "[^"]*"', text):
            raise AssertionError(f"a header value appears in the schema: {example}")
        assert ADMIN_KEY not in client.get("/docs").text


class TestParametersAndSchemas:
    @staticmethod
    def parameters(spec, path, method="get"):
        return {p["name"]: p for p in spec["paths"][path][method]["parameters"]}

    def test_the_rankings_parameters(self, spec):
        parameters = self.parameters(spec, "/rankings")
        assert set(parameters) == {"limit", "offset", "team", "include_unavailable", "active_only"}
        limit = parameters["limit"]["schema"]
        assert (limit["minimum"], limit["maximum"], limit["default"]) == (1, 500, 50)
        offset = parameters["offset"]["schema"]
        assert (offset["minimum"], offset["default"]) == (0, 0)
        assert parameters["include_unavailable"]["schema"]["default"] is False
        assert parameters["active_only"]["schema"]["default"] is True
        assert all(GREEK.search(p["description"]) for p in parameters.values())

    def test_the_players_parameters(self, spec):
        parameters = self.parameters(spec, "/players")
        assert set(parameters) == {"search", "team", "limit"}
        limit = parameters["limit"]["schema"]
        assert (limit["minimum"], limit["maximum"], limit["default"]) == (1, 100, 20)

    def test_the_player_id_path_parameter(self, spec):
        parameters = self.parameters(spec, "/predict/{player_id}")
        player_id = parameters["player_id"]
        assert player_id["in"] == "path" and player_id["required"] is True
        assert "P007200" in player_id["description"] and "PADF" in player_id["description"]
        assert parameters["include_features"]["schema"]["default"] is False

    def test_the_availability_filter_lists_the_statuses(self, spec):
        parameters = self.parameters(spec, "/availability")
        schema = parameters["status"]["schema"]
        values = {v for branch in schema["anyOf"] for v in resolve_enum(spec, branch)}
        assert values == {"out", "doubtful", "available"}

    def test_the_request_body_of_the_post(self, spec):
        body = spec["paths"]["/availability"]["post"]["requestBody"]["content"]["application/json"]
        schema = resolve(spec, body["schema"]["$ref"])
        assert set(schema["properties"]) == {
            "player_id",
            "status",
            "source",
            "note",
            "expected_return",
        }
        assert set(schema["required"]) == {"player_id", "status"}
        assert schema["additionalProperties"] is False
        note = schema["properties"]["note"]
        assert any(branch.get("maxLength") == 500 for branch in note["anyOf"])

    def test_the_features_of_the_prediction_are_optional(self, spec):
        schema = spec["components"]["schemas"]["PredictionOut"]
        assert "features" in schema["properties"] and "features" not in schema["required"]
        assert {
            "player_id",
            "name",
            "team_code",
            "team_name",
            "is_active",
            "next_game",
            "predicted_fantasy",
            "model_predicted_fantasy",
            "predicted_pir",
            "availability",
            "n_prior_appearances",
            "last_appearance_date",
            "model_version",
            "notes",
        } <= set(schema["required"])

    @pytest.mark.parametrize(
        "model",
        [
            schemas.PredictionOut,
            schemas.RankingsOut,
            schemas.PlayersOut,
            schemas.AvailabilityIn,
            schemas.AvailabilityEntry,
            schemas.HealthResponse,
            schemas.ErrorOut,
        ],
    )
    def test_the_examples_of_the_models_are_valid(self, model):
        examples = model.model_json_schema()["examples"]
        assert examples
        for example in examples:
            model.model_validate(example)

    def test_the_examples_are_in_the_openapi_schemas(self, spec):
        for name in (
            "PredictionOut",
            "RankingsOut",
            "PlayersOut",
            "AvailabilityIn",
            "HealthResponse",
        ):
            assert spec["components"]["schemas"][name]["examples"], name

    def test_the_field_descriptions_are_in_greek(self, spec):
        properties = spec["components"]["schemas"]["PredictionOut"]["properties"]
        for name in ("predicted_fantasy", "model_predicted_fantasy", "predicted_pir", "is_active"):
            assert GREEK.search(properties[name]["description"]), name
        assert "captain" in properties["predicted_fantasy"]["description"]
        assert "ακατέργαστη" in properties["predicted_pir"]["description"]


def resolve_enum(spec, branch):
    if "$ref" in branch:
        return resolve(spec, branch["$ref"])["enum"]
    return branch.get("enum", [])


class TestDocsPages:
    def test_swagger_ui_is_served(self, client):
        response = client.get("/docs")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "swagger" in response.text.lower()
        assert "/openapi.json" in response.text

    def test_redoc_is_served(self, client):
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "redoc" in response.text.lower()

    def test_the_root_redirects_to_the_docs_and_is_not_in_the_schema(self, client, spec):
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/docs"
        assert "/" not in spec["paths"]
