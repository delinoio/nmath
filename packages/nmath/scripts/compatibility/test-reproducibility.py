# SPDX-License-Identifier: Apache-2.0
"""Regression tests for committed oracle fixture comparison."""
import contextlib
import copy
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("reproducibility", SCRIPTS / "check-reproducibility.py")
reproducibility = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproducibility)

ENVIRONMENT = {"python": "3.12.14", "system": "Darwin", "machine": "arm64", "byteorder": "little"}
FIXTURE = {
    "versions": {"numpy": "2.5.4"},
    "sources": {"numpy": {"commit": "original"}},
    "criteria": {"relative": 0.001},
    "cases": [{"id": "one", "result": 1.0}],
    "environment": ENVIRONMENT,
}


def artifact(value):
    return (json.dumps(value, indent=2) + "\n").encode()


class CommittedFixtureComparison(unittest.TestCase):
    def setUp(self):
        self.expected = {path: artifact(FIXTURE) for path in reproducibility.FIXTURES}
        self.expected["inventory/numpy.jsonl"] = b'{"name":"numpy.add"}\n'

    def compare(self, fresh):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            differences = reproducibility.committed_differences(self.expected, fresh)
        return differences, output.getvalue()

    def test_each_valid_host_field_can_change_in_each_fixture(self):
        for path in reproducibility.FIXTURES:
            for field, replacement in (("python", "3.13.1"), ("system", "Linux"),
                                       ("machine", "x86_64"), ("byteorder", "big")):
                with self.subTest(path=path, field=field):
                    changed = copy.deepcopy(FIXTURE)
                    changed["environment"][field] = replacement
                    fresh = dict(self.expected)
                    fresh[path] = artifact(changed)
                    differences, output = self.compare(fresh)
                    self.assertEqual(differences, [])
                    self.assertIn(f"Committed {path} environment:", output)
                    self.assertIn(f"Fresh {path} environment:", output)
                    self.assertIn(replacement, output)

    def test_deterministic_fields_remain_strict(self):
        changes = (
            ("result", lambda value: value["cases"][0].update(result=1.5)),
            ("numeric type", lambda value: value["cases"][0].update(result=1)),
            ("case", lambda value: value["cases"][0].update(id="other")),
            ("criterion", lambda value: value["criteria"].update(relative=0.002)),
            ("version", lambda value: value["versions"].update(numpy="2.5.5")),
            ("source", lambda value: value["sources"]["numpy"].update(commit="other")),
        )
        for name, change in changes:
            with self.subTest(name=name):
                value = copy.deepcopy(FIXTURE)
                change(value)
                fresh = dict(self.expected)
                fresh["fixtures/random.json"] = artifact(value)
                self.assertEqual(self.compare(fresh)[0], ["fixtures/random.json"])

    def test_invalid_provenance_has_a_specific_diagnostic(self):
        for name, change, diagnostic in (
            ("missing", lambda value: value.pop("environment"), "environment must be an object"),
            ("nonobject", lambda value: value.update(environment=[]), "environment must be an object"),
            ("missing field", lambda value: value["environment"].pop("python"), "environment.python"),
            ("empty field", lambda value: value["environment"].update(system=" "), "environment.system"),
            ("byteorder", lambda value: value["environment"].update(byteorder="middle"), "environment.byteorder"),
        ):
            with self.subTest(name=name):
                value = copy.deepcopy(FIXTURE)
                change(value)
                fresh = dict(self.expected)
                fresh["fixtures/mutation.json"] = artifact(value)
                with self.assertRaisesRegex(ValueError, diagnostic):
                    self.compare(fresh)

    def test_committed_provenance_is_validated_too(self):
        self.expected["fixtures/primitive.json"] = artifact({**FIXTURE, "environment": None})
        with self.assertRaisesRegex(ValueError, "Committed fixtures/primitive.json: environment"):
            self.compare(dict(self.expected))

    def test_inventory_and_file_sets_remain_exact(self):
        for change in (
            lambda fresh: fresh.update({"inventory/numpy.jsonl": b"changed"}),
            lambda fresh: fresh.pop("fixtures/random.json"),
            lambda fresh: fresh.update({"fixtures/extra.json": b"{}"}),
        ):
            fresh = dict(self.expected)
            change(fresh)
            self.assertTrue(self.compare(fresh)[0])

    def test_fresh_generations_remain_byte_compared_before_provenance_comparison(self):
        with mock.patch.object(reproducibility, "files", side_effect=[{"a": b"one"}, {"a": b"two"}]), \
             mock.patch.object(reproducibility.subprocess, "run"), \
             mock.patch.object(sys, "argv", ["check-reproducibility.py", "--numpy-source", "n",
                                            "--pandas-source", "p", "--polars-source", "l"]):
            with self.assertRaisesRegex(SystemExit, "Fresh generations differ"):
                reproducibility.main()


if __name__ == "__main__":
    unittest.main()
