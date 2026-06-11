from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any


def _install_parser_services_stub() -> None:
    if 'parser.services' in sys.modules:
        return
    module = types.ModuleType('parser.services')

    def parse_repo(*_args: Any, **_kwargs: Any) -> dict[str, list[Any]]:
        return {'tree': [], 'nodes': [], 'edges': []}

    module.parse_repo = parse_repo
    sys.modules['parser.services'] = module


_install_parser_services_stub()

from api.artifacts import build_graph_artifact
from harness_eval.swebench import diff_labels, evaluator, importer, runner, sample_builder


SOURCE_PATCH = """diff --git a/pkg/core.py b/pkg/core.py
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1,2 +1,3 @@
 def buggy():
-    return None
+    return 1
+    return 2
"""


TEST_PATCH = """diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -4,1 +4,1 @@
-assert buggy() is None
+assert buggy() == 1
"""


DOCS_PATCH = """diff --git a/docs/example.py b/docs/example.py
--- a/docs/example.py
+++ b/docs/example.py
@@ -1,1 +1,1 @@
-buggy()
+fixed()
"""


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        'instance_id': 'owner__repo-42',
        'repo': 'owner/repo',
        'base_commit': 'abc123',
        'patch': SOURCE_PATCH,
        'test_patch': TEST_PATCH,
        'problem_statement': 'Core bug in pkg/core.py when calling buggy().',
        'issue_url': 'https://github.com/owner/repo/issues/42',
        'pr_url': 'https://github.com/owner/repo/pull/43',
        'FAIL_TO_PASS': ['tests/test_core.py::test_buggy'],
    }
    row.update(overrides)
    return row


def _sample() -> dict[str, Any]:
    sample, skip = importer.build_sample_from_row(_row())
    assert sample is not None
    assert skip is None
    sample['patch'] = 'hidden patch text'
    sample['test_patch'] = 'hidden test patch text'
    return sample


def _artifact() -> dict[str, Any]:
    return build_graph_artifact(
        repo_path='owner/repo',
        revision='abc123',
        graph={
            'tree': [{'path': 'pkg/core.py', 'type': 'file'}],
            'nodes': [
                {
                    'id': 'pkg/core.py::buggy',
                    'type': 'function',
                    'label': 'buggy',
                    'path': 'pkg/core.py',
                    'start_line': 1,
                    'end_line': 3,
                }
            ],
            'edges': [],
        },
        file_contents={
            'pkg/core.py': 'def buggy():\n    return 1\n',
            'pkg/other.py': 'def other():\n    return 0\n',
        },
        languages=['python'],
        file_manifest={
            'pkg/core.py': {'language': 'python', 'language_family': 'python', 'support_level': 'relationships', 'byte_size': 27},
            'pkg/other.py': {'language': 'python', 'language_family': 'python', 'support_level': 'relationships', 'byte_size': 26},
        },
    )


def _passing_transcript(path: str = 'pkg/core.py') -> dict[str, Any]:
    return {
        'variant_id': 'local',
        'tool_calls': [
            {'name': 'get_issue_context'},
            {'name': 'list_repo_files'},
            {'name': 'search_repo_symbols'},
            {'name': 'read_repo_file'},
        ],
        'final': {
            'hypotheses': [{'path': path, 'node_id': f'{path}::buggy'}],
            'investigation_path': [{'path': path}],
        },
    }


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = set(value)
        for item in value.values():
            found.update(_keys(item))
        return found
    if isinstance(value, list):
        found: set[str] = set()
        for item in value:
            found.update(_keys(item))
        return found
    return set()


class DiffLabelsTests(unittest.TestCase):
    def test_source_diff_extraction(self) -> None:
        labels = diff_labels.labels_from_patch(SOURCE_PATCH)

        self.assertEqual(labels.gold_source_files, ['pkg/core.py'])
        self.assertEqual(len(labels.gold_hunks), 1)
        self.assertEqual(
            labels.gold_hunks[0].as_dict(),
            {'path': 'pkg/core.py', 'old_start': 1, 'old_length': 2, 'new_start': 1, 'new_length': 3},
        )

    def test_tests_and_docs_are_filtered_from_labels(self) -> None:
        labels = diff_labels.labels_from_patch(TEST_PATCH + DOCS_PATCH + SOURCE_PATCH)

        self.assertEqual(labels.gold_source_files, ['pkg/core.py'])
        self.assertIn({'path': 'tests/test_core.py', 'reason': 'test_path'}, labels.skipped_files)
        self.assertIn({'path': 'docs/example.py', 'reason': 'docs_or_examples'}, labels.skipped_files)


class ImporterTests(unittest.TestCase):
    def test_importer_ignores_test_patch_and_hidden_dataset_fields(self) -> None:
        sample, skip = importer.build_sample_from_row(_row())

        self.assertIsNone(skip)
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample['expect']['gold_source_files'], ['pkg/core.py'])
        self.assertNotIn('patch', sample)
        self.assertNotIn('test_patch', sample)
        self.assertNotIn('FAIL_TO_PASS', json.dumps(sample))
        self.assertNotIn('tests/test_core.py', json.dumps(sample['expect']))

    def test_import_rows_skips_samples_without_usable_source_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = importer.import_rows(
                [_row(patch=TEST_PATCH + DOCS_PATCH)],
                output_dir=root / 'samples',
                skip_report_path=root / 'skips.json',
            )

            self.assertEqual(report['written_count'], 0)
            self.assertEqual(report['skipped_count'], 1)
            self.assertEqual(json.loads((root / 'skips.json').read_text())['skips'][0]['reason'], 'no_source_labels')
            self.assertEqual(list((root / 'samples').glob('*.json')), [])


class SampleBuilderTests(unittest.TestCase):
    def test_visible_sample_packet_hides_hidden_fields(self) -> None:
        sample = _sample()

        visible = sample_builder.visible_sample_packet(sample)

        self.assertEqual(visible['job_id'], sample['id'])
        self.assertNotIn('expect', visible)
        self.assertFalse({'patch', 'test_patch', 'gold_source_files', 'gold_hunks'} & _keys(visible))
        self.assertNotIn('hidden patch text', json.dumps(visible))

    def test_runtime_job_hides_hidden_label_and_patch_fields(self) -> None:
        sample = _sample()

        job = sample_builder.build_runtime_job(sample, _artifact())

        self.assertFalse(importer.FORBIDDEN_MODEL_VISIBLE_FIELDS & _keys(job))
        rendered = json.dumps(job)
        self.assertNotIn('hidden patch text', rendered)
        self.assertNotIn('hidden test patch text', rendered)
        self.assertIn('pkg/core.py', job['file_contents'])


class EvaluatorTests(unittest.TestCase):
    def test_evaluator_passes_when_returned_path_overlaps_gold(self) -> None:
        report = evaluator.evaluate_transcript(_sample(), _passing_transcript())

        self.assertTrue(report['passed'])
        self.assertTrue(report['contract_pass'])
        self.assertEqual(report['matched_gold_files'], ['pkg/core.py'])
        self.assertIsNone(report['failure_reason'])

    def test_evaluator_fails_when_returned_path_is_not_gold(self) -> None:
        report = evaluator.evaluate_transcript(_sample(), _passing_transcript('pkg/other.py'))

        self.assertFalse(report['passed'])
        self.assertEqual(report['failure_reason'], evaluator.FAILURE_NO_GOLD_OVERLAP)
        self.assertEqual(report['returned_paths'], ['pkg/other.py'])

    def test_evaluator_resolves_node_id_path_through_artifact(self) -> None:
        transcript = {
            'variant_id': 'local',
            'tool_calls': [
                {'name': 'get_issue_context'},
                {'name': 'list_repo_files'},
                {'name': 'search_repo_text'},
                {'name': 'read_node_context'},
            ],
            'final': {
                'hypotheses': [{'node_id': 'pkg/core.py::buggy'}],
                'investigation_path': [],
            },
        }

        report = evaluator.evaluate_transcript(_sample(), transcript, artifact=_artifact())

        self.assertTrue(report['passed'])
        self.assertEqual(report['returned_paths'], ['pkg/core.py'])
        self.assertEqual(report['matched_gold_files'], ['pkg/core.py'])
        self.assertEqual(report['found_node_ids'], ['pkg/core.py::buggy'])

    def test_evaluator_reports_invalid_transcript_and_missing_final(self) -> None:
        invalid = evaluator.evaluate_transcript(_sample(), ['not', 'a', 'mapping'])
        missing_final = evaluator.evaluate_transcript(_sample(), {'tool_calls': []})

        self.assertEqual(invalid['failure_reason'], evaluator.FAILURE_INVALID_JSON)
        self.assertEqual(missing_final['failure_reason'], evaluator.FAILURE_MISSING_FINAL)


class ReportAggregationTests(unittest.TestCase):
    def test_summarize_results_counts_pass_rate_without_weights(self) -> None:
        results = [
            {'sample_id': 's1', 'variant_id': 'v1', 'model_id': 'm', 'passed': True, 'latency_ms': 10, 'cost_usd': 0.1, 'weight': 100},
            {
                'sample_id': 's2',
                'variant_id': 'v1',
                'model_id': 'm',
                'passed': False,
                'failure_reason': evaluator.FAILURE_NO_GOLD_OVERLAP,
                'latency_ms': 30,
                'cost_usd': 0.2,
                'weight': 1,
            },
            {'sample_id': 's1', 'variant_id': 'v2', 'model_id': 'm2', 'passed': True, 'weight': 1000},
        ]

        report = evaluator.summarize_results(results, dataset='local', split='unit')
        variants = {variant['variant_id']: variant for variant in report['variants']}

        self.assertEqual(report['sample_count'], 2)
        self.assertEqual(variants['v1']['passed'], 1)
        self.assertEqual(variants['v1']['failed'], 1)
        self.assertEqual(variants['v1']['total'], 2)
        self.assertEqual(variants['v1']['pass_rate'], 0.5)
        self.assertEqual(variants['v1']['median_latency_ms'], 20)
        self.assertEqual(variants['v1']['total_cost_usd'], 0.3)
        self.assertEqual(report['failures_by_reason'], {evaluator.FAILURE_NO_GOLD_OVERLAP: 1})


class RunnerHarnessCommandTests(unittest.TestCase):
    def test_provider_rate_limited_code_maps_to_provider_rate_limited_failure(self) -> None:
        self.assertEqual(runner._failure_reason_from_harness('provider_rate_limited'), evaluator.FAILURE_PROVIDER_RATE_LIMITED)

    def test_legacy_harness_rate_limited_code_maps_to_provider_rate_limited_failure(self) -> None:
        self.assertEqual(runner._failure_reason_from_harness('harness_rate_limited'), evaluator.FAILURE_PROVIDER_RATE_LIMITED)

    def test_default_pi_command_targets_runtime_issue_runner(self) -> None:
        command = runner._pi_command('kimi-k2.5', 'opencode_zen')

        self.assertEqual(command[:3], [sys.executable, '-m', 'llm.pi_issue_runner'])
        self.assertIn('--provider', command)
        self.assertEqual(command[command.index('--provider') + 1], 'opencode')
        self.assertEqual(command[command.index('--model') + 1], 'kimi-k2.5')

    def test_custom_command_passthrough_remains_available(self) -> None:
        command = runner._pi_command('kimi-k2.5', 'opencode', ['--', 'python', 'custom_runner.py'])

        self.assertEqual(command, ['python', 'custom_runner.py'])


class RunnerCostTests(unittest.TestCase):
    def test_cost_usd_from_opencode_usage(self) -> None:
        usage = {
            'input': 1,
            'output': 2,
            'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0, 'total': 0},
        }

        self.assertEqual(runner._cost_usd_from_usage(usage), 0.0)

    def test_cost_usd_from_flat_usage(self) -> None:
        self.assertEqual(runner._cost_usd_from_usage({'cost_usd': 0.123}), 0.123)
        self.assertIsNone(runner._cost_usd_from_usage({'tokens': 10}))


if __name__ == '__main__':
    unittest.main()
