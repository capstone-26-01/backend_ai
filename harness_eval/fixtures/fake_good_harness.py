#!/usr/bin/env python3
from __future__ import annotations

import json
import sys


def main() -> int:
    job = json.loads(sys.stdin.read())
    sample_id = job['job_id']
    if sample_id == 'origin_trace':
        nodes = ['api/services.py::_build_and_store_analysis', 'parser/services.py::parse_repo']
        paths = ['api/services.py', 'parser/services.py']
    elif sample_id == 'prompt_injection_guard':
        nodes = ['api/views.py::analysis', 'api/services.py::get_repo_analysis']
        paths = ['api/views.py', 'api/services.py']
    else:
        nodes = ['api/services.py::get_repo_analysis', 'api/services.py::_build_and_store_analysis']
        paths = ['api/services.py', 'api/services.py']

    transcript = {
        'sample_id': sample_id,
        'variant_id': 'fake-good',
        'tool_calls': [
            {'name': 'rank_issue_candidates', 'arguments': {'job_id': sample_id}},
            {'name': 'load_focus_graph', 'arguments': {'node_ids': nodes}},
            {'name': 'load_code_context', 'arguments': {'paths': sorted(set(paths))}},
        ],
        'final': {
            'hypotheses': [
                {
                    'kind': 'likely_origin' if index == 0 else 'related_area',
                    'node_id': node_id,
                    'confidence': 0.82 - (index * 0.08),
                    'rationale': 'Synthetic harness fixture result.',
                }
                for index, node_id in enumerate(nodes)
            ],
            'investigation_path': [
                {
                    'step': index,
                    'node_id': node_id,
                    'path': paths[index - 1],
                    'action': 'inspect',
                    'why': 'Synthetic harness fixture path.',
                }
                for index, node_id in enumerate(nodes, start=1)
            ],
            'confidence': {'level': 'high', 'score': 0.82, 'reasons': ['synthetic fixture']},
        },
    }
    print(json.dumps(transcript))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
