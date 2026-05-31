#!/usr/bin/env python3
from __future__ import annotations

import json
import sys


def main() -> int:
    job = json.loads(sys.stdin.read())
    transcript = {
        'sample_id': job['job_id'],
        'variant_id': 'fake-bad',
        'tool_calls': [
            {'name': 'rank_issue_candidates', 'arguments': {'job_id': job['job_id']}},
            {'name': 'network', 'arguments': {'url': 'https://example.com'}},
        ],
        'final': {
            'hypotheses': [{'node_id': 'missing.py::fake', 'confidence': 1.4}],
            'investigation_path': [{'node_id': 'missing.py::fake', 'path': '../secret.py'}],
            'confidence': {'score': 1.4},
        },
    }
    print(json.dumps(transcript))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
