from __future__ import annotations

import re

from django.conf import settings


PYTHON_PROFILE_ID = 'python-v1'
MULTI_LANG_JS_TS_PROFILE_ID = 'multi-lang-js-ts-v1'
KNOWN_ANALYSIS_PROFILES = frozenset({
    PYTHON_PROFILE_ID,
    MULTI_LANG_JS_TS_PROFILE_ID,
})

ANALYSIS_PROFILE_PATTERN = re.compile(r'^[A-Za-z0-9_.-]+$')


def is_safe_analysis_profile(value: str) -> bool:
    return 1 <= len(value) <= 128 and ANALYSIS_PROFILE_PATTERN.fullmatch(value) is not None


def is_known_analysis_profile(value: str) -> bool:
    return is_safe_analysis_profile(value) and value in KNOWN_ANALYSIS_PROFILES


def get_active_analysis_profile() -> str:
    configured = str(getattr(settings, 'ANALYZER_PROFILE_ID', PYTHON_PROFILE_ID) or PYTHON_PROFILE_ID)
    return configured if is_known_analysis_profile(configured) else PYTHON_PROFILE_ID


def normalize_analysis_profile(value: str | None = None) -> str:
    profile = str(value or get_active_analysis_profile())
    return profile if is_known_analysis_profile(profile) else ''
