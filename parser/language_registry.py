from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable


SUPPORT_FILE_ONLY = 'file_only'
SUPPORT_TEXT_CONTEXT = 'text_context'
SUPPORT_SYMBOLS = 'symbols'
SUPPORT_RELATIONSHIPS = 'relationships'


@dataclass(frozen=True)
class LanguageSpec:
    id: str
    family: str
    display_name: str
    extensions: tuple[str, ...]
    support_level: str
    parser_name: str | None
    parser_entrypoint: str | None
    resolver_name: str | None
    cache_sensitivity: str
    max_files_setting: str
    max_total_bytes_setting: str
    ignored_path_parts: tuple[str, ...] = ()
    ignored_suffixes: tuple[str, ...] = ()
    builtin_globals: tuple[str, ...] = ()


DEFAULT_IGNORED_PATH_PARTS = frozenset({
    '.git',
    'node_modules',
    'dist',
    'build',
    'coverage',
    'lcov-report',
    'visual-tests',
    '__snapshots__',
    'storybook-static',
    '.next',
    '.nuxt',
    '.turbo',
    '.angular',
    '.parcel-cache',
    '.svelte-kit',
    '.terraform',
    '.serverless',
    '.worktrees',
    '.venv',
    'venv',
    '__pycache__',
    'vendor',
    'target',
    'out',
})

DEFAULT_IGNORED_SUFFIXES = (
    '.min.js',
    '.bundle.js',
    '.map',
    '.d.ts',
)

DEFAULT_IGNORED_FILES = frozenset({
    'package-lock.json',
    'yarn.lock',
    'pnpm-lock.yaml',
    'bun.lockb',
    'npm-shrinkwrap.json',
    'go.sum',
    'go.work.sum',
    'poetry.lock',
    'uv.lock',
    'Pipfile.lock',
    'Gemfile.lock',
    'Cargo.lock',
    'composer.lock',
})

DEFAULT_MANIFEST_FILES = frozenset({
    'package.json',
})

SENSITIVE_FILE_NAMES = frozenset({
    '.env',
    '.env.local',
    '.env.development',
    '.env.production',
    '.npmrc',
    '.pypirc',
    'id_rsa',
    'id_dsa',
    'id_ecdsa',
    'id_ed25519',
})

SENSITIVE_SUFFIXES = (
    '.pem',
    '.key',
    '.crt',
    '.cer',
    '.p12',
    '.pfx',
)

PYTHON_BUILTINS = (
    'abs',
    'dict',
    'len',
    'list',
    'print',
    'set',
    'str',
    'tuple',
)

JAVASCRIPT_BUILTINS = (
    'Array',
    'Boolean',
    'Date',
    'Error',
    'JSON',
    'Map',
    'Math',
    'Number',
    'Object',
    'Promise',
    'Set',
    'String',
    'console',
    'fetch',
    'parseFloat',
    'parseInt',
    'setInterval',
    'setTimeout',
)


PYTHON = LanguageSpec(
    id='python',
    family='python',
    display_name='Python',
    extensions=('.py',),
    support_level=SUPPORT_RELATIONSHIPS,
    parser_name='tree_sitter_python',
    parser_entrypoint='language',
    resolver_name='python',
    cache_sensitivity='file_content',
    max_files_setting='GITHUB_REPO_MAX_PYTHON_FILES',
    max_total_bytes_setting='GITHUB_REPO_MAX_TOTAL_ANALYZED_BYTES',
    ignored_path_parts=(),
    ignored_suffixes=(),
    builtin_globals=PYTHON_BUILTINS,
)

JAVASCRIPT = LanguageSpec(
    id='javascript',
    family='javascript',
    display_name='JavaScript',
    extensions=('.js', '.jsx', '.mjs', '.cjs'),
    support_level=SUPPORT_RELATIONSHIPS,
    parser_name='tree_sitter_javascript',
    parser_entrypoint='language',
    resolver_name='javascript',
    cache_sensitivity='cross_file_resolution',
    max_files_setting='GITHUB_REPO_MAX_JS_TS_FILES',
    max_total_bytes_setting='GITHUB_REPO_MAX_TOTAL_ANALYZED_BYTES',
    ignored_path_parts=tuple(sorted(DEFAULT_IGNORED_PATH_PARTS)),
    ignored_suffixes=DEFAULT_IGNORED_SUFFIXES,
    builtin_globals=JAVASCRIPT_BUILTINS,
)

TYPESCRIPT = LanguageSpec(
    id='typescript',
    family='javascript',
    display_name='TypeScript',
    extensions=('.ts', '.tsx', '.mts', '.cts'),
    support_level=SUPPORT_RELATIONSHIPS,
    parser_name='tree_sitter_typescript',
    parser_entrypoint='language_typescript',
    resolver_name='javascript',
    cache_sensitivity='cross_file_resolution',
    max_files_setting='GITHUB_REPO_MAX_JS_TS_FILES',
    max_total_bytes_setting='GITHUB_REPO_MAX_TOTAL_ANALYZED_BYTES',
    ignored_path_parts=tuple(sorted(DEFAULT_IGNORED_PATH_PARTS)),
    ignored_suffixes=DEFAULT_IGNORED_SUFFIXES,
    builtin_globals=JAVASCRIPT_BUILTINS,
)

LANGUAGE_SPECS = (PYTHON, JAVASCRIPT, TYPESCRIPT)
LANGUAGE_BY_ID = {spec.id: spec for spec in LANGUAGE_SPECS}
_EXTENSION_MAP = {
    extension: spec
    for spec in LANGUAGE_SPECS
    for extension in spec.extensions
}


def normalize_enabled_languages(enabled: Iterable[str] | None) -> tuple[str, ...]:
    if enabled is None:
        return ('python',)
    normalized: list[str] = []
    for item in enabled:
        language_id = str(item).strip().lower()
        if not language_id or language_id not in LANGUAGE_BY_ID or language_id in normalized:
            continue
        normalized.append(language_id)
    return tuple(normalized or ('python',))


def language_for_path(path: str, *, enabled_languages: Iterable[str] | None = None) -> LanguageSpec | None:
    normalized_path = PurePosixPath(path)
    suffix = normalized_path.suffix.lower()
    spec = _EXTENSION_MAP.get(suffix)
    if spec is None:
        return None
    enabled = set(normalize_enabled_languages(enabled_languages))
    if spec.id not in enabled:
        return None
    return spec


def language_family_for_path(path: str, *, enabled_languages: Iterable[str] | None = None) -> str | None:
    spec = language_for_path(path, enabled_languages=enabled_languages)
    return spec.family if spec is not None else None


def analyzable_extensions(*, enabled_languages: Iterable[str] | None = None) -> set[str]:
    enabled = set(normalize_enabled_languages(enabled_languages))
    return {
        extension
        for spec in LANGUAGE_SPECS
        if spec.id in enabled
        for extension in spec.extensions
    }


def ignored_reason(path: str, *, language_spec: LanguageSpec | None = None) -> str | None:
    normalized = PurePosixPath(path)
    parts = set(normalized.parts)
    name = normalized.name
    lower_name = name.lower()
    lower_path = normalized.as_posix().lower()
    ignored_path_parts = DEFAULT_IGNORED_PATH_PARTS if language_spec is None else set(language_spec.ignored_path_parts)
    ignored_suffixes = DEFAULT_IGNORED_SUFFIXES if language_spec is None else language_spec.ignored_suffixes

    if parts & ignored_path_parts:
        return 'generated_or_vendor'
    if name in DEFAULT_IGNORED_FILES:
        return 'lockfile'
    if name in DEFAULT_MANIFEST_FILES:
        return 'manifest_deferred'
    if name in SENSITIVE_FILE_NAMES or lower_name.startswith('.env.'):
        return 'sensitive'
    if lower_name.endswith(SENSITIVE_SUFFIXES):
        return 'sensitive'
    if any(lower_path.endswith(suffix) for suffix in ignored_suffixes):
        return 'generated'
    return None


def is_supported_source_path(path: str, *, enabled_languages: Iterable[str] | None = None) -> bool:
    spec = language_for_path(path, enabled_languages=enabled_languages)
    return spec is not None and ignored_reason(path, language_spec=spec) is None


def is_text_context_path(path: str, *, enabled_languages: Iterable[str] | None = None) -> bool:
    spec = language_for_path(path, enabled_languages=enabled_languages)
    return spec is not None and spec.support_level in {SUPPORT_TEXT_CONTEXT, SUPPORT_SYMBOLS, SUPPORT_RELATIONSHIPS} and ignored_reason(path, language_spec=spec) is None


def language_inventory(files: Iterable[str], *, enabled_languages: Iterable[str] | None = None) -> dict[str, dict[str, int | str]]:
    enabled = set(normalize_enabled_languages(enabled_languages))
    inventory: dict[str, dict[str, int | str]] = {
        spec.id: {
            'id': spec.id,
            'display_name': spec.display_name,
            'family': spec.family,
            'support_level': spec.support_level,
            'file_count': 0,
            'stored_file_count': 0,
            'skipped_file_count': 0,
        }
        for spec in LANGUAGE_SPECS
        if spec.id in enabled
    }
    for file_path in files:
        spec = language_for_path(file_path, enabled_languages=enabled)
        if spec is None:
            continue
        inventory[spec.id]['file_count'] = int(inventory[spec.id]['file_count']) + 1
        if ignored_reason(file_path, language_spec=spec) is None:
            inventory[spec.id]['stored_file_count'] = int(inventory[spec.id]['stored_file_count']) + 1
        else:
            inventory[spec.id]['skipped_file_count'] = int(inventory[spec.id]['skipped_file_count']) + 1
    return inventory
