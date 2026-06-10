from __future__ import annotations

from pathlib import PurePosixPath
import json
import posixpath
import re
from typing import Any, Mapping

import yaml


SOURCE_EXTENSIONS = ('.ts', '.tsx', '.js', '.jsx', '.mts', '.cts', '.mjs', '.cjs')
INDEX_FILENAMES = tuple(f'index{extension}' for extension in SOURCE_EXTENSIONS)
SIDE_CAR_EXTENSIONS = ('.svelte.ts', '.svelte.js')


def _strip_json_comments(text: str) -> str:
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    return re.sub(r'(^|[^:])//.*$', r'\1', text, flags=re.M)


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(_strip_json_comments(text))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _safe_join(base_dir: str, target: str) -> str | None:
    base = PurePosixPath(base_dir)
    target_path = PurePosixPath(target)
    if base.is_absolute() or target_path.is_absolute():
        return None
    joined = posixpath.normpath(posixpath.join(base.as_posix(), target_path.as_posix()))
    if joined == '.':
        return ''
    parts = PurePosixPath(joined).parts
    if any(part == '..' for part in parts):
        return None
    return joined


def _dirname(path: str) -> str:
    parent = PurePosixPath(path).parent.as_posix()
    return '' if parent == '.' else parent


def _candidate_files(base_path: str) -> list[str]:
    path = PurePosixPath(base_path)
    suffix = path.suffix
    candidates = [path.as_posix()]
    if suffix == '.js':
        candidates.insert(0, path.with_suffix('.ts').as_posix())
    elif suffix == '.jsx':
        candidates.insert(0, path.with_suffix('.tsx').as_posix())
    elif suffix == '.svelte':
        candidates.extend(f'{path.as_posix()}{extension.removeprefix(".svelte")}' for extension in SIDE_CAR_EXTENSIONS)
    elif suffix == '':
        candidates.extend(f'{path.as_posix()}{extension}' for extension in SOURCE_EXTENSIONS)
    return list(dict.fromkeys(candidates))


def _candidate_indexes(base_path: str) -> list[str]:
    return [PurePosixPath(base_path).joinpath(filename).as_posix() for filename in INDEX_FILENAMES]


def _resolve_existing(base_path: str, all_paths: set[str]) -> str | None:
    for candidate in _candidate_files(base_path):
        if candidate in all_paths:
            return candidate
    for candidate in _candidate_indexes(base_path):
        if candidate in all_paths:
            return candidate
    return None


def _merge_compiler_options(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == 'paths' and isinstance(value, dict):
            merged_paths = dict(merged.get('paths') or {})
            merged_paths.update(value)
            merged[key] = merged_paths
        else:
            merged[key] = value
    return merged


def _read_tsconfig(path: str, manifest_contents: Mapping[str, str], seen: set[str] | None = None) -> dict[str, Any]:
    seen = seen or set()
    if path in seen:
        return {}
    seen.add(path)
    data = _json_object(manifest_contents.get(path, ''))
    if not data:
        return {}

    compiler_options: dict[str, Any] = {}
    extends_value = data.get('extends')
    extends_items = extends_value if isinstance(extends_value, list) else [extends_value]
    for item in extends_items:
        if not isinstance(item, str) or not item.startswith('.'):
            continue
        extended_path = _safe_join(_dirname(path), item)
        if extended_path is None:
            continue
        if not extended_path.endswith('.json'):
            extended_path = f'{extended_path}.json'
        extended = _read_tsconfig(extended_path, manifest_contents, seen)
        compiler_options = _merge_compiler_options(compiler_options, extended)

    own_options = data.get('compilerOptions')
    if isinstance(own_options, dict):
        compiler_options = _merge_compiler_options(compiler_options, own_options)
    return compiler_options


def _nearest_config_options(from_path: str, manifest_contents: Mapping[str, str]) -> tuple[str, dict[str, Any]]:
    current = PurePosixPath(_dirname(from_path))
    candidates: list[str] = []
    while True:
        prefix = '' if current.as_posix() == '.' else current.as_posix()
        for name in ('tsconfig.json', 'jsconfig.json'):
            candidates.append(PurePosixPath(prefix).joinpath(name).as_posix() if prefix else name)
        if current.as_posix() in {'', '.'}:
            break
        current = current.parent
    for candidate in candidates:
        if candidate in manifest_contents:
            return _dirname(candidate), _read_tsconfig(candidate, manifest_contents)
    return '', {}


def _resolve_tsconfig_alias(from_path: str, specifier: str, all_paths: set[str], manifest_contents: Mapping[str, str]) -> str | None:
    config_dir, options = _nearest_config_options(from_path, manifest_contents)
    paths = options.get('paths')
    if not isinstance(paths, dict):
        return None
    base_url = options.get('baseUrl')
    raw_base = str(base_url) if isinstance(base_url, str) and base_url else '.'
    base = _safe_join(config_dir, raw_base)
    if base is None:
        return None

    for pattern, raw_targets in paths.items():
        if not isinstance(pattern, str):
            continue
        targets = raw_targets if isinstance(raw_targets, list) else [raw_targets]
        if '*' in pattern:
            prefix, suffix = pattern.split('*', 1)
            if not specifier.startswith(prefix) or (suffix and not specifier.endswith(suffix)):
                continue
            wildcard = specifier[len(prefix): len(specifier) - len(suffix) if suffix else len(specifier)]
        elif specifier != pattern:
            continue
        else:
            wildcard = ''
        for target in targets:
            if not isinstance(target, str):
                continue
            target_value = target.replace('*', wildcard)
            joined = _safe_join(base, target_value)
            if joined is None:
                continue
            resolved = _resolve_existing(joined, all_paths)
            if resolved is not None:
                return resolved
    return None


def _workspace_package_dirs(manifest_contents: Mapping[str, str]) -> list[str]:
    workspace_text = manifest_contents.get('pnpm-workspace.yaml')
    if not workspace_text:
        return []
    try:
        workspace = yaml.safe_load(workspace_text) or {}
    except yaml.YAMLError:
        return []
    packages = workspace.get('packages') if isinstance(workspace, dict) else None
    if not isinstance(packages, list):
        return []
    dirs: list[str] = []
    for item in packages:
        if not isinstance(item, str) or item.startswith('!'):
            continue
        normalized = item.rstrip('/')
        if normalized in {'.', ''}:
            dirs.append('')
        elif normalized.endswith('/*'):
            prefix = normalized[:-2].rstrip('/')
            for path in manifest_contents:
                if path.startswith(f'{prefix}/') and path.endswith('/package.json'):
                    dirs.append(path.removesuffix('/package.json'))
        else:
            dirs.append(normalized)
    return list(dict.fromkeys(dirs))


def _package_entry_candidates(package_dir: str, package_json: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    exports = package_json.get('exports')
    if isinstance(exports, str):
        candidates.append(exports)
    elif isinstance(exports, dict):
        root_export = exports.get('.')
        if isinstance(root_export, str):
            candidates.append(root_export)
        elif isinstance(root_export, dict):
            for key in ('types', 'import', 'module', 'default', 'require'):
                value = root_export.get(key)
                if isinstance(value, str):
                    candidates.append(value)
    for key in ('types', 'module', 'main', 'svelte'):
        value = package_json.get(key)
        if isinstance(value, str):
            candidates.append(value)
    candidates.extend(('src/index', 'index'))
    resolved: list[str] = []
    for candidate in candidates:
        joined = _safe_join(package_dir, candidate)
        if joined is not None:
            resolved.append(joined)
    return list(dict.fromkeys(resolved))


def _resolve_workspace_package(specifier: str, all_paths: set[str], manifest_contents: Mapping[str, str]) -> str | None:
    for package_dir in _workspace_package_dirs(manifest_contents):
        package_path = f'{package_dir}/package.json' if package_dir else 'package.json'
        package_json = _json_object(manifest_contents.get(package_path, ''))
        if package_json.get('name') != specifier:
            continue
        for candidate in _package_entry_candidates(package_dir, package_json):
            resolved = _resolve_existing(candidate, all_paths)
            if resolved is not None:
                return resolved
    return None


def resolve_js_module_path(
    from_path: str,
    specifier: str,
    all_paths: set[str],
    manifest_contents: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None]:
    manifest_contents = manifest_contents or {}
    if not specifier:
        return None, None

    if specifier.startswith(('./', '../')):
        joined = _safe_join(_dirname(from_path), specifier)
        if joined is None:
            return None, None
        resolved = _resolve_existing(joined, all_paths)
        return (resolved, 'relative') if resolved is not None else (None, None)

    alias_resolved = _resolve_tsconfig_alias(from_path, specifier, all_paths, manifest_contents)
    if alias_resolved is not None:
        return alias_resolved, 'tsconfig_paths'

    workspace_resolved = _resolve_workspace_package(specifier, all_paths, manifest_contents)
    if workspace_resolved is not None:
        return workspace_resolved, 'pnpm_workspace'

    return None, None
