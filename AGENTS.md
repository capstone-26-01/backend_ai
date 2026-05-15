# Repository Knowledge Base: capstone-26-01 Backend

Generated: Tue May 05 2026  
Branch: dev  
Commit: 2e37acc

## OVERVIEW
Backend for GitHub Python-repo analysis. Main flow: `api/` receives request -> `github_repo/` fetches files -> `parser/` builds tree/graph -> `llm/` answers QA.

Stack: Django, DRF, drf-spectacular, tree-sitter-python, OpenAI API.

## STRUCTURE
```text
backend/
├── api/          # DRF app: endpoints, serializers, exception handling
├── config/       # Django project settings + URL routing + WSGI/ASGI
├── github_repo/  # GitHub REST fetch helpers
├── llm/          # OpenAI prompt/response wrapper
├── parser/       # Python symbol/tree/edge extraction
├── private_docs/ # Confidential mentoring/internal notes; non-default
├── manage.py     # Django dev entry point
├── Procfile      # Gunicorn runtime entry
└── requirements.txt
```

## WHERE TO LOOK
| Task | Location | Notes |
|---|---|---|
| Add or change API endpoint | `api/views.py`, `api/urls.py` | Keep view thin; reuse services |
| Validate input | `api/serializers.py` | `repo_url` becomes `repo_path` |
| Adjust API error shape | `api/exceptions.py` | DRF custom exception handler |
| Change repo fetch behavior | `github_repo/services.py` | Only place for GitHub API calls |
| Change tree/graph output | `parser/services.py` | Core parse logic |
| Change QA prompt/model flow | `llm/services.py` | OpenAI wrapper |
| Change routing/docs | `config/urls.py` | Swagger at `/api/docs/` |
| Change settings/env behavior | `config/settings.py` | `.env` required |
| Check current tests | `api/tests.py` | Minimal coverage |

## CONVENTIONS
- Views use `@api_view` and `extend_schema`.
- Business logic stays outside `api/views.py`; prefer service helpers.
- `api/serializers.py` normalizes GitHub URLs into `repo_path` strings.
- Parser scope is Python only. Do not expand to other file types casually.
- Use Korean-facing validation/error text where existing files already do.
- When a user assigns a task, agents must create a dedicated working branch before making changes.
- Branch names must use conventional task prefixes plus a short kebab-case slug, e.g. `feat/repo-graph-endpoint`, `fix/parser-edge-resolution`, `chore/update-agents-guide`, `docs/api-usage-notes`, `refactor/view-service-split`, `test/api-qa-endpoint`.
- This repo currently uses a `main` / `dev` / task-branch flow. Complete task branches must merge into `dev`. Never merge user-task work directly into `main`.
- If a commit is explicitly requested, agents must use conventional commit naming: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`. Optional scope is allowed, e.g. `fix(api): normalize repo path`. Keep subject short and purpose-first.
- Keep commit history tidy. Prefer a small number of meaningful commits over noisy micro-commits; if the history gets messy during execution, clean it up before landing to `dev`.
- Runtime scratch files must stay under `./temp` only. Do not scatter temporary outputs elsewhere in the repo.
- Downloaded or cloned external repositories must live under `./playground` only.
- When a user asks for work to be landed, push the finished result to `origin/dev` after merging the task branch into local `dev`.
- Do not inject agent-specific identity into git metadata or docs. Never plug the agent's own email address or contact info into repository history/config unless the user explicitly asks for it.

## ANTI-PATTERNS
- Do not place domain logic in `api/models.py`; it is effectively unused here.
- Do not call GitHub APIs outside `github_repo/services.py`.
- Do not bypass serializers for repo URL validation.
- Do not change parser output keys (`tree`, `nodes`, `edges`) without tracing API impact.
- Do not treat `private_docs/` as normal source input.
- Do not assume CI, lint, typecheck, Docker, or Make targets exist.
- Do not work directly on `main` for user tasks.
- Do not merge completed task branches into `main`; merge to `dev`.

## UNIQUE STYLES
- Telegraphic comments, often Korean, are normal in routing and error code.
- Symbol IDs use `file_path::symbol_name` and nested method IDs append another `::name`.
- `db.sqlite3` and `venv/` exist in the repo even though `.gitignore` excludes them. Treat both as present-but-non-authoritative.
- All `AGENTS.md` files are local working guidance and are gitignored.

## COMMANDS
```bash
python manage.py runserver
python manage.py test api
python manage.py makemigrations
python manage.py migrate
pip install -r requirements.txt
```

## NOTES
- Entry points: `manage.py`, `config/wsgi.py`, `config/asgi.py`, `Procfile`.
- API docs: `/api/schema/` and `/api/docs/`.
- Required env keys: `SECRET_KEY`, `OPENAI_API_KEY`.
- No existing AGENTS hierarchy before this initialization.
- Child guides live only in `api/` and `parser/`.
