# WC-1: Project Setup + CI Pipeline

**Branch:** `wc-1-project-setup-ci`
**Dependencies:** None — gates everything else.

## Why This Is First

Every subsequent chapter produces code that must pass lint, typecheck, and tests. If CI isn't in place from the start, quality ratchets are impossible — broken code merges, and fixing it later is 10x harder. This chapter produces zero application logic but is the most important one.

## Files

| File | Action | Notes |
|------|--------|-------|
| `pyproject.toml` | Modify | Add deps, tool configs |
| `.gitignore` | Modify | Build/cache artifacts |
| `src/hypertoric/py.typed` | Create | Empty PEP 561 marker |
| `.pre-commit-config.yaml` | Create | Ruff only (mypy too slow for hooks) |
| `.github/workflows/ci.yml` | Create | 4 jobs: lint, typecheck, test, integration |
| `tests/conftest.py` | Create | Taichi CPU fixture |

## Implementation Details

### pyproject.toml — Dependencies

Current state has only `taichi` and `numpy`. Add:

- **`hydra-core>=1.3`** — all config goes through Hydra structured configs. Pin >=1.3 for ConfigStore API stability.
- **`omegaconf>=2.3`** — Hydra's underlying config library. Explicit dep so mypy can see its types.

Dev dependencies need significant expansion:

- **`pytest-cov>=6.0`** — coverage tracking. `fail_under = 50` initially; raise as chapters land.
- **`pytest-timeout>=2.0`** — Taichi kernels can hang on misconfigured backends. Default timeout prevents CI from running forever.
- **`mypy>=1.14`** — strict mode. Version pin matters because mypy frequently changes strictness behavior.
- **`pre-commit>=4.0`** — hook runner.

### pyproject.toml — Tool Configs

**mypy strict mode** is non-negotiable for this project. The Taichi/Python boundary is where bugs hide — strict typing catches them at the border. Key decisions:

- `python_version = "3.11"` — minimum supported version, ensures we don't use 3.12+ features accidentally.
- `files = ["src/"]` — don't typecheck tests (too many `# type: ignore` needed for test fixtures).
- `ignore_missing_imports = true` for `taichi.*` — Taichi ships no stubs and likely never will. Every `ti.field()`, `ti.kernel`, etc. is `Any` from mypy's perspective. We contain this by typing our own wrapper layer (fields.py, kernel factories).

**ruff lint rules** — the `select` list is intentionally curated:
- `TCH` (type-checking imports) is important because Taichi pollutes the namespace. `TYPE_CHECKING` blocks keep heavy imports out of runtime.
- `B` (bugbear) catches common mistakes like mutable default arguments in dataclasses.
- `SIM` (simplify) prevents over-complicated conditionals that hurt readability in kernel factory code.
- Notably absent: `D` (docstrings) — we don't enforce docstrings. Code should be self-documenting; forced docstrings become stale lies.

**pytest markers** — three markers that control what runs where:
- `gpu` — skipped in CI (no GPU). Developers run these locally.
- `slow` — anything over ~5s. Skipped in the fast test job, included in integration.
- `integration` — end-to-end tests. Run in a separate CI job after unit tests pass.

**coverage** — `fail_under = 50` is deliberately low. Taichi kernel internals are hard to cover (they run on the Taichi VM, not Python). Coverage measures the Python orchestration layer. Will increase as chapters land.

### .gitignore Additions

```
*.egg-info/
dist/
.mypy_cache/
.pytest_cache/
htmlcov/
.coverage
outputs/
```

The `outputs/` entry is critical — Hydra creates `outputs/YYYY-MM-DD/HH-MM-SS/` directories on every run. Without this gitignore entry, `git status` becomes unusable after a single test run.

### tests/conftest.py — Taichi CPU Fixture

Taichi has global state: `ti.init()` sets the backend for the entire process, and fields allocated under one `ti.init()` are invalid after `ti.reset()`. The fixture must:

1. `ti.reset()` — clear any prior state (defensive, in case a test crashed)
2. `ti.init(arch=ti.cpu)` — always CPU in tests for determinism and CI compatibility
3. `yield` — run the test
4. `ti.reset()` — clean up

This is `autouse=True` so every test gets a clean Taichi state without explicitly requesting it. Tests that need custom init (e.g., testing GPU detection) can override by calling `ti.reset()` + `ti.init()` themselves.

**Important edge case:** Taichi's `ti.reset()` is not fully reliable in all versions — some global caches survive. If tests start interfering with each other, the fallback is subprocess isolation via `pytest-forked`. Don't add this preemptively; add it when/if needed.

### CI Workflow Design

Four jobs, not one, for parallelism and clear failure signals:

**lint** (fastest, ~30s): `ruff check` + `ruff format --check`. Fails fast on style issues before burning compute on typecheck/tests.

**typecheck** (~1min): `mypy src/`. Separate from lint because mypy is slow and its failures are qualitatively different (logic errors vs style).

**test** (matrix: 3.11, 3.12, 3.13, ~2min each): `pytest -m "not gpu and not slow" --cov`. Matrix tests catch version-specific issues (Taichi's numpy interop changes between Python versions). The `--cov` flag only runs on the test job.

**integration** (after test, ~2min): `pytest -m integration --timeout=120`. Depends on `test` passing — no point running expensive integration tests if unit tests fail. The 120s timeout catches infinite loops in learning convergence tests.

All jobs use:
- `astral-sh/setup-uv@v4` — installs uv
- `uv sync --frozen --dev` — installs exact lockfile deps (reproducible)
- `TI_ARCH: cpu` env var — forces Taichi to CPU even if a GPU is somehow available

Concurrency group `${{ github.workflow }}-${{ github.ref }}` cancels in-progress runs when a new push arrives on the same branch. Saves CI minutes during rapid iteration.

### .pre-commit-config.yaml

Only ruff, not mypy. Rationale: mypy takes 5-15s even on a small codebase, which makes `git commit` feel sluggish. Ruff takes <100ms. Mypy catches issues in CI where the latency doesn't matter.

Pre-commit-hooks for `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml` — these catch formatting issues that ruff doesn't cover (YAML/TOML syntax errors, missing final newlines).

## Verification

```bash
uv sync                                    # all deps install cleanly
uv run ruff check src/ tests/              # passes on skeleton
uv run ruff format --check src/ tests/     # passes
uv run mypy src/                           # passes (only __init__.py exists)
uv run pytest                              # 0 tests collected, exit 0
```
