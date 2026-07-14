# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multi-user AI agent that runs end-to-end data science workflows (EDA, feature engineering, AutoML, evaluation, reporting) on user datasets. The core is a **LangGraph pipeline** orchestrated by a node/service split, fed work through a database-backed job queue.

## The three processes

The system is **three independent processes** that must run separately (each in its own terminal). They communicate only through the shared application database — there is no in-process coupling.

```bash
uv run python src/mcp_server.py   # MCP tool server — SQL + schema tools for the agent (port 9001, /mcp)
uv run python main.py             # FastAPI API server — serves UI, auth, projects, uploads (port 8000, /docs)
uv run python worker_main.py      # Background worker — polls job queue, runs the LangGraph pipeline
```

The API **only enqueues jobs** (writes `BackgroundJobModel` rows); it never runs the graph. `main.py`'s lifespan handler is intentionally empty — the worker is fully decoupled in `worker_main.py`. Heavy ML therefore never blocks web traffic. You can run multiple workers in parallel safely (Postgres `SELECT ... FOR UPDATE SKIP LOCKED`). On startup the worker re-queues jobs left `running`/`pending` from a previous crash.

Production runs the API via the root `Dockerfile` and the worker via `deploy/Dockerfile.worker` (see `deploy/README.md`, Turkish).

## Running tests

No pytest config exists; tests run directly. Most files in `tests/` and the root `test_*.py` are standalone scripts (often `__main__`-style mock harnesses), not a pytest suite.

```bash
uv run pytest tests/test_intent_discovery_agent.py   # the one true pytest-style file
uv run python test_phase1.py                          # root test_*.py are standalone scripts
```

## Architecture

### Two separate LangGraphs
- **`src/graph.py`** — the main DS pipeline (`build_graph()`). This is what the worker runs.
- **`src/graphs/talktodata/graph.py`** — a separate conversational "talk to data" graph (intent → SQL → validate → chart → interpret) with its own `TTDState` and `ttd_*` nodes. Don't confuse the two.

### Pipeline routing (src/graph.py)
`intent_node` classifies the query, then `db_context_node` routes by `intent`:
- `sql_query` → `sql_node` → END
- everything else → `target_discovery_node` → `planner_node` → `execution_node`, which then forks:
  - `classification`/`regression`/`clustering` → `entity_resolution_node` → `leakage_detector_node` → `feature_plan_node` → `feature_engineering_node` → `automl_node` → `evaluation_node` → `critic_node`
  - `deep_analysis` → `analysis_node` → END (**no model training, no feature/leakage path — fully separate**)
  - `anomaly_detection` → `automl_node`; `eda`/`report` → `report_node`
- `critic_node` does self-correction: it sets `next_retry_node` to loop back to `feature_plan_node` or `automl_node`, or falls through to `report_node`.

When changing routing, edit both the routing functions (`route_after_*`) **and** the `add_conditional_edges` maps — they must stay in sync.

### Node / service split
Nodes in `src/nodes/` are thin LangGraph adapters: read from `AgentState`, call a service, write results back into the state dict. The real logic lives in `src/services/` (`ml_trainer`, `feature_builder`, `leakage_detector`, `analysis_engine`, `evaluator`, `critic_advisor`, `report_generator`, etc.). Put logic in the service; keep the node as glue. `AgentState` (`src/state.py`) is a single `total=False` TypedDict — add new fields there.

**Nodes return only the keys they change (partial returns), not `{**state, ...}`.** LangGraph keeps unlisted channels at their existing value, so spreading the whole state is redundant and breaks accumulating fields. The accumulating fields `critic_feedback` and `retry_history` use `Annotated[list, operator.add]` reducers — nodes emit only the **new delta** (e.g. `"critic_feedback": decision["feedback"]`), never the full concatenated list, or the reducer will double it.

**Every node is wrapped with `@node("name")` from `src/core/nodes.py`.** The decorator provides three cross-cutting behaviors so node bodies stay focused: (1) upstream-error short-circuit — if `state["error"]` is set the body is skipped and only `{"current_step": name}` is returned, so the error rides through to `report_node` (the one node decorated `@node(..., runs_on_error=True)`); (2) an exception safety net that converts any uncaught exception into a structured `error` + `error_category` (via `classify_error`) instead of crashing the whole graph; (3) structured start/done/error timing logs. Don't re-add manual `if state.get("error"): return ...` guards or `[NODE START]` banner prints — the decorator handles both. Conditional-edge maps in `graph.py` use the **list form** (`["a","b"]`) since the `route_*` functions already return the destination node name; this removes routing/map drift.

### Analysis registry pattern
`src/services/analysis_engine.py` runs deep EDA via a registry: `ANALYSIS_MODULES` maps an `analysis_type` ("general", "fintech") to a list of `compute_*` functions. **Every `compute_*` shares the signature `compute_x(df, profile, target_col) -> dict | None`** and returns a dict keyed by its own output (e.g. `{"correlation_matrix": {...}}`); `None` skips it. To add an analysis, write a `compute_*` with that signature and register it in `ANALYSIS_MODULES` — `run_deep_analysis` iterates the list and merges results. Analyses must stay dataset-agnostic (no hardcoded column names).

### Databases (two distinct layers)
1. **Application DB** (`API_DATABASE_URL`, `api/core/database.py`) — users, projects, data groups, CSVs, the job queue. Shared by API + worker. SQLite fallback if unset; Postgres recommended for multi-user. Models in `api/models/`, declared on the `Base` there; **import every model in `main.py`/`worker_main.py`** so `create_all` and SQLAlchemy mappers see them.
2. **Dataset DBs** (`src/core/db_provider.py`, reached via the MCP server) — the user's actual data the agent queries. Configured via `DB_PROVIDER`/`DB_CONNECTION_STRING` or the multi-DB `DATABASES` JSON env var. Providers: `sqlite`, `postgresql`, `mysql`, `mssql`.

LangGraph checkpoints are stored separately again: Postgres (`AsyncPostgresSaver`) when `API_DATABASE_URL` is Postgres, else `data/checkpoints.sqlite`. The graph's `thread_id` is the `project_id`; `force_restart` deletes checkpoint rows and clears `ProjectStepModel`.

**Tests/migrations must NEVER hit the remote app DB.** `.env`'s `API_DATABASE_URL` usually points at production Postgres. `api/core/database.py` uses `load_dotenv(override=False)`, so a **real env var wins over `.env`** — to run any test/migration against a sandbox, set `API_DATABASE_URL=sqlite:///<abs/path>` before invoking (no wrapper needed). As a hard backstop, destructive/DDL scripts (`scripts/migrate_db.py`, `scripts/reset_app_db.py`) call `assert_local_db()` from `api/core/db_safety.py`, which **aborts before connecting** if the target is remote (non-sqlite, non-localhost) unless you opt in with `ALLOW_REMOTE_DB_OPS=1`. Production migrations are run by hand with that flag set. Never flip `database.py` back to `override=True`, and gate any new schema/data-mutating script with `assert_local_db()`.

### LLM access
Always get models through `get_llm(model_name, temperature, provider)` in `src/core/llm.py` — never instantiate `ChatOpenAI` etc. directly. Provider/model defaults come from `src/core/config.py` `settings` (env-driven: `LLM_PROVIDER`, `LLM_MODEL_NAME`, plus a separate `INTELLIGENCE_ENGINE_*` pair for the heavier intelligence engine). Determinism is enabled by default (`RANDOM_SEED=42`, `ENABLE_DETERMINISM`).

## Conventions & gotchas
- **Config via `from src.core.config import settings`** — don't read `os.getenv` ad hoc for things already on `settings`.
- **Structured logging via `src/core/logging_config.py`.** `setup_logging(process=..., db_logging=...)` is the single control point; each process calls it once at startup (worker `process="worker", db_logging=True`; API `process="api"`; MCP `process="mcp"`). The worker wraps each job with `set_project(project_id)` so every log line (including the `@node` decorator's logs) is tagged `[project=<id>]`. Use `get_logger(__name__)` — **don't use `print`** (almost all `src/`+`api/` prints were migrated to the logger; the only deliberate exception is the stderr fallback in `src/core/log_handlers.py`, which must stay `print` to avoid a logging feedback loop). Handlers: console (always) + optional DB (`DBLogQueueHandler` → async/batched into `app_logs`, on for the worker). New handlers (file, remote) go in `_build_handlers()`. Retention via `log_handlers.prune_logs()`. Env: `LOG_LEVEL`, `LOG_DB_ENABLED`, `LOG_DB_LEVEL`, `LOG_DB_BATCH_SIZE`, `LOG_DB_FLUSH_INTERVAL`, `LOG_DB_QUEUE_MAXSIZE`.
- **Worker state serialization:** anything written to `ProjectStepModel.step_data` is sanitized in `api/core/worker.py` — DataFrames/model artifacts are stripped, large lists truncated to 5 rows, and NaN/Inf → `None` (psycopg2's JSON rejects NaN). If you add big or non-JSON state fields, account for this filter.
- **Measuring anything downstream of the LLM feature plan: one run proves nothing.** The plan comes from an LLM, and gpt-4o is not bit-deterministic even at `temperature=0`. Measured on the e-commerce reference set: **the same arm, same input, same seed swings ±0.004 ROC between runs.** So an A/B whose effect is smaller than that (most of them) cannot be read from a single pair of runs — you will "measure" noise and believe it. Either **pin the plan** (generate `feature_plan` once, reuse the identical dict in both arms, so the only difference is the thing under test) or **average 3-5 runs per arm**. Differences below 0.004 are noise and must not be interpreted. This matters most for engine-on/off baseline comparisons. Where a leakage/contamination question can be settled *deterministically* (e.g. `tests/test_analysis_hints_leakage.py` compares the hint payload built from all rows vs train-only rows), prefer that over a ROC A/B — ROC's resolution is too coarse to answer it.
- Comments and log strings are frequently **Turkish** (`docs/`, `deploy/README.md`, inline) — match the surrounding language when editing a file.
- `app.py` (root, ~80KB) is a legacy Streamlit dashboard; `legacy/` holds older entrypoints. The current product is the FastAPI + worker + MCP trio, not these.
- `data/` is gitignored (datasets, models, checkpoints, SQLite DBs). Don't commit anything under it.
- Package manager is **uv** (`uv.lock`); Python **3.12+**. `requirements.txt` exists for the Docker build.
