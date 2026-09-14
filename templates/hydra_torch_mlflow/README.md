# Hydra (Python Configs) + Torch + MLflow

A generic skeleton for new ML codebases. Key traits:

- **Python configs**: experiment configurations are dataclasses that are auto-registered
  into Hydra's `ConfigStore` from the root-level `configs/` package (no YAML files).
- **Pure PyTorch**: no Lightning Fabric; device handling and seeding are plain torch.
- **MLflow logging**: a generic `MLFlowLogger` writes the resolved `config.yaml` and
  per-sample metrics, images, tensors and metadata into the run directory, and
  additionally mirrors everything to an MLflow tracking server when one is configured.

## Installation

1. Install [direnv](https://direnv.net/docs/installation.html) and its shell hook.
2. Fill in the values in `.envrc` (outputs dir, MLflow server credentials).
   For local machines without a GPU, use `cp misc/local.envrc .envrc` instead.
3. `direnv allow` — env variables are loaded whenever you enter the project dir.
4. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:
   ```bash
   uv sync --all-groups
   uv run pre-commit install
   ```

## Running

Always run from the project root with `uv run`:

```bash
HYDRA_FULL_ERROR=1 uv run main.py --config-name example
```

`DATA`/`MODEL` environment variables parameterize the selected dataset/model variants
(defaulting to `dummy` when unset); other components are selected with Hydra overrides:

```bash
HYDRA_FULL_ERROR=1 DATA=dummy MODEL=dummy uv run main.py --config-name example
uv run main.py --config-name example logger=dummy
```

Each run lands in `$OUTPUTS_DIR/<date>/<time>/<uuid>/` (see `log_dir` in
`configs/example.py`) containing:

```
.hydra/          # hydra's config snapshot
config.yaml      # fully resolved config, written by the logger
logs/<id>/       # per-sample artifacts
  images/*.png
  data/*.pt
  metadata/*.json
logs/metrics/    # metric series: one CSV per key (step,value), appended per step
```

## MLflow

The logger reads its settings from environment variables (see `.envrc`):

| Variable | Effect |
|---|---|
| `TRACKING_URI` (or `MLFLOW_TRACKING_URI`) | tracking server; when empty/unset, remote tracking is skipped and the run directory is self-contained |
| `EXPERIMENT_NAME` | MLflow experiment name, created if it does not exist (server mode only) |
| `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` | server credentials |

The run directory is the source of truth and has **identical content in both
modes**: `config.yaml`, per-sample artifacts under `logs/<id>/` (images, data,
metadata — one file per entry per step), and metric series under
`logs/metrics/` (one CSV per key, appended step by step).

When a tracking server is configured, the same run is additionally mirrored to
the server: the **full resolved config is always uploaded** as the `config.yaml`
artifact (byte-identical to the local file, so no config content is ever lost
online), each metric key as a single metric series (one point per `log()`
call, no per-sample series), and images and metadata as artifacts. `data/`
tensors are intentionally kept local only. The params table is a *searchable
mirror only*: values longer than 500 characters are truncated there, while the
authoritative config remains the `config.yaml` artifact. No local MLflow file
store (`mlruns/` + `mlflow.db`) is ever created in either mode.

## Config system

- `configs/registry.py` walks the `configs/` package and registers every dataclass into
  Hydra's `ConfigStore`. `configs/<name>.py` becomes `--config-name <name>`;
  `configs/<group>/<name>.py` becomes group `<group>`, config `<name>` (nesting is
  supported: `configs/<group>/<sub>/<name>.py` -> group `<group>/<sub>`).
- A module may export only part of its dataclasses via `__all__` (useful for nested
  sub-configs shared between several configs).
- Component configs carry an absolute `_target_` path and are instantiated with
  `hydra.utils.instantiate` inside the experiment.
- The root config carries `defaults`, an `exp` section (`seed`, `log_dir`, `run_func`)
  and a `hydra` section pointing `run.dir` at `exp.log_dir`, so hydra's own artifacts
  and the logger's artifacts (`run_path = ${hydra:runtime.output_dir}`) share one dir.
  These two settings are coupled — change both together.
- The `short_uuid` resolver (registered in `configs/registry.py`) suffixes `log_dir`
  with a short uuid so every run gets a unique directory.

### Adding a component

1. Implement the class under `src/<package>/...`, subclassing the matching base class.
2. Add a config dataclass in `configs/<group>/<name>.py` with `_target_` set to the
   absolute import path of the class.
3. Reference it in the root config's `defaults` (hardcoded, or via
   `${oc.env:VAR,default}` for env-parameterized selection).

## Logging conventions

`logger.log(log_dict, id=sample_id)` dispatches on key prefixes; subpaths in the
key map to subdirectories (e.g. `images/generated/foo` lands in
`logs/<id>/images/generated/`):

| Prefix | Handling |
|---|---|
| `images/...` | saved as PNG under `logs/<id>/images/` and uploaded via `mlflow.log_image` when tracking is enabled |
| `data/...` | saved as `.pt` tensors under `logs/<id>/data/` |
| `metrics/...` | saved as JSON under `logs/<id>/metrics/` (one file per metric per step) and logged via `mlflow.log_metrics` as `<id>/<key>` (step = per-id counter) when tracking is enabled |
| `metadata/...` | saved as JSON under `logs/<id>/metadata/` and uploaded via `mlflow.log_dict` when tracking is enabled |

Use the `exclude_*` lists in `configs/logger/mlflow.py` to drop individual keys.
Use `logger=dummy` to smoke-test runs without MLflow.

## Tests

```bash
uv run pytest -m unit
```

## Renaming for your project

Replace the placeholder name `hydra_torch_mlflow` with your project name:

1. `pyproject.toml` -> `name`
2. `src/hydra_torch_mlflow/` -> `src/<your_project>/` (and every
   `from hydra_torch_mlflow...` import inside it)
3. every `_target_` string in `configs/` and the `run_func` in `configs/example.py`
4. `.envrc` -> `PROJECT`
