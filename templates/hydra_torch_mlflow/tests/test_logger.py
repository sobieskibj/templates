import json
from pathlib import Path

import mlflow
import pytest
import torch
from omegaconf import OmegaConf

from hydra_torch_mlflow.logger.mlflow import MLFlowLogger


@pytest.mark.unit
def test_mlflow_logger_writes_locally_without_tracking(tmp_path, monkeypatch):
    """Without a tracking URI everything must land in the run dir, and no MLflow
    file store (mlruns/, mlflow.db) may be created anywhere."""
    monkeypatch.delenv("TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    run_path = tmp_path / "run"
    logger = MLFlowLogger(run_path=run_path, root_config={"exp": {"seed": 42}})

    logger.log(
        {
            "images/generated/frame": torch.zeros(3, 8, 8),
            "images/real/frame": torch.zeros(1, 8, 8),
            "data/outputs": torch.zeros(1, 2),
            "metadata/batch": {"index": 0, "shape": [1, 2]},
            "metrics/val/acc": 0.75,
        },
        id="0",
    )
    logger.log({"metrics/dummy_mean": -0.25}, id="summary")
    logger.close()

    # config persisted locally
    assert (run_path / "config.yaml").read_text().startswith("exp:")

    # one file per entry, key subpaths preserved
    assert (run_path / "logs/0/images/generated/frame_0.png").is_file()
    assert (run_path / "logs/0/images/real/frame_0.png").is_file()
    assert torch.load(run_path / "logs/0/data/outputs_0.pt").shape == (1, 2)

    meta = json.loads((run_path / "logs/0/metadata/batch_0.json").read_text())
    assert meta == {"step": 0, "index": 0, "shape": [1, 2]}

    # metrics: one run-level CSV per key, one appended row per step
    assert (run_path / "logs/metrics/val/acc.csv").read_text() == "step,value\n0,0.75\n"
    assert (run_path / "logs/metrics/dummy_mean.csv").read_text() == "step,value\n1,-0.25\n"
    # metrics are no longer per-sample
    assert not (run_path / "logs/0/metrics").exists()
    assert not (run_path / "logs/summary/metrics").exists()
    assert not (run_path / "metrics.json").exists()

    # no MLflow file store, neither inside the run dir nor in the cwd
    assert not (run_path / "mlruns").exists()
    assert not (run_path / "mlflow.db").exists()
    assert not (Path.cwd() / "mlruns").exists()
    assert not (Path.cwd() / "mlflow.db").exists()


@pytest.mark.unit
def test_full_config_uploaded_even_when_params_fail(tmp_path, monkeypatch, caplog):
    """The complete config must reach the server as the config.yaml artifact even
    if the best-effort params call fails, and the artifact must be byte-identical
    to the local config.yaml (no truncation of any kind online)."""
    monkeypatch.delenv("TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    run_path = tmp_path / "run"
    logger = MLFlowLogger(run_path=run_path, root_config=None)
    logger.mlflow_enabled = True  # simulate server mode without a real server

    long_value = "x" * 700  # longer than the 500-char params cap
    config = OmegaConf.create({"exp": {"seed": 42, "long_value": long_value}})

    uploaded = []

    def _record_text(content, artifact_file=None, *args, **kwargs):
        uploaded.append((content, artifact_file))

    def _params_fail(*args, **kwargs):
        raise RuntimeError("backend rejected params batch")

    monkeypatch.setattr(mlflow, "log_text", _record_text)
    monkeypatch.setattr(mlflow, "log_params", _params_fail)

    logger._log_config(config)

    # config.yaml was uploaded exactly once, despite the params failure
    assert len(uploaded) == 1
    content, artifact_file = uploaded[0]
    assert artifact_file == "config.yaml"
    # full, untruncated content, byte-identical to the local file
    assert long_value in content
    assert content == (run_path / "config.yaml").read_text(encoding="utf-8")
    # the params failure was reported, not swallowed silently
    assert any("Failed to log params" in record.message for record in caplog.records)


@pytest.mark.unit
def test_metrics_are_series_not_per_id(tmp_path, monkeypatch):
    """Each metric key forms one series: a single MLflow series name (no sample
    id) and a single run-level CSV, stepped by the global call counter."""
    monkeypatch.delenv("TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    run_path = tmp_path / "run"
    logger = MLFlowLogger(run_path=run_path, root_config=None)
    logger.mlflow_enabled = True  # simulate server mode without a real server

    logged = []
    monkeypatch.setattr(
        mlflow, "log_metrics", lambda metrics, step=None: logged.append((metrics, step))
    )

    logger.log({"metrics/val/acc": torch.tensor(0.5)}, id="0")
    logger.log({"metrics/val/acc": 0.75, "metrics/other": 1}, id="1")

    # one series per key, no sample id in the name, global step per call
    assert logged == [({"val/acc": 0.5}, 0), ({"val/acc": 0.75, "other": 1}, 1)]
    # one CSV per key, appended row per step, independent of the sample id
    assert (run_path / "logs/metrics/val/acc.csv").read_text() == "step,value\n0,0.5\n1,0.75\n"
    assert (run_path / "logs/metrics/other.csv").read_text() == "step,value\n1,1\n"
    # no per-sample metric artifacts
    assert not (run_path / "logs/0/metrics").exists()
    assert not (run_path / "logs/1/metrics").exists()
