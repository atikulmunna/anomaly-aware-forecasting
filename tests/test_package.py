from pathlib import Path

import tomllib

from aaf import __version__
from aaf.data import SMDMachineSplit
from aaf.experiments import SuiteJob, collect_run_rows
from aaf.pipelines import SMDBaselineConfig, SMDJointConfig, SMDMDNConfig


def test_package_version_is_defined() -> None:
    assert __version__


def test_data_package_exports_smd_container() -> None:
    assert SMDMachineSplit.__name__ == "SMDMachineSplit"


def test_pipeline_package_exports_smd_configs() -> None:
    assert SMDBaselineConfig.__name__ == "SMDBaselineConfig"
    assert SMDJointConfig.__name__ == "SMDJointConfig"
    assert SMDMDNConfig.__name__ == "SMDMDNConfig"


def test_project_scripts_include_smd_entrypoints() -> None:
    project_root = Path(__file__).resolve().parents[1]
    payload = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))

    scripts = payload["project"]["scripts"]

    assert scripts["aaf-smd-baseline"] == "aaf.pipelines.smd_baseline:main"
    assert scripts["aaf-smd-joint"] == "aaf.pipelines.smd_joint:main"
    assert scripts["aaf-smd-mdn"] == "aaf.pipelines.smd_mdn:main"
    assert scripts["aaf-compare-runs"] == "aaf.experiments.cli:main"
    assert scripts["aaf-run-suite"] == "aaf.experiments.suite_cli:main"


def test_experiments_package_exports_collector() -> None:
    assert collect_run_rows.__name__ == "collect_run_rows"
    assert SuiteJob.__name__ == "SuiteJob"
