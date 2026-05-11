from aaf import __version__
from aaf.data import SMDMachineSplit
from aaf.pipelines import SMDBaselineConfig, SMDJointConfig


def test_package_version_is_defined() -> None:
    assert __version__


def test_data_package_exports_smd_container() -> None:
    assert SMDMachineSplit.__name__ == "SMDMachineSplit"


def test_pipeline_package_exports_smd_configs() -> None:
    assert SMDBaselineConfig.__name__ == "SMDBaselineConfig"
    assert SMDJointConfig.__name__ == "SMDJointConfig"
