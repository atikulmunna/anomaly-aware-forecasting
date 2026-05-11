from aaf import __version__
from aaf.data import SMDMachineSplit


def test_package_version_is_defined() -> None:
    assert __version__


def test_data_package_exports_smd_container() -> None:
    assert SMDMachineSplit.__name__ == "SMDMachineSplit"
