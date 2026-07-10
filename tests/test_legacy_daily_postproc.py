from pathlib import Path
import sys

import pytest


LEGACY_DIRECTORY = Path(__file__).resolve().parents[1] / "legacy"
sys.path.insert(0, str(LEGACY_DIRECTORY))
from daily_postproc import DailyInputError, complete_daily_inputs  # noqa: E402


def _write_day(directory, day, times=range(8), prefixes=("output_tau", "output_Tb", "am_input_file")):
    directory.mkdir(parents=True, exist_ok=True)
    for prefix in prefixes:
        for time in times:
            (directory / "{0}_day{1:02d}_time{2}.txt".format(prefix, day, time)).touch()


def test_discovers_a_complete_daily_group(tmp_path):
    _write_day(tmp_path / "2020" / "02", 29)

    groups = complete_daily_inputs(tmp_path / "2020", "02")

    assert len(groups) == 1
    assert groups[0].day == 29
    assert len(groups[0].tau) == 8
    assert groups[0].tau[0].name.endswith("time0.txt")
    assert groups[0].tau[-1].name.endswith("time7.txt")


def test_skips_a_year_month_with_no_inputs(tmp_path):
    assert complete_daily_inputs(tmp_path / "2021", "02") == []


def test_rejects_partial_daily_inputs(tmp_path):
    directory = tmp_path / "2021" / "02"
    _write_day(directory, 28, times=range(7))

    with pytest.raises(DailyInputError, match="Incomplete daily inputs"):
        complete_daily_inputs(tmp_path / "2021", "02")
