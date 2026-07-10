"""Validated discovery of legacy three-hour weather inputs for daily averages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


NATIVE_SAMPLES_PER_DAY = 8
_FILENAME = re.compile(
    r"^(?:output_tau|output_Tb|am_input_file)_day(?P<day>\d{2})_time(?P<time>\d)\.txt$"
)


class DailyInputError(RuntimeError):
    """Raised when a day has incomplete or inconsistent raw weather inputs."""


@dataclass(frozen=True)
class DailyInputFiles:
    """The complete three-hour inputs needed to aggregate one calendar day."""

    day: int
    tau: tuple[Path, ...]
    tb: tuple[Path, ...]
    am: tuple[Path, ...]


def _day_and_time(path: Path) -> tuple[int, int]:
    match = _FILENAME.fullmatch(path.name)
    if match is None:
        raise DailyInputError("Unexpected weather input filename: {0}".format(path))
    return int(match.group("day")), int(match.group("time"))


def _files_by_day_and_time(directory: Path, prefix: str) -> dict[int, dict[int, Path]]:
    files: dict[int, dict[int, Path]] = {}
    for path in sorted(directory.glob("{0}_day*_time*.txt".format(prefix))):
        day, time = _day_and_time(path)
        if time < 0 or time >= NATIVE_SAMPLES_PER_DAY:
            raise DailyInputError("Unexpected time index in {0}".format(path))
        by_time = files.setdefault(day, {})
        if time in by_time:
            raise DailyInputError("Duplicate time index in {0}".format(path))
        by_time[time] = path
    return files


def complete_daily_inputs(year_directory: str | Path, month: str) -> list[DailyInputFiles]:
    """Return only complete daily groups, rejecting partial source days.

    A year may legitimately have no data for a requested month. In that case,
    return an empty list rather than manufacturing an all-zero daily record.
    """

    directory = Path(year_directory) / month
    tau_by_day = _files_by_day_and_time(directory, "output_tau")
    tb_by_day = _files_by_day_and_time(directory, "output_Tb")
    am_by_day = _files_by_day_and_time(directory, "am_input_file")
    days = sorted(set(tau_by_day) | set(tb_by_day) | set(am_by_day))
    expected_times = set(range(NATIVE_SAMPLES_PER_DAY))
    groups: list[DailyInputFiles] = []

    for day in days:
        tau_times = set(tau_by_day.get(day, {}))
        tb_times = set(tb_by_day.get(day, {}))
        am_times = set(am_by_day.get(day, {}))
        if tau_times != expected_times or tb_times != expected_times or am_times != expected_times:
            raise DailyInputError(
                "Incomplete daily inputs for {0}-{1} day {2:02d}: "
                "tau={3}, Tb={4}, am={5}.".format(
                    year_directory,
                    month,
                    day,
                    len(tau_times),
                    len(tb_times),
                    len(am_times),
                )
            )
        groups.append(
            DailyInputFiles(
                day=day,
                tau=tuple(tau_by_day[day][time] for time in range(NATIVE_SAMPLES_PER_DAY)),
                tb=tuple(tb_by_day[day][time] for time in range(NATIVE_SAMPLES_PER_DAY)),
                am=tuple(am_by_day[day][time] for time in range(NATIVE_SAMPLES_PER_DAY)),
            )
        )
    return groups
