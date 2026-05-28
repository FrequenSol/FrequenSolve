"""Timing summaries and plots for frequency-domain job runs.

These helpers turn per-task solver metadata into rows suitable for notebooks,
reports, or matplotlib plots, including task durations, core-hour estimates,
and solver phase breakdowns when the fast solver records them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


class JobTimingMixin:
    """Timing table and plotting helpers for frequency-domain job runs.

    The mixin consumes per-task metadata collected by ``JobRunStateMixin`` and
    produces tabular summaries or matplotlib views of task and solver-phase
    runtimes.
    """

    def task_timings(self) -> List[Dict[str, Any]]:
        """Return frequency rows that include per-task runtime.

        Returns:
            List of timing rows containing task number, frequency,
            duration in seconds, optional core count/core-hours, status, and
            trace file.
        """

        rows = self.frequency_status()
        default_cores = self._default_core_count()
        timings = []
        for row in rows:
            duration = row.get("duration_seconds")
            if duration is None:
                continue
            core_count = self._record_core_count(row.get("metadata", {}))
            if core_count is None:
                core_count = default_cores
            duration = float(duration)
            core_hours = (
                duration * float(core_count) / 3600.0
                if core_count is not None
                else None
            )
            timings.append(
                {
                    "task": row["task"],
                    "frequency": row["frequency"],
                    "duration_seconds": duration,
                    "core_count": core_count,
                    "core_hours": core_hours,
                    "status": row["status"],
                    "trace_file": row["trace_file"],
                }
            )
        return timings

    def plot_task_timings(
        self,
        ax: Optional[Any] = None,
        *,
        x: str = "frequency",
        style: str = "auto",
        unit: str = "seconds",
        cores: Optional[float] = None,
        max_xticks: int = 8,
        show: bool = False,
        title: Optional[str] = None,
        **bar_kwargs: Any,
    ) -> Any:
        """Plot per-frequency task runtimes and return the matplotlib axes.

        ``unit`` may be ``"seconds"``, ``"hours"``, or ``"core-hours"``. For
        large frequency sweeps the default style switches from bars to lines so
        the axis stays readable.

        Args:
            ax: Optional matplotlib axes. A new axes is created when omitted.
            x: Horizontal axis, either ``"frequency"`` or ``"task"``.
            style: Plot style, one of ``"auto"``, ``"bar"``, or ``"line"``.
            unit: Runtime unit, one of ``"seconds"``, ``"hours"``, or
                ``"core-hours"``.
            cores: Optional core count override for core-hour plots.
            max_xticks: Maximum number of x-axis ticks.
            show: Whether to call ``plt.show()`` before returning.
            title: Optional plot title.
            **bar_kwargs: Additional plotting keyword arguments for bar plots.

        Returns:
            Matplotlib axes containing the plot.

        Raises:
            ValueError: If no task timings are available or an option is
                invalid.
        """

        timings = self.task_timings()
        if not timings:
            raise ValueError(
                "No per-task timings were found in run metadata for this job"
            )

        plt = self._matplotlib("Job timing plotting")
        if ax is None:
            _, ax = plt.subplots()

        xs, rows, integer_axis = self._timing_x_values(timings, x=x, ax=ax)
        values, ylabel = self._task_timing_values(rows, unit=unit, cores=cores)
        plot_style = self._timing_plot_style(style, rows)

        if plot_style == "bar":
            bar_kwargs.setdefault(
                "color",
                [
                    self._TASK_STATUS_COLORS.get(row["status"], "#546e7a")
                    for row in rows
                ],
            )
            ax.bar(
                xs,
                values,
                width=self._timing_bar_width(xs, integer_axis=integer_axis),
                **bar_kwargs,
            )
        else:
            marker = "o" if len(rows) <= 50 else None
            ax.plot(xs, values, marker=marker, linewidth=1.8, color="#2e7d32")

        self._finish_timing_plot(
            ax,
            ylabel=ylabel,
            title=title or f"{self.name} task timings",
            integer_axis=integer_axis,
            max_xticks=max_xticks,
        )
        if show:
            plt.show()
        return ax

    def phase_timings(
        self,
        phases: Optional[Sequence[str]] = None,
        *,
        include_zero: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return per-frequency solver phase timings from run metadata.

        The fast solver writes phase timings in ``_fs_run/timings.json`` when
        they are available. Rows returned by this method are keyed by task and
        frequency, with one numeric column per phase. Missing phases are filled
        with zero so the result is easy to tabulate or plot.

        Args:
            phases: Optional phase names to include and order explicitly.
            include_zero: Keep rows/phases whose values are zero.

        Returns:
            List of rows containing task, frequency, status, total seconds, and
            one column per phase.
        """

        requested = [str(phase) for phase in phases] if phases is not None else None
        rows_by_task = {row["task"]: row for row in self.frequency_status()}
        timing_records: Dict[int, Mapping[str, Any]] = {}
        phase_names: List[str] = []

        for record in self._task_records():
            phase_map = record.get("phases") or record.get("phase_timings")
            if not isinstance(phase_map, Mapping):
                continue
            task = self._task_number_from_record(record)
            if task is None:
                continue
            timing_records[task] = phase_map
            for phase in phase_map:
                phase = str(phase)
                if requested is not None and phase not in requested:
                    continue
                if phase not in phase_names:
                    phase_names.append(phase)

        if requested is not None:
            phase_names = requested

        rows: List[Dict[str, Any]] = []
        for task in sorted(timing_records):
            phase_map = timing_records[task]
            values: Dict[str, float] = {}
            total = 0.0
            for phase in phase_names:
                try:
                    seconds = float(phase_map.get(phase, 0.0))
                except (TypeError, ValueError):
                    seconds = 0.0
                if seconds != 0.0 or include_zero:
                    values[phase] = seconds
                total += seconds
            if not values and not include_zero:
                continue
            status_row = rows_by_task.get(task, {})
            rows.append(
                {
                    "task": task,
                    "frequency": status_row.get("frequency"),
                    "status": status_row.get("status"),
                    "total_seconds": total,
                    **values,
                }
            )
        return rows

    def plot_phase_timings(
        self,
        ax: Optional[Any] = None,
        *,
        phases: Optional[Sequence[str]] = None,
        x: str = "frequency",
        style: str = "auto",
        unit: str = "seconds",
        include_zero: bool = False,
        max_xticks: int = 8,
        show: bool = False,
        title: Optional[str] = None,
        colors: Optional[Mapping[str, str]] = None,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot stacked per-frequency solver phase timings.

        ``unit`` may be ``"seconds"`` or ``"hours"``. For long frequency
        sweeps the default style switches from stacked bars to one line per
        phase so the plot remains readable.

        Args:
            ax: Optional matplotlib axes. A new axes is created when omitted.
            phases: Optional phase names to include and order explicitly.
            x: Horizontal axis, either ``"frequency"`` or ``"task"``.
            style: Plot style, one of ``"auto"``, ``"bar"``, or ``"line"``.
            unit: Runtime unit, either ``"seconds"`` or ``"hours"``.
            include_zero: Include zero-valued phases in the table and plot.
            max_xticks: Maximum number of x-axis ticks.
            show: Whether to call ``plt.show()`` before returning.
            title: Optional plot title.
            colors: Optional mapping from phase name to matplotlib color.
            **plot_kwargs: Additional keyword arguments forwarded to the
                matplotlib plotting calls.

        Returns:
            Matplotlib axes containing the plot.

        Raises:
            ValueError: If no phase timings are available or an option is
                invalid.
        """

        rows = self.phase_timings(phases=phases, include_zero=include_zero)
        if not rows:
            raise ValueError(
                "No solver phase timings were found in run metadata for this job"
            )

        plt = self._matplotlib("Job phase timing plotting")
        if ax is None:
            _, ax = plt.subplots()

        xs, rows, integer_axis = self._timing_x_values(rows, x=x, ax=ax)
        scale, ylabel = self._phase_timing_scale(unit)
        phase_names = self._phase_names(
            rows,
            requested=phases,
            include_zero=include_zero,
        )
        plot_style = self._timing_plot_style(style, rows)
        palette = {**self._PHASE_COLORS, **dict(colors or {})}
        values_by_phase = {
            phase: np.asarray([float(row.get(phase, 0.0)) * scale for row in rows])
            for phase in phase_names
        }
        if not include_zero:
            values_by_phase = {
                phase: values
                for phase, values in values_by_phase.items()
                if np.any(values != 0.0)
            }
            phase_names = [phase for phase in phase_names if phase in values_by_phase]
            if not phase_names:
                raise ValueError("No nonzero solver phases were found for plotting")

        if plot_style == "bar":
            bottoms = np.zeros(len(rows), dtype=float)
            for phase in phase_names:
                values = values_by_phase[phase]
                ax.bar(
                    xs,
                    values,
                    bottom=bottoms,
                    width=self._timing_bar_width(xs, integer_axis=integer_axis),
                    label=phase.replace("_", " "),
                    color=palette.get(phase),
                    **plot_kwargs,
                )
                bottoms += values
        else:
            marker = "o" if len(rows) <= 50 else None
            for phase in phase_names:
                ax.plot(
                    xs,
                    values_by_phase[phase],
                    marker=marker,
                    linewidth=1.8,
                    label=phase.replace("_", " "),
                    color=palette.get(phase),
                    **plot_kwargs,
                )

        self._finish_timing_plot(
            ax,
            ylabel=ylabel,
            title=title or f"{self.name} solver phase timings",
            integer_axis=integer_axis,
            max_xticks=max_xticks,
        )
        ax.legend()
        if show:
            plt.show()
        return ax

    _TASK_STATUS_COLORS = {
        "succeeded": "#2e7d32",
        "failed": "#c62828",
        "not_run": "#757575",
    }
    _PHASE_COLORS = {
        "setup": "#546e7a",
        "mesh": "#00897b",
        "assembly": "#3949ab",
        "solve_forward": "#ef6c00",
        "solve_adjoint": "#8e24aa",
        "imaging": "#c62828",
    }

    @staticmethod
    def _matplotlib(feature: str) -> Any:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            from frequensolve._optional import optional_dependency_error

            raise optional_dependency_error(
                feature,
                extra="visual",
                dependencies=("matplotlib",),
                error=exc,
            ) from exc
        return plt

    @staticmethod
    def _timing_frequency_value(value: Any) -> Optional[float]:
        if isinstance(value, Mapping) and "real" in value:
            return float(value["real"])
        try:
            return float(np.real(value))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _timing_x_values(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        x: str,
        ax: Any,
    ) -> tuple[np.ndarray, List[Mapping[str, Any]], bool]:
        if x not in {"frequency", "task"}:
            raise ValueError("x must be 'frequency' or 'task'")

        x_values = [cls._timing_frequency_value(row["frequency"]) for row in rows]
        use_frequency_axis = x == "frequency" and all(
            value is not None for value in x_values
        )
        if use_frequency_axis:
            plot_rows = sorted(zip(x_values, rows), key=lambda item: float(item[0]))
            xs = np.asarray([float(value) for value, _row in plot_rows])
            ordered_rows = [row for _value, row in plot_rows]
            ax.set_xlabel("Frequency (Hz)")
            return xs, ordered_rows, False

        ax.set_xlabel("Frequency task")
        return np.asarray([row["task"] for row in rows], dtype=float), list(rows), True

    @staticmethod
    def _timing_plot_style(style: str, rows: Sequence[Mapping[str, Any]]) -> str:
        if style == "auto":
            return "bar" if len(rows) <= 80 else "line"
        if style in {"bar", "line"}:
            return style
        raise ValueError("style must be 'auto', 'bar', or 'line'")

    @staticmethod
    def _timing_bar_width(xs: np.ndarray, *, integer_axis: bool) -> float:
        if integer_axis or len(xs) <= 1:
            return 0.8
        diffs = np.diff(np.unique(xs))
        diffs = diffs[diffs > 0.0]
        if len(diffs):
            return 0.8 * float(np.min(diffs))
        return 0.8

    @staticmethod
    def _task_timing_values(
        rows: Sequence[Mapping[str, Any]],
        *,
        unit: str,
        cores: Optional[float],
    ) -> tuple[np.ndarray, str]:
        normalized_unit = unit.strip().lower().replace("_", "-")
        if normalized_unit in {"s", "sec", "secs", "second", "seconds"}:
            return (
                np.asarray([float(row["duration_seconds"]) for row in rows]),
                "Runtime (s)",
            )
        if normalized_unit in {"h", "hr", "hrs", "hour", "hours"}:
            return (
                np.asarray([float(row["duration_seconds"]) / 3600.0 for row in rows]),
                "Runtime (hours)",
            )
        if normalized_unit in {"core-hour", "core-hours", "core hour", "core hours"}:
            plot_values = []
            for row in rows:
                core_count = cores if cores is not None else row.get("core_count")
                if core_count is None:
                    raise ValueError(
                        "Core-hour plotting requires per-task core metadata or "
                        "a `cores=` override."
                    )
                plot_values.append(
                    float(row["duration_seconds"]) * float(core_count) / 3600.0
                )
            return np.asarray(plot_values), "Runtime (core-hours)"
        raise ValueError("unit must be 'seconds', 'hours', or 'core-hours'")

    @staticmethod
    def _phase_timing_scale(unit: str) -> tuple[float, str]:
        normalized_unit = unit.strip().lower().replace("_", "-")
        if normalized_unit in {"s", "sec", "secs", "second", "seconds"}:
            return 1.0, "Runtime (s)"
        if normalized_unit in {"h", "hr", "hrs", "hour", "hours"}:
            return 1.0 / 3600.0, "Runtime (hours)"
        raise ValueError("unit must be 'seconds' or 'hours'")

    @staticmethod
    def _phase_names(
        rows: Sequence[Mapping[str, Any]],
        *,
        requested: Optional[Sequence[str]],
        include_zero: bool,
    ) -> List[str]:
        phase_names = list(requested) if requested is not None else []
        if phase_names:
            return phase_names
        for row in rows:
            for key, value in row.items():
                if key in {"task", "frequency", "status", "total_seconds"}:
                    continue
                if not include_zero and float(value) == 0.0:
                    continue
                if key not in phase_names:
                    phase_names.append(key)
        if not phase_names:
            raise ValueError("No nonzero solver phases were found for plotting")
        return phase_names

    @staticmethod
    def _finish_timing_plot(
        ax: Any,
        *,
        ylabel: str,
        title: str,
        integer_axis: bool,
        max_xticks: int,
    ) -> None:
        from matplotlib.ticker import MaxNLocator

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=max(2, int(max_xticks)), integer=integer_axis)
        )
        ax.grid(axis="y", alpha=0.25)
