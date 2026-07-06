"""
Convergence chart widget — matplotlib Figure embedded in a Tk Frame.

Each run gets a colored line; an optional BKS reference line is drawn
horizontally. Updates are throttled to roughly 5 redraws per second so
the GUI stays responsive even with frequent progress callbacks.

Performance note: xs/ys are stored in separate flat lists so appending
a single point is O(1). ``line.set_data`` is called only when a redraw
actually happens, avoiding O(n) work per incoming point.
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402


class ChartWidget(ttk.Frame):
    """Live convergence chart with one line per run."""

    REDRAW_THROTTLE_S = 0.50  # ~2 FPS — sufficient for convergence plots

    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self._run_lines: Dict[int, Any] = {}
        self._run_xs: Dict[int, List[int]] = {}
        self._run_ys: Dict[int, List[float]] = {}
        self._run_colors: Dict[int, Any] = {}
        self._run_dirty: Dict[int, bool] = {}
        self._bks_line: Optional[Any] = None
        self._last_bks: Optional[float] = None
        self._last_draw = 0.0
        self._build()

    # ----------------------------------------------------------------- UI

    def _build(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(7, 4.2), dpi=100)
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.10)
        self.ax.set_xlabel("Fitness Evaluations (FE)")
        self.ax.set_ylabel("Miglior costo trovato")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        # No navigation toolbar — keep UI simple and uncluttered
        self.canvas.draw_idle()
        self._last_draw = time.time()

    # ------------------------------------------------------------- API

    def reset_runs(self, n_runs: int) -> None:
        """Clear the chart and prepare for ``n_runs`` convergence curves."""
        self.ax.clear()
        self.ax.set_xlabel("Fitness Evaluations (FE)")
        self.ax.set_ylabel("Miglior costo trovato")
        self.ax.grid(True, alpha=0.3)

        self._run_lines.clear()
        self._run_xs.clear()
        self._run_ys.clear()
        self._run_colors.clear()
        self._run_dirty.clear()
        self._bks_line = None

        if n_runs > 0:
            cmap = plt.get_cmap("tab10")
            for i in range(1, n_runs + 1):
                color = cmap((i - 1) % 10)
                self._run_colors[i] = color
                (line,) = self.ax.plot(
                    [], [], color=color, alpha=0.75, linewidth=1.2,
                    label=f"Run {i}",
                )
                self._run_lines[i] = line
                self._run_xs[i] = []
                self._run_ys[i] = []
                self._run_dirty[i] = False

        if self._last_bks is not None:
            self._bks_line = self.ax.axhline(
                y=self._last_bks, color="red", linestyle="--",
                linewidth=1.5, alpha=0.7, label=f"BKS ({self._last_bks})",
            )

        if self._run_lines or self._bks_line is not None:
            self.ax.legend(loc="upper right", fontsize=8)

        self.canvas.draw_idle()
        self._last_draw = time.time()

    def set_bks(self, bks: Optional[float]) -> None:
        """Set or clear the BKS reference horizontal line."""
        if self._bks_line is not None:
            self._bks_line.remove()
            self._bks_line = None
        self._last_bks = bks
        if bks is not None:
            self._bks_line = self.ax.axhline(
                y=bks, color="red", linestyle="--",
                linewidth=1.5, alpha=0.7, label=f"BKS ({bks:g})",
            )
            if self._run_lines:
                self.ax.legend(loc="upper right", fontsize=8)
            self.canvas.draw_idle()
            self._last_draw = time.time()

    def add_point(self, run_id: int, fe: int, cost: float) -> None:
        """Append a single (FE, cost) point for ``run_id``. O(1), no redraw.

        The line data is only pushed to matplotlib on the next throttled
        redraw so that a burst of 100 points only costs one ``set_data`` call.
        """
        if run_id not in self._run_xs:
            return
        self._run_xs[run_id].append(fe)
        self._run_ys[run_id].append(cost)
        self._run_dirty[run_id] = True
        self._maybe_redraw()

    def add_points(self, run_id: int, points: List[Tuple[int, float]]) -> None:
        """Bulk-append multiple (FE, cost) points (used when loading history)."""
        if run_id not in self._run_xs or not points:
            return
        for fe, cost in points:
            self._run_xs[run_id].append(fe)
            self._run_ys[run_id].append(cost)
        self._run_dirty[run_id] = True
        self._maybe_redraw(force=True)

    def _maybe_redraw(self, force: bool = False) -> None:
        now = time.time()
        if force or ((now - self._last_draw) >= self.REDRAW_THROTTLE_S):
            # Push dirty line data to matplotlib lines
            for run_id, dirty in self._run_dirty.items():
                if dirty:
                    line = self._run_lines[run_id]
                    line.set_data(self._run_xs[run_id], self._run_ys[run_id])
                    self._run_dirty[run_id] = False
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw_idle()
            self._last_draw = now

    def force_redraw(self) -> None:
        """Final redraw after an experiment ends — push all dirty data."""
        for run_id, dirty in self._run_dirty.items():
            if dirty:
                line = self._run_lines[run_id]
                line.set_data(self._run_xs[run_id], self._run_ys[run_id])
                self._run_dirty[run_id] = False
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()
        self._last_draw = time.time()
