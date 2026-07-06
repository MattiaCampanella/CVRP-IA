"""
Stats panel — three sections:
  1. Live "current run" labels (updated on every step event)
  2. Treeview that accumulates one row per completed run
  3. Final aggregated stats once the experiment ends
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Dict


def _format(value: Any, spec: str) -> str:
    """Format helper used to keep labels compact and readable."""
    if value is None:
        return "-"
    if spec == "int":
        return f"{int(value):,}"
    if spec == "money":
        return f"{value:,.2f}"
    if spec == "percent":
        return f"{value:.2f}%"
    if spec == "seconds":
        return f"{value:.1f}"
    return str(value)


class StatsPanel(ttk.Frame):
    """Live stats display + completed-runs table + final summary."""

    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self._current_labels: Dict[str, ttk.Label] = {}
        self._final_labels: Dict[str, ttk.Label] = {}
        self._n_customers: int = 0
        self._build()

    # ----------------------------------------------------------------- UI

    def _build(self) -> None:
        # ----- Current run stats (top) -----
        current = ttk.LabelFrame(self, text="Stato corrente")
        current.pack(fill="x", padx=8, pady=(8, 4))

        current_specs = [
            ("run_id",       "Run #:"),            ("fe",           "FE correnti:"),
            ("generazione",  "Generazione:"),      ("max_fe",       "Budget FE:"),
            ("best_cost",    "Miglior costo:"),    ("current_best", "Pop best:"),
            ("elapsed",      "Tempo (s):"),        ("progress_pct", "Avanzamento:"),
        ]
        for i, (key, label_text) in enumerate(current_specs):
            r, c = divmod(i, 2)
            ttk.Label(current, text=label_text, style="Muted.TLabel").grid(
                row=r, column=c * 2, sticky="w", padx=(8, 4), pady=2,
            )
            lab = ttk.Label(
                current, text="-", style="Value.TLabel",
                width=14, anchor="e",
            )
            lab.grid(row=r, column=c * 2 + 1, sticky="e", padx=(0, 8), pady=2)
            self._current_labels[key] = lab

        # ----- Treeview (middle) -----
        summary = ttk.LabelFrame(self, text="Riepilogo run completate")
        summary.pack(fill="both", expand=True, padx=8, pady=(4, 4))

        columns = ("run_id", "best", "routes", "served",
                   "iter_to_best", "time_s")
        headings = {
            "run_id": ("Run #",        70),
            "best":   ("Best Cost",    110),
            "routes": ("# Rotte",      70),
            "served": ("# Serviti",    90),
            "iter_to_best": ("Iter → Best", 100),
            "time_s": ("Tempo (s)",    90),
        }
        self.tree = ttk.Treeview(
            summary, columns=columns, show="headings", height=7,
        )
        for col in columns:
            text, width = headings[col]
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor="center")

        scroll = ttk.Scrollbar(summary, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # ----- Final aggregate (bottom) -----
        final = ttk.Frame(self)
        final.pack(fill="x", padx=8, pady=(0, 8))
        final_specs = [
            ("best",   "Best:",   "money"),
            ("mean",   "Mean:",   "money"),
            ("std",    "Std:",    "money"),
            ("gap",    "Gap %:",  "percent"),
            ("satisf", "Sodd.:",  "raw"),
            ("time",   "Tot (s):", "seconds"),
        ]
        for i, (key, text, spec) in enumerate(final_specs):
            ttk.Label(final, text=text, style="Muted.TLabel").grid(
                row=0, column=i * 2, sticky="w", padx=(8, 4), pady=2,
            )
            lab = ttk.Label(final, text="-", style="Value.TLabel",
                            width=11, anchor="e")
            lab.grid(row=0, column=i * 2 + 1, sticky="e", padx=(0, 8), pady=2)
            self._final_labels[key] = (lab, spec)

    # ------------------------------------------------------------- API

    def update_current(self, **kwargs: Any) -> None:
        """Update one or more live stats. Values are pre-formatted strings
        or numbers; pass ``None`` to clear."""
        spec_map = {
            "run_id":       "raw",
            "generazione":  "int",
            "fe":           "int",
            "max_fe":       "int",
            "best_cost":    "money",
            "current_best": "money",
            "elapsed":      "seconds",
            "progress_pct": "percent",
        }
        for key, value in kwargs.items():
            if key not in self._current_labels:
                continue
            label = self._current_labels[key]
            label.config(text=_format(value, spec_map.get(key, "raw")))

    def reset_runs(self, n_customers: int = 0) -> None:
        """Clear the run table and the final aggregate labels."""
        for row in self.tree.get_children():
            self.tree.delete(row)
        self._n_customers = n_customers
        for lab, _ in self._final_labels.values():
            lab.config(text="-")
        for key in self._current_labels:
            if key == "max_fe":
                continue
            self._current_labels[key].config(text="-")

    def set_n_customers(self, n_customers: int) -> None:
        self._n_customers = n_customers

    def add_run(self, info: Dict[str, Any]) -> None:
        """Append a row for the just-completed run."""
        served = info.get("customers_served", "?")
        served_str = (
            f"{served}/{self._n_customers}" if self._n_customers else str(served)
        )
        values = (
            info.get("run_id", "?"),
            _format(info.get("best_cost"), "money"),
            info.get("n_routes", "?"),
            served_str,
            info.get("iterations_to_best", "?"),
            _format(info.get("elapsed_sec"), "seconds"),
        )
        self.tree.insert("", "end", values=values)

    def set_final(self, **kwargs: Any) -> None:
        """Populate the final aggregate row."""
        # Backwards-compatible with key name used in MainWindow
        for key, value in kwargs.items():
            if key not in self._final_labels:
                continue
            lab, spec = self._final_labels[key]
            lab.config(text=_format(value, spec))
