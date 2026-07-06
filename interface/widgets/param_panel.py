"""
Parameter panel — labeled spinboxes / entries for every algorithm knob.

Each parameter has a (key, label, type, default, min, max, step) tuple.
``get_params()`` returns the validated dictionary ready to be passed
to ``run_experiment``.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Tuple


# (key, label_text, type, default, min, max, step)
_PARAM_SPEC: List[Tuple[str, str, type, Any, Any, Any, Any]] = [
    ("n_runs",             "Numero di run (N):",     int,   5,         1,   50,    1),
    ("max_fe",             "Budget FE per run:",     int,   350_000,   1_000, 5_000_000, 1_000),
    ("pop_size",           "Popolazione:",           int,   100,       10,  1000,   10),
    ("clone_factor",       "Clone factor:",          float, 10.0,      0.1, 200.0,  0.1),
    ("beta",               "Beta (frazione selez.):",float, 0.2,       0.01, 1.0,    0.01),
    ("rho",                "Rho (mutazione):",       float, 3.0,       0.1, 20.0,   0.1),
    ("replacement_rate",   "Replacement rate:",      float, 0.2,       0.0, 1.0,    0.01),
    ("local_search_top_k", "Local search top-K:",    int,   10,        0,   100,    1),
    ("seed",               "Seed base:",             int,   1,         0,   999_999, 1),
]


class ParamPanel(ttk.Frame):
    """Sidebar form with labeled spinboxes for all parameters."""

    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self._vars: Dict[str, Tuple[tk.Variable, type, Any, Any]] = {}
        self._build()

    # ------------------------------------------------------------------ UI

    def _build(self) -> None:
        header = ttk.Label(
            self, text="Parametri", style="Heading.TLabel"
        )
        header.grid(row=0, column=0, columnspan=2, sticky="w",
                    padx=8, pady=(8, 6))

        sub = ttk.Label(
            self,
            text="Configurazione CLONALG e protocollo sperimentale.",
            style="Muted.TLabel", wraplength=230, justify="left",
        )
        sub.grid(row=1, column=0, columnspan=2, sticky="w",
                 padx=8, pady=(0, 6))

        ttk.Separator(self, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 6)
        )

        row = 3
        for spec in _PARAM_SPEC:
            key, label, typ, default, mn, mx, step = spec
            ttk.Label(self, text=label, style="TLabel").grid(
                row=row, column=0, sticky="w", padx=(8, 6), pady=3,
            )
            var: tk.Variable
            var = tk.DoubleVar(value=float(default)) if typ is float \
                  else tk.IntVar(value=int(default))
            self._vars[key] = (var, typ, mn, mx)

            spin = ttk.Spinbox(
                self,
                textvariable=var,
                from_=mn,
                to=mx,
                increment=step,
                width=10,
                justify="right",
            )
            spin.grid(row=row, column=1, sticky="ew", padx=(0, 8), pady=3)
            row += 1

        # Bottom: per-param hint strip
        self.columnconfigure(1, weight=1)

    # ------------------------------------------------------------- helpers

    def get_params(self) -> Dict[str, Any]:
        """Validate and return the current parameter values as a dict.

        Raises:
            ValueError: if any parameter is invalid.
        """
        result: Dict[str, Any] = {}
        for key, (var, typ, mn, mx) in self._vars.items():
            try:
                value = var.get()
            except tk.TclError as exc:
                raise ValueError(
                    f"Parametro '{key}' non valido: {exc}"
                ) from exc
            if typ is int:
                v = int(round(float(value)))
            else:
                v = float(value)
            if not (mn <= v <= mx):
                raise ValueError(
                    f"Parametro '{key}': {v} fuori range [{mn}, {mx}]"
                )
            result[key] = v
        return result

    def set_params(self, params: Dict[str, Any]) -> None:
        """Bulk-load parameter values (used by Reset/Defaults)."""
        for key, value in params.items():
            if key in self._vars:
                var, typ, _, _ = self._vars[key]
                if typ is int:
                    var.set(int(value))
                else:
                    var.set(float(value))

    def get_defaults(self) -> Dict[str, Any]:
        """Return the default parameter values (for Reset button)."""
        return {spec[0]: spec[3] for spec in _PARAM_SPEC}
