"""
MainWindow — orchestrates the GUI: assembles toolbar, parameter panel,
convergence chart, routes view, stats panel, log panel. Owns the
ExperimentWorker thread and routes its events from a thread-safe queue.
"""
from __future__ import annotations

import os
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, Optional

from src.experiment import BKS, Instance  # noqa: F401 — Instance imported lazily
from src.instance import Instance as _Instance

from interface import styles
from interface.workers.experiment_worker import ExperimentWorker
from interface.widgets.chart_widget import ChartWidget
from interface.widgets.log_panel import LogPanel
from interface.widgets.param_panel import ParamPanel
from interface.widgets.routes_view import RoutesView
from interface.widgets.stats_panel import StatsPanel


class MainWindow:
    """Top-level Tk controller.

    Wire-up::

        Tk root
        ├── Toolbar (instance, run/stop/reset, save log)
        ├── Body
        │   ├── ParamPanel   (algorithm knobs)
        │   └── Notebook
        │       ├── Convergence chart (matplotlib)
        │       └── Best solution routes (matplotlib)
        │   └── StatsPanel
        └── LogPanel

    Threading model: an ``ExperimentWorker`` thread runs the algorithm
    and pushes events into a thread-safe ``queue.Queue``. The main thread
    drains it every ``POLL_INTERVAL_MS`` ms via ``root.after``.
    """

    POLL_INTERVAL_MS = 50
    MAX_MSGS_PER_POLL = 30   # cap messages drained per cycle
    INSTANCES_DIR = "instances"
    CUSTOM_PREFIX = "📁 "  # prefix marking a user-loaded .vrp in the combo

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(
            "CVRP-IA · Algoritmo Immunologico (CLONALG) — Interfaccia Grafica"
        )
        self.root.geometry("1480x900")
        self.root.minsize(1180, 720)

        styles.apply_theme(root)

        # ---- State --------------------------------------------------------
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.worker: Optional[ExperimentWorker] = None
        self.best_known: Optional[float] = None
        self._custom_path: Optional[str] = None  # path when combo entry is custom
        self._current_instance: Optional[_Instance] = None

        self._build_layout()
        self._populate_instances()
        self._poll_queue()
        self.log_panel.log(
            "info",
            "Interfaccia pronta.  Seleziona un'istanza, regola i parametri "
            "e premi ▶ Avvia.",
        )

    # ====================================================================
    # Layout
    # ====================================================================

    def _build_layout(self) -> None:
        # -- Top toolbar --------------------------------------------------
        toolbar = ttk.Frame(self.root, style="Panel.TFrame", padding=(12, 8))
        toolbar.pack(side="top", fill="x")

        ttk.Label(toolbar, text="Istanza:", style="TLabel").pack(
            side="left", padx=(0, 4),
        )
        self.instance_var = tk.StringVar()
        self.instance_combo = ttk.Combobox(
            toolbar, textvariable=self.instance_var, width=24,
            state="readonly",
        )
        self.instance_combo.pack(side="left", padx=(0, 8))
        self.instance_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_instance_change()
        )

        self.bks_label = ttk.Label(toolbar, text="BKS: –", style="Muted.TLabel")
        self.bks_label.pack(side="left", padx=(0, 12))

        ttk.Button(
            toolbar, text="📂 Carica .vrp…",
            command=self._on_load_custom,
        ).pack(side="left", padx=(0, 8))

        ttk.Separator(toolbar, orient="vertical").pack(
            side="left", fill="y", padx=8,
        )

        self.run_button = ttk.Button(
            toolbar, text="▶  Avvia", style="Accent.TButton",
            command=self._on_run,
        )
        self.run_button.pack(side="left", padx=(0, 4))

        self.stop_button = ttk.Button(
            toolbar, text="■  Stop", style="Danger.TButton",
            command=self._on_stop, state="disabled",
        )
        self.stop_button.pack(side="left", padx=(0, 4))

        self.reset_button = ttk.Button(
            toolbar, text="↻  Reset parametri",
            command=self._on_reset_params,
        )
        self.reset_button.pack(side="left", padx=(0, 4))

        ttk.Separator(toolbar, orient="vertical").pack(
            side="left", fill="y", padx=8,
        )

        ttk.Button(
            toolbar, text="💾 Salva log",
            command=self._on_save_log,
        ).pack(side="left", padx=(0, 4))

        self.status_label = ttk.Label(
            toolbar, text="Pronto", style="Big.TLabel",
        )
        self.status_label.pack(side="right", padx=(4, 4))

        # -- Body ---------------------------------------------------------
        body = ttk.Frame(self.root)
        body.pack(side="top", fill="both", expand=True, padx=10, pady=8)

        # Left sidebar: parameters
        self.param_panel = ParamPanel(body, style="Panel.TFrame",
                                       padding=8, relief="ridge", borderwidth=1)
        self.param_panel.pack(side="left", fill="y", padx=(0, 8))

        # Center + right: charts and stats
        center = ttk.Frame(body)
        center.pack(side="left", fill="both", expand=True)

        # Tabbed area for the two visualizations
        notebook = ttk.Notebook(center)
        notebook.pack(fill="both", expand=True)

        chart_tab = ttk.Frame(notebook, style="Panel.TFrame")
        notebook.add(chart_tab, text="📈  Convergenza live")
        self.chart = ChartWidget(chart_tab)
        self.chart.pack(fill="both", expand=True, padx=4, pady=4)

        routes_tab = ttk.Frame(notebook, style="Panel.TFrame")
        notebook.add(routes_tab, text="🗺  Miglior soluzione")
        self.routes_view = RoutesView(routes_tab)
        self.routes_view.pack(fill="both", expand=True, padx=4, pady=4)

        # Stats panel below the notebook
        stats_box = ttk.Frame(center)
        stats_box.pack(fill="x", pady=(6, 0))
        self.stats_panel = StatsPanel(stats_box)
        self.stats_panel.pack(fill="x")

        # -- Bottom: Log --------------------------------------------------
        log_box = ttk.LabelFrame(
            self.root, text="Log eventi", padding=4,
        )
        log_box.pack(side="bottom", fill="x", padx=10, pady=(0, 10))
        self.log_panel = LogPanel(log_box)
        self.log_panel.pack(fill="both", expand=True)

    # ====================================================================
    # Instance discovery & selection
    # ====================================================================

    def _populate_instances(self) -> None:
        """Scan the ``instances/`` directory for available CVRP files.

        The combo values are instance basenames (without .vrp). Custom-
        loaded files are added with a ``📁 `` prefix.
        """
        names: list[str] = []
        if os.path.isdir(self.INSTANCES_DIR):
            for f in sorted(os.listdir(self.INSTANCES_DIR)):
                if f.lower().endswith(".vrp"):
                    names.append(f[:-4])
        self._combo_builtin_values = list(names)
        self._refresh_combo_values()
        if names:
            self.instance_var.set(names[0])
            self._on_instance_change()
        else:
            self.status_label.config(text="Nessuna istanza trovata")
            self.log_panel.log(
                "warning",
                f"Nessun file .vrp trovato in '{self.INSTANCES_DIR}/'. "
                "Usa '📂 Carica .vrp…' per selezionare un'istanza manualmente.",
            )

    def _refresh_combo_values(self) -> None:
        values = list(self._combo_builtin_values)
        if self._custom_path:
            custom_name = self.CUSTOM_PREFIX + os.path.basename(self._custom_path)
            values = [custom_name] + values
        self.instance_combo["values"] = values

    def _on_instance_change(self) -> None:
        name = self.instance_var.get()
        if not name:
            return
        if name.startswith(self.CUSTOM_PREFIX):
            path = self._custom_path or ""
            instance_label = os.path.basename(path)
        else:
            instance_label = name
            path = os.path.join(self.INSTANCES_DIR, f"{name}.vrp")
            if not os.path.exists(path):
                path = ""

        if path and os.path.exists(path):
            bks = BKS.get(name if not name.startswith(self.CUSTOM_PREFIX) else instance_label[:-4])
            self.best_known = bks
            bks_text = f"BKS: {bks}" if bks else "BKS: ?"
            self.bks_label.config(text=bks_text)
            self.status_label.config(text=f"Istanza: {instance_label}")
            # Pre-load to validate and stash it.
            try:
                self._current_instance = _Instance.from_file(path)
                self.stats_panel.set_n_customers(
                    self._current_instance.dimension - 1
                )
            except Exception as exc:  # noqa: BLE001
                self._current_instance = None
                self.log_panel.log(
                    "error", f"Impossibile caricare l'istanza: {exc}")
        else:
            self.best_known = None
            self.bks_label.config(text="BKS: ?")
            self.status_label.config(text="Istanza non valida")
            self._current_instance = None

    def _on_load_custom(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleziona un'istanza CVRP (.vrp)",
            filetypes=[("VRP files", "*.vrp"), ("All files", "*.*")],
        )
        if not path:
            return
        self._custom_path = path
        self._refresh_combo_values()
        self.instance_var.set(self.CUSTOM_PREFIX +
                              os.path.basename(path))
        self._on_instance_change()

    # ====================================================================
    # Run / Stop / Reset
    # ====================================================================

    def _on_run(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showwarning(
                "Attenzione",
                "Un esperimento è già in corso. Premi Stop per interromperlo.",
            )
            return

        instance_path = self._resolve_instance_path()
        if not instance_path or not os.path.exists(instance_path):
            messagebox.showerror(
                "Errore", f"File istanza non trovato:\n{instance_path}",
            )
            return

        # Validate parameters
        try:
            params = self.param_panel.get_params()
        except ValueError as exc:
            messagebox.showerror("Parametri non validi", str(exc))
            return

        # Validate / preload instance
        try:
            self._current_instance = _Instance.from_file(instance_path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(
                "Errore", f"Impossibile caricare l'istanza:\n{exc}",
            )
            return

        n_customers = self._current_instance.dimension - 1
        n_runs = int(params["n_runs"])
        max_fe = int(params["max_fe"])

        # Reset UI surfaces
        self.stats_panel.reset_runs(n_customers)
        self.chart.reset_runs(n_runs)
        self.chart.set_bks(self.best_known)
        self.routes_view.set_empty(
            f"Solver in esecuzione su {self._current_instance.name}…\n"
            f"{n_runs} run × {max_fe:,} FE."
        )
        self.stats_panel.update_current(
            run_id=f"1/{n_runs}",
            generazione=0,
            fe=0,
            max_fe=max_fe,
            best_cost=None,
            current_best=None,
            elapsed=0.0,
            progress_pct=0.0,
        )

        # Spin up worker
        worker_params: Dict[str, Any] = dict(params)
        worker_params["instance_path"] = instance_path
        worker_params["best_known"] = self.best_known
        worker_params["n_customers"] = n_customers

        self.worker = ExperimentWorker(worker_params, self.queue)
        self.worker.start()

        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text=f"In esecuzione: {self._current_instance.name}")
        self.log_panel.log(
            "heading",
            f"▶ Avvio esperimento: {self._current_instance.name} "
            f"({n_customers} clienti, capacità={self._current_instance.capacity})",
        )
        self.log_panel.log(
            "info",
            f"   Configurazione: N={n_runs}, FE={max_fe:,}, pop={params['pop_size']}, "
            f"clone={params['clone_factor']}, β={params['beta']}, "
            f"ρ={params['rho']}, d_n={params['replacement_rate']}",
        )

    def _on_stop(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            return
        self.log_panel.log("warning", "⏹ Stop richiesto — interruzione in corso…")
        self.worker.request_stop()
        self.stop_button.config(state="disabled")

    def _on_reset_params(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo(
                "In corso",
                "Stop l'esperimento prima di resettare i parametri.",
            )
            return
        self.param_panel.set_params(self.param_panel.get_defaults())
        self.log_panel.log("info", "↻ Parametri ripristinati ai valori di default.")

    def _on_save_log(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Salva log su file",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.log_panel.get_full_text())
            self.log_panel.log("success", f"💾 Log salvato in: {path}")
        except OSError as exc:
            messagebox.showerror("Errore di salvataggio", str(exc))

    # ====================================================================
    # Queue polling & event dispatch
    # ====================================================================

    def _poll_queue(self) -> None:
        """Drain the event queue, batching consecutive 'step' events.

        Only the *last* step event in a burst is dispatched to the UI,
        because intermediate stats updates are visually identical and
        just waste main-thread time.  Non-step events (run_start,
        new_best, run_end, etc.) are always dispatched immediately.
        """
        try:
            count = 0
            last_step_msg = None
            while count < self.MAX_MSGS_PER_POLL:
                msg = self.queue.get_nowait()
                count += 1
                if msg.get("event") == "step":
                    # Accumulate — only the latest step matters for the UI
                    last_step_msg = msg
                else:
                    # Flush any pending step before a non-step event
                    if last_step_msg is not None:
                        self._handle_message(last_step_msg)
                        last_step_msg = None
                    self._handle_message(msg)
        except queue.Empty:
            pass
        # Flush trailing step event (if the burst ended with steps)
        if last_step_msg is not None:
            self._handle_message(last_step_msg)
        self.root.after(self.POLL_INTERVAL_MS, self._poll_queue)

    def _handle_message(self, msg: Dict[str, Any]) -> None:
        event = msg.get("event", "?")

        if event == "experiment_start":
            self.stats_panel.update_current(
                run_id=f"1/{msg['n_runs']}",
                generazione=0,
                fe=0,
                max_fe=msg["max_fe"],
                best_cost=None,
                current_best=None,
                elapsed=0.0,
                progress_pct=0.0,
            )

        elif event == "run_start":
            self.stats_panel.update_current(
                run_id=f"{msg['run_id']}/{msg['n_runs']}",
                generazione=0,
                fe=0,
                max_fe=msg["max_fe"],
                best_cost=None,
                current_best=None,
                elapsed=0.0,
                progress_pct=0.0,
            )
            self.log_panel.log(
                "info",
                f"  ▶ Run {msg['run_id']}/{msg['n_runs']} avviata",
            )

        elif event == "step":
            fe = msg.get("fe", 0)
            fe_max = msg.get("max_fe", 1) or 1
            self.stats_panel.update_current(
                fe=fe,
                generazione=msg.get("generation", 0),
                best_cost=msg.get("best_cost"),
                current_best=msg.get("current_best"),
                elapsed=msg.get("elapsed_sec", 0.0),
                progress_pct=(fe / fe_max * 100.0),
            )
            if "history_point" in msg:
                hp = msg["history_point"]
                self.chart.add_point(msg["run_id"], int(hp[0]),
                                     float(hp[1]))
            # Live route updates during simulation (periodic snapshots)
            if "routes" in msg and self._current_instance is not None:
                self.routes_view.set_routes(
                    msg["routes"], self._current_instance,
                    extra_info={"best_cost": msg.get("best_cost"),
                                "bks": self.best_known},
                )

        elif event == "new_best":
            self.log_panel.log(
                "success",
                f"     🏆 Run {msg['run_id']}: nuovo best = "
                f"{msg['best_cost']:,.2f} (FE={msg.get('best_found_at_fe', '?')})",
            )
            if "routes" in msg and self._current_instance is not None:
                self.routes_view.set_routes(
                    msg["routes"], self._current_instance,
                    extra_info={"best_cost": msg["best_cost"],
                                "bks": self.best_known},
                )

        elif event == "run_end":
            self.stats_panel.add_run({
                "run_id": msg["run_id"],
                "best_cost": msg["best_cost"],
                "n_routes": msg["n_routes"],
                "customers_served": msg["customers_served"],
                "iterations_to_best": msg["iterations_to_best"],
                "elapsed_sec": msg["elapsed_sec"],
            })
            self.log_panel.log(
                "success",
                f"  ✅ Run {msg['run_id']} completata — best={msg['best_cost']:,.2f}, "
                f"{msg['n_routes']} rotte, {msg['elapsed_sec']:.1f}s",
            )

        elif event == "run_stopped":
            self.log_panel.log(
                "warning",
                f"     ⚠️  Run {msg.get('run_id', '?')} interrotta prima del completamento",
            )

        elif event == "experiment_end":
            self._on_experiment_end(msg)

        elif event == "log":
            level = msg.get("level", "info")
            self.log_panel.log(level, msg.get("message", ""))

        elif event == "error":
            self.log_panel.log("error", f"❌ {msg['message']}")
            tb = msg.get("traceback")
            if tb:
                for line in tb.splitlines():
                    self.log_panel.log("error", line)
            self.run_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.status_label.config(text="Errore")
            messagebox.showerror("Errore nello worker", msg["message"])

        else:
            self.log_panel.log("info", f"[{event}] {msg}")

    def _on_experiment_end(self, msg: Dict[str, Any]) -> None:
        # Restore UI buttons
        self.run_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.chart.force_redraw()

        bks = msg.get("best_known")
        gap_str: Optional[str] = None
        if bks and msg.get("best") is not None and bks > 0:
            gap_str = (msg["best"] - bks) / bks * 100.0

        self.stats_panel.set_final(
            best=msg.get("best"),
            mean=msg.get("mean"),
            std=msg.get("std"),
            gap=gap_str,
            time=msg.get("total_time"),
            satisf=msg.get("satisfability"),
        )

        cancelled = msg.get("cancelled", False)
        n_done = msg.get("n_runs_completed", 0)
        n_req = msg.get("n_runs_requested", n_done)
        prefix = "🏁 Esperimento concluso"
        if cancelled:
            prefix = "⏹ Esperimento interrotto"
        self.status_label.config(text=prefix)
        self.log_panel.log(
            "heading",
            f"{prefix}: {n_done}/{n_req} run completate · "
            f"best={msg.get('best', 0):,.2f} · "
            f"tempo totale={msg.get('total_time', 0):.1f}s",
        )
        # Display final routes of the last completed run if the final
        # best solution is still available.
        run_results = msg.get("run_results") or []
        if run_results and self._current_instance is not None:
            last_best_run = min(run_results, key=lambda r: r["best_cost"])
            # Re-route view with a textual summary only (no per-run
            # routes dict is shipped with run_end); the user can rerun
            # a single run if they want a visual of the routes.
            self.routes_view.set_empty(
                f"Esperimento terminato.\n"
                f"Miglior best-in-run: {last_best_run['best_cost']:,.2f}\n"
                f"({last_best_run['n_routes']} rotte, "
                f"{last_best_run['customers_served']} clienti serviti)."
            )

        self.worker = None

    # ====================================================================
    # Path helpers
    # ====================================================================

    def _resolve_instance_path(self) -> Optional[str]:
        """Compute the absolute path of the currently selected instance."""
        name = self.instance_var.get()
        if not name:
            return None
        if name.startswith(self.CUSTOM_PREFIX):
            return self._custom_path
        return os.path.join(self.INSTANCES_DIR, f"{name}.vrp")
