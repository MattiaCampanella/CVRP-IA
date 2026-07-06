"""
Routes visualization widget — draws the best solution's routes on the
customer coordinates of the instance being solved.

Depot is rendered as a red square, customers as blue dots, and each
route is drawn with a different color (cycling through ``tab10``).
Redraws are throttled to roughly once per second so the GUI stays
responsive when route updates arrive at high frequency.

Vehicle animation: coloured dots move along each route path in a
continuous loop driven by ``tk.after`` callbacks (~20 FPS). Each
vehicle has a phase offset so they are spread evenly along the routes.

Performance: the animation uses **matplotlib blitting** — the static
background (routes, nodes, axes) is rendered once and cached; each
frame only redraws the vehicle scatter artist on top of the cached
background, reducing per-frame cost from ~30–50 ms to ~2–3 ms.
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk
from typing import Any, List, Optional

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402

import numpy as np  # noqa: E402


class RoutesView(ttk.Frame):
    """Visualize the depot, customers, and each route of the best solution."""

    REDRAW_THROTTLE_S = 0.8   # max ~1.25 redraws/s
    ANIM_FRAME_MS    = 50     # ~20 FPS for vehicle animation (was 33)
    ANIM_SPEED       = 0.12   # full cycle in ~8.3 seconds

    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self._instances_served: int = 0
        self._last_draw = 0.0
        self._pending: Optional[tuple] = None  # (routes, instance, extra_info)
        self._deferred_scheduled = False

        # ---- Vehicle animation state ---------------------------------
        self._anim_id: Optional[str] = None     # current after() callback id
        self._anim_paths: List[np.ndarray] = []  # (N,2) per-route path coords
        self._anim_cumd: List[np.ndarray] = []   # cumulative distances per route
        self._anim_total: List[float] = []        # total length per route
        self._vehicle_scatter: Any = None         # scatter artist for vehicles
        self._anim_t0: float = 0.0               # wall-clock start of anim cycle

        # ---- Blitting state ------------------------------------------
        self._bg_cache: Any = None               # cached background bitmap
        self._canvas_size: tuple = (0, 0)        # (w, h) to detect resizes

        self._build()
        self.set_empty("Nessuna soluzione disponibile.\nAvvia un esperimento per vedere le rotte.")

    # ----------------------------------------------------------------- UI

    def _build(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(7, 4.2), dpi=100)
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------- API

    def set_empty(self, message: str = "Nessuna soluzione disponibile") -> None:
        """Show a placeholder message instead of any routes."""
        self._stop_animation()
        self._vehicle_scatter = None
        self._bg_cache = None
        self.ax.clear()
        self.ax.text(
            0.5, 0.5, message, ha="center", va="center",
            transform=self.ax.transAxes,
            fontsize=11, color="#6b7280",
            wrap=True,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect("equal", adjustable="datalim")
        self.canvas.draw_idle()

    def set_routes(
        self,
        routes: List[List[int]],
        instance: Any,
        extra_info: Optional[dict] = None,
    ) -> None:
        """Draw depot, customers and the given routes of the best solution.

        If called faster than ``REDRAW_THROTTLE_S``, the update is deferred.
        """
        now = time.time()
        if now - self._last_draw < self.REDRAW_THROTTLE_S:
            # Defer: store the latest request, redraw later
            self._pending = (routes, instance, extra_info)
            # Schedule a deferred redraw via Tk's event loop
            if not self._deferred_scheduled:
                self._deferred_scheduled = True
                self.after(
                    int(self.REDRAW_THROTTLE_S * 1000),
                    self._flush_deferred,
                )
            return
        self._deferred_scheduled = False
        self._pending = None
        self._do_draw(routes, instance, extra_info)
        self._last_draw = now

    # ------------------------------------------------------------- animation

    def _capture_background(self) -> None:
        """Render the canvas fully and cache the background for blitting.

        Must be called AFTER the static plot has been drawn and the canvas
        has been fully rendered (i.e. after ``canvas.draw()``).
        """
        try:
            self._bg_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            w = self.fig.canvas.get_width_height()[0]
            h = self.fig.canvas.get_width_height()[1]
            self._canvas_size = (w, h)
        except Exception:  # noqa: BLE001
            # copy_from_bbox can fail if the canvas hasn't been fully rendered
            self._bg_cache = None

    def _start_animation(
        self, routes: List[List[int]], coords: np.ndarray, depot: int,
    ) -> None:
        """Precompute path geometry and launch the vehicle-animation loop.

        Each route's full cycle goes depot→customers→depot.  Vehicles
        are assigned evenly-spaced phase offsets so they do not all
        cluster at the depot at the same time.
        """
        self._stop_animation()
        n = len(routes)
        if n == 0:
            return

        cmap = plt.get_cmap("tab10")
        self._anim_paths = []
        self._anim_cumd = []
        self._anim_total = []
        colors: List[Any] = []

        for i, route in enumerate(routes):
            if not route:
                continue
            pts_np = np.array(
                [coords[depot]] + [coords[c] for c in route] + [coords[depot]],
                dtype=float,
            )
            diffs = np.diff(pts_np, axis=0)
            seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
            cumd = np.concatenate(([0.0], np.cumsum(seg_lens)))
            total = cumd[-1]
            self._anim_paths.append(pts_np)
            self._anim_cumd.append(cumd)
            self._anim_total.append(total)
            colors.append(cmap(i % 10))

        if not self._anim_paths:
            return

        # Create scatter artist for vehicles (one dot per route)
        # Start all vehicles at the depot (first point of each path).
        init_xs = [p[0, 0] for p in self._anim_paths]
        init_ys = [p[0, 1] for p in self._anim_paths]
        self._vehicle_scatter = self.ax.scatter(
            init_xs, init_ys,
            c=colors, s=70, zorder=6, marker="o",
            edgecolors="white", linewidths=1.0, alpha=0.95,
        )

        # Full draw to render everything, then capture background for blitting
        self.canvas.draw()
        self._capture_background()

        self._anim_t0 = time.time()
        self._anim_id = self.after(self.ANIM_FRAME_MS, self._animate_frame)

    def _stop_animation(self) -> None:
        """Cancel any running vehicle animation."""
        if self._anim_id is not None:
            self.after_cancel(self._anim_id)
            self._anim_id = None

    def _animate_frame(self) -> None:
        """Advance vehicles along their paths and schedule the next frame.

        Uses matplotlib blitting: restores the cached background, redraws
        only the vehicle scatter artist, and blits the result. This avoids
        the full ``canvas.draw_idle()`` which would re-render all axes,
        grids, tick labels, routes, etc.
        """
        if self._vehicle_scatter is None or not self._anim_paths:
            self._anim_id = None
            return

        elapsed = time.time() - self._anim_t0
        n = len(self._anim_paths)

        xs = np.empty(n, dtype=float)
        ys = np.empty(n, dtype=float)

        for i in range(n):
            total = self._anim_total[i]
            if total <= 0:
                xs[i] = self._anim_paths[i][0, 0]
                ys[i] = self._anim_paths[i][0, 1]
                continue
            # Evenly-spaced phase offsets so vehicles don't overlap
            phase = (i / max(n, 1)) * total
            dist = (elapsed * self.ANIM_SPEED * total + phase) % total
            # Locate segment via binary search on cumulative distances
            cumd = self._anim_cumd[i]
            pts = self._anim_paths[i]
            idx = np.searchsorted(cumd, dist, side="right") - 1
            idx = max(0, min(idx, len(pts) - 2))
            seg_start = cumd[idx]
            seg_len = cumd[idx + 1] - seg_start
            t = ((dist - seg_start) / seg_len) if seg_len > 0 else 0.0
            t = max(0.0, min(1.0, t))
            xs[i] = pts[idx, 0] + t * (pts[idx + 1, 0] - pts[idx, 0])
            ys[i] = pts[idx, 1] + t * (pts[idx + 1, 1] - pts[idx, 1])

        self._vehicle_scatter.set_offsets(np.column_stack((xs, ys)))

        # ---- Blitting path (fast) ------------------------------------
        # Check if canvas was resized → background cache is invalid
        try:
            cur_size = self.fig.canvas.get_width_height()
        except Exception:  # noqa: BLE001
            cur_size = self._canvas_size

        if self._bg_cache is not None and cur_size == self._canvas_size:
            # Fast path: restore background, redraw only the scatter, blit
            self.fig.canvas.restore_region(self._bg_cache)
            self.ax.draw_artist(self._vehicle_scatter)
            self.fig.canvas.blit(self.ax.bbox)
        else:
            # Fallback: full redraw + recapture (happens on resize)
            self.canvas.draw()
            self._capture_background()

        self._anim_id = self.after(self.ANIM_FRAME_MS, self._animate_frame)

    def _flush_deferred(self) -> None:
        """Draw the most recently deferred route update (if any)."""
        self._deferred_scheduled = False
        pending = self._pending
        self._pending = None
        if pending is not None:
            routes, instance, extra_info = pending
            self._do_draw(routes, instance, extra_info)
            self._last_draw = time.time()

    def _do_draw(
        self,
        routes: List[List[int]],
        instance: Any,
        extra_info: Optional[dict] = None,
    ) -> None:
        """Internal draw method — no throttling, just drawing."""
        self._stop_animation()
        self._bg_cache = None
        self._vehicle_scatter = None
        self.ax.clear()

        coords = np.asarray(instance.coordinates, dtype=float)
        depot = int(instance.depot)

        # Customers (excluding depot)
        if len(coords) > 0:
            mask = np.ones(len(coords), dtype=bool)
            mask[depot] = False
            self.ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c="#3b82f6", s=22, zorder=3, label="Clienti",
                edgecolors="white", linewidths=0.4,
            )
            # Depot
            self.ax.scatter(
                coords[depot, 0], coords[depot, 1],
                c="#dc2626", s=180, marker="s", zorder=5, label="Deposito",
                edgecolors="white", linewidths=0.8,
            )

        # Routes
        cmap = plt.get_cmap("tab10")
        n_routes = len(routes) if routes else 0
        for i, route in enumerate(routes or []):
            color = cmap(i % 10)
            pts = [coords[depot]] + [coords[c] for c in route] + [coords[depot]]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            self.ax.plot(
                xs, ys, color=color, linewidth=1.0, alpha=0.85,
                label=f"Rotta {i+1} ({len(route)} clienti)" if i < 10 else None,
            )

        # Title and caption
        title = f"Miglior soluzione — {n_routes} rotte"
        if instance is not None:
            title += f"  ·  {instance.name}"
        self.ax.set_title(title, fontsize=10, fontweight="bold")

        info_parts = []
        if extra_info:
            if "best_cost" in extra_info:
                info_parts.append(f"Costo: {extra_info['best_cost']:,.2f}")
            if "bks" in extra_info and extra_info["bks"]:
                info_parts.append(f"BKS: {extra_info['bks']:,}")
                if "best_cost" in extra_info:
                    gap = (extra_info["best_cost"] - extra_info["bks"]) \
                          / extra_info["bks"] * 100
                    info_parts.append(f"Gap: {gap:.2f}%")
        if info_parts:
            self.ax.text(
                0.99, 0.01, "  ·  ".join(info_parts),
                transform=self.ax.transAxes,
                ha="right", va="bottom",
                fontsize=9, color="#1f2937",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#eef2f7", edgecolor="#e2e8f0"),
            )

        # Legend only if at most 8 routes (otherwise it's overwhelming)
        if 0 < n_routes <= 8:
            self.ax.legend(loc="upper left", fontsize=7, ncol=2)

        self.ax.set_xlabel("X", fontsize=9)
        self.ax.set_ylabel("Y", fontsize=9)
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True, alpha=0.2)

        # ---- Launch vehicle animation --------------------------------
        self._start_animation(routes, coords, depot)
