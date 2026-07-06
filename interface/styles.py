"""
ttk Style configuration for the CVRP-IA interface.

A clean, light, clam-based theme with custom colors and a monospace
value font. Call ``apply_theme(root)`` once after the Tk root is
created.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLORS = {
    "bg":            "#f5f7fa",   # app background
    "panel_bg":      "#ffffff",   # card/panel background
    "panel_alt":     "#eef2f7",   # alt rows
    "accent":        "#2563eb",   # primary blue
    "accent_hover":  "#1d4ed8",
    "success":       "#16a34a",
    "warning":       "#d97706",
    "error":         "#dc2626",
    "text":          "#1f2937",
    "muted":         "#6b7280",
    "border":        "#e2e8f0",
}

# Choose a sensible system font; fall back gracefully if missing.
_FONT_CANDIDATES = ["Segoe UI", "Helvetica Neue", "Helvetica", "Arial"]
_MONO_CANDIDATES = ["Consolas", "Menlo", "DejaVu Sans Mono", "Courier New"]


def _pick_font(candidates: list[str]) -> str:
    """Return the first available font name, or the first candidate."""
    try:
        from tkinter import font as tkfont
        available = set(tkfont.families())
        for c in candidates:
            if c in available:
                return c
    except Exception:
        pass
    return candidates[0]


UI_FONT_FAMILY = _pick_font(_FONT_CANDIDATES)
MONO_FONT_FAMILY = _pick_font(_MONO_CANDIDATES)

FONTS = {
    "heading": (UI_FONT_FAMILY, 11, "bold"),
    "label":   (UI_FONT_FAMILY, 10),
    "value":   (MONO_FONT_FAMILY, 10, "bold"),
    "small":   (UI_FONT_FAMILY, 9),
    "log":     (MONO_FONT_FAMILY, 9),
    "big":     (UI_FONT_FAMILY, 14, "bold"),
}


def apply_theme(root: tk.Tk) -> None:
    """Configure ttk styles for the application."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass  # clam unavailable, keep default

    # Frame / LabelFrame / Notebook base colors
    style.configure(".", background=COLORS["bg"], foreground=COLORS["text"])
    style.configure("TFrame",           background=COLORS["bg"])
    style.configure("Panel.TFrame",     background=COLORS["panel_bg"])
    style.configure("TLabelFrame",      background=COLORS["bg"],
                                        bordercolor=COLORS["border"])
    style.configure("TLabelFrame.Label", background=COLORS["bg"],
                                        foreground=COLORS["text"],
                                        font=FONTS["heading"])

    style.configure("TNotebook",        background=COLORS["bg"],
                                        borderwidth=0)
    style.configure("TNotebook.Tab",    padding=(14, 6), font=FONTS["label"])
    style.map("TNotebook.Tab",
              background=[("selected", COLORS["panel_bg"])],
              foreground=[("selected", COLORS["accent"])])

    # Labels
    style.configure("TLabel",
                    background=COLORS["bg"],
                    foreground=COLORS["text"],
                    font=FONTS["label"])
    style.configure("Heading.TLabel",
                    background=COLORS["bg"],
                    foreground=COLORS["text"],
                    font=FONTS["heading"])
    style.configure("Muted.TLabel",
                    background=COLORS["bg"],
                    foreground=COLORS["muted"],
                    font=FONTS["small"])
    style.configure("Value.TLabel",
                    background=COLORS["panel_bg"],
                    foreground=COLORS["accent"],
                    font=FONTS["value"])
    style.configure("Big.TLabel",
                    background=COLORS["bg"],
                    foreground=COLORS["accent"],
                    font=FONTS["big"])

    # Buttons
    style.configure("TButton",
                    padding=(10, 6),
                    font=FONTS["label"])
    style.configure("Accent.TButton",
                    padding=(14, 8),
                    font=FONTS["heading"],
                    foreground="white",
                    background=COLORS["accent"])
    style.map("Accent.TButton",
              background=[("active", COLORS["accent_hover"]),
                          ("disabled", COLORS["muted"])],
              foreground=[("disabled", "#e5e7eb")])
    style.configure("Danger.TButton",
                    padding=(14, 8),
                    font=FONTS["heading"],
                    foreground="white",
                    background=COLORS["error"])
    style.map("Danger.TButton",
              background=[("active", "#b91c1c"),
                          ("disabled", COLORS["muted"])],
              foreground=[("disabled", "#e5e7eb")])

    # Inputs
    style.configure("TEntry",
                    padding=4,
                    fieldbackground=COLORS["panel_bg"],
                    bordercolor=COLORS["border"])
    style.configure("TSpinbox",
                    padding=4,
                    arrowsize=14)
    style.configure("TCombobox",
                    padding=4,
                    fieldbackground=COLORS["panel_bg"])

    # Treeview
    style.configure("Treeview",
                    rowheight=24,
                    background=COLORS["panel_bg"],
                    fieldbackground=COLORS["panel_bg"],
                    font=FONTS["small"])
    style.configure("Treeview.Heading",
                    background=COLORS["panel_alt"],
                    foreground=COLORS["text"],
                    font=FONTS["heading"],
                    relief="flat")
    style.map("Treeview",
              background=[("selected", COLORS["accent"])],
              foreground=[("selected", "white")])

    style.configure("Separator.Horizontal.TSeparator", background=COLORS["border"])

    # Vertical scrollbar a bit slimmer
    style.configure("Vertical.TScrollbar", background=COLORS["panel_alt"],
                    bordercolor=COLORS["border"], arrowcolor=COLORS["muted"])
