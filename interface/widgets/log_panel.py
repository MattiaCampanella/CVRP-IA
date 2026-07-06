"""
Log panel — colored scrolled text widget showing live algorithm events.

Limit the buffer to ``MAX_LINES`` so memory usage stays bounded on long
experiments.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from collections import deque
from typing import Deque


class LogPanel(ttk.Frame):
    """Auto-scrolling colored log widget."""

    MAX_LINES = 1000

    # Color tags (kept in sync with the palette in styles.py)
    _TAG_STYLES = {
        "info":    {"foreground": "#1f2937"},
        "heading": {"foreground": "#2563eb", "font": ("Consolas", 9, "bold")},
        "success": {"foreground": "#16a34a"},
        "warning": {"foreground": "#d97706"},
        "error":   {"foreground": "#dc2626",
                    "font": ("Consolas", 9, "bold")},
    }

    def __init__(self, master: tk.Misc, **kwargs: tk.Any) -> None:
        super().__init__(master, **kwargs)
        self._buffer: Deque[str] = deque(maxlen=self.MAX_LINES)
        self._build()

    # ----------------------------------------------------------------- UI

    def _build(self) -> None:
        self.text = tk.Text(
            self,
            wrap="word",
            height=10,
            state="disabled",
            background="#f8fafc",
            foreground="#1f2937",
            font=("Consolas", 9),
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=8,
            pady=6,
        )
        scroll = ttk.Scrollbar(self, orient="vertical",
                               command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        for tag, style in self._TAG_STYLES.items():
            self.text.tag_configure(tag, **style)

    # ------------------------------------------------------------- API

    def log(self, level: str, message: str) -> None:
        """Append a line at the given level. Level: info|heading|success|warning|error."""
        tag = level if level in self._TAG_STYLES else "info"
        line = message + "\n"
        self._buffer.append(line)

        self.text.config(state="normal")
        self.text.insert("end", line, tag)
        # Trim the widget if it's overflowing the buffer
        line_count = int(self.text.index("end-1c").split(".")[0])
        excess = line_count - self.MAX_LINES
        if excess > 0:
            self.text.delete("1.0", f"{excess + 1}.0")
        self.text.see("end")
        self.text.config(state="disabled")

    def clear(self) -> None:
        """Erase all log content."""
        self._buffer.clear()
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.config(state="disabled")

    def get_full_text(self) -> str:
        """Return the entire log content as a string."""
        return self.text.get("1.0", "end-1c")
