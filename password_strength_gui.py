import math
import random
import string
import threading
import tkinter as tk
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack

from feature_engineering import (
    build_feature_matrix,
    has_sequential_pattern,
    load_common_passwords,
    password_features,
    shannon_entropy,
)
from password_strength import predict_password, train_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "password_strength_xgboost.joblib"

PALETTE = {
    "bg": "#090B12",
    "bg_2": "#0D111C",
    "card": "#111827",
    "card_2": "#151D2B",
    "surface": "#0D1422",
    "surface_2": "#172033",
    "stroke": "#29364A",
    "stroke_soft": "#202B3E",
    "stroke_focus": "#8B5CF6",
    "text": "#F8FAFC",
    "muted": "#A7B1C2",
    "subtle": "#728096",
    "blue": "#4F7CFF",
    "purple": "#8B5CF6",
    "cyan": "#22D3EE",
    "green": "#34D399",
    "yellow": "#FBBF24",
    "red": "#FB7185",
    "shadow": "#050814",
}

STRENGTH = {
    0: {"label": "Weak", "color": PALETTE["red"], "badge": "#3A1420"},
    1: {"label": "Medium", "color": PALETTE["yellow"], "badge": "#33270F"},
    2: {"label": "Strong", "color": PALETTE["green"], "badge": "#0F3024"},
}

FONT = "Inter"
FALLBACK_FONT = "Segoe UI"


def font(size, weight="normal"):
    return (FONT, size, weight)


def hex_to_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb


def blend(start, end, amount):
    start_rgb = hex_to_rgb(start)
    end_rgb = hex_to_rgb(end)
    return rgb_to_hex(tuple(int(start_rgb[i] + (end_rgb[i] - start_rgb[i]) * amount) for i in range(3)))


def rounded_rect(canvas, x1, y1, x2, y2, radius, **kwargs):
    points = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


def format_seconds(seconds):
    if seconds < 1:
        return "Instant"

    units = [
        ("centuries", 60 * 60 * 24 * 365 * 100),
        ("years", 60 * 60 * 24 * 365),
        ("days", 60 * 60 * 24),
        ("hours", 60 * 60),
        ("minutes", 60),
        ("seconds", 1),
    ]
    for label, unit in units:
        if seconds >= unit:
            value = seconds / unit
            if value > 999:
                return f"{value:.1e} {label}"
            return f"{value:.1f} {label}"
    return "Instant"


def estimate_crack_time(password):
    pool = 0
    pool += 26 if any(char.islower() for char in password) else 0
    pool += 26 if any(char.isupper() for char in password) else 0
    pool += 10 if any(char.isdigit() for char in password) else 0
    pool += 33 if any(not char.isalnum() for char in password) else 0
    if not password or pool == 0:
        return "Instant"

    entropy_bits = len(password) * math.log2(pool)
    guesses_per_second = 10_000_000_000
    seconds = (2 ** max(entropy_bits - 1, 0)) / guesses_per_second
    return format_seconds(seconds)


class PremiumButton(tk.Canvas):
    def __init__(self, parent, text, command, icon="", width=260, height=56, variant="primary"):
        super().__init__(parent, width=width, height=height, bg=PALETTE["card"], highlightthickness=0, bd=0)
        self.text = text
        self.icon = icon
        self.command = command
        self.width = width
        self.height = height
        self.variant = variant
        self.enabled = True
        self.hovered = False
        self.pressed = False
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Button-1>", self._click)
        self.draw()

    def draw(self):
        self.delete("all")
        inset = 2 if not self.pressed else 4
        radius = 18

        if not self.enabled:
            rounded_rect(self, inset, inset, self.width - inset, self.height - inset, radius, fill="#263249", outline="#334155", width=1)
            text_color = "#7D8AA0"
        elif self.variant == "primary":
            start = "#5B8CFF" if self.hovered else PALETTE["blue"]
            end = "#A075FF" if self.hovered else PALETTE["purple"]
            steps = max(1, self.width - inset * 2)
            for index in range(steps):
                color = blend(start, end, index / steps)
                self.create_line(inset + index, inset + 2, inset + index, self.height - inset - 2, fill=color)
            rounded_rect(self, inset, inset, self.width - inset, self.height - inset, radius, fill="", outline="#8FB4FF", width=1)
            self.create_line(28, inset + 2, self.width - 28, inset + 2, fill="#BFD3FF", width=1)
            text_color = PALETTE["text"]
        else:
            fill = "#1A2436" if self.hovered else "#141D2D"
            rounded_rect(self, inset, inset, self.width - inset, self.height - inset, radius, fill=fill, outline=PALETTE["stroke"], width=1)
            self.create_line(22, inset + 2, self.width - 22, inset + 2, fill="#2D3B54", width=1)
            text_color = PALETTE["text"]

        label = f"{self.icon}  {self.text}" if self.icon else self.text
        self.create_text(
            self.width / 2,
            self.height / 2,
            text=label,
            fill=text_color,
            font=font(11, "bold"),
        )

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.draw()

    def _enter(self, _event):
        self.hovered = True
        self.configure(cursor="hand2")
        self.draw()

    def _leave(self, _event):
        self.hovered = False
        self.pressed = False
        self.configure(cursor="")
        self.draw()

    def _press(self, event):
        if self.enabled:
            self.pressed = True
            self.draw()
            self._ripple(event.x, event.y, 4)

    def _release(self, _event):
        self.pressed = False
        self.draw()

    def _click(self, _event):
        if self.enabled:
            self.command()

    def _ripple(self, x, y, radius):
        if radius > 34 or not self.pressed:
            return
        self.delete("ripple")
        self.create_oval(x - radius, y - radius, x + radius, y + radius, outline="#D7E3FF", width=1, tags="ripple")
        self.after(18, self._ripple, x, y, radius + 5)


class CircularMeter(tk.Canvas):
    def __init__(self, parent, size=178):
        super().__init__(parent, width=size, height=size, bg=PALETTE["card"], highlightthickness=0, bd=0)
        self.size = size
        self.score = 0
        self.color = PALETTE["cyan"]
        self.draw(0, self.color)

    def draw(self, score, color):
        self.score = max(0, min(100, int(score)))
        self.color = color
        self.delete("all")
        pad = 16
        self.create_oval(pad, pad, self.size - pad, self.size - pad, outline="#26344E", width=13)
        extent = -360 * (self.score / 100)
        self.create_arc(
            pad,
            pad,
            self.size - pad,
            self.size - pad,
            start=90,
            extent=extent,
            outline=color,
            width=13,
            style="arc",
        )
        self.create_oval(34, 34, self.size - 34, self.size - 34, fill="#0B1020", outline="#1F2A44", width=1)
        self.create_text(
            self.size / 2,
            self.size / 2 - 8,
            text=str(self.score),
            fill=PALETTE["text"],
            font=font(31, "bold"),
        )
        self.create_text(
            self.size / 2,
            self.size / 2 + 28,
            text="score",
            fill=PALETTE["muted"],
            font=font(10),
        )

    def animate_to(self, target, color):
        start = self.score
        steps = 18
        delta = (target - start) / steps

        def step(index=1):
            value = start + delta * index
            self.draw(value, color)
            if index < steps:
                self.after(16, step, index + 1)

        step()


class PremiumPasswordApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Password Strength AI")
        self.geometry("1224x824")
        self.minsize(1080, 744)
        self.configure(bg=PALETTE["bg"])

        self.model = None
        self.vectorizer = None
        self.model_name = "Loading"
        self.common_passwords = load_common_passwords()
        self.password_visible = False
        self.password_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Status: Loading model")
        self.result_label_var = tk.StringVar(value="Awaiting password")
        self.badge_var = tk.StringVar(value="Idle")
        self.crack_var = tk.StringVar(value="Estimated crack time: --")
        self.entropy_var = tk.StringVar(value="Entropy: --")
        self.length_var = tk.StringVar(value="Length: --")
        self.confidence_var = tk.StringVar(value="Confidence: --")
        self.placeholder = "Enter or generate a password"
        self.placeholder_active = False

        self._build_shell()
        self._set_ready(False)
        self._start_loading()

    def _build_shell(self):
        self.bg_canvas = tk.Canvas(self, bg=PALETTE["bg"], highlightthickness=0, bd=0)
        self.bg_canvas.pack(fill="both", expand=True)
        self.bg_canvas.bind("<Configure>", self._paint_background)

        self.content = tk.Frame(self.bg_canvas, bg=PALETTE["bg"])
        self.window_id = self.bg_canvas.create_window(0, 0, anchor="nw", window=self.content)
        self.content.bind("<Configure>", lambda _event: self.bg_canvas.configure(scrollregion=self.bg_canvas.bbox("all")))

        self._build_header()
        self._build_dashboard()
        self._build_status_bar()

    def _paint_background(self, event):
        self.bg_canvas.delete("bg")
        width, height = event.width, event.height
        self.bg_canvas.coords(self.window_id, 0, 0)
        self.bg_canvas.itemconfigure(self.window_id, width=width, height=height)
        for i in range(36):
            color = PALETTE["bg"] if i % 2 else "#0A0D16"
            self.bg_canvas.create_rectangle(0, i * height / 36, width, (i + 1) * height / 36, fill=color, outline="", tags="bg")
        self.bg_canvas.create_oval(width - 430, -190, width + 100, 320, fill="#10245A", outline="", tags="bg")
        self.bg_canvas.create_oval(-180, height - 330, 330, height + 160, fill="#211052", outline="", tags="bg")
        self.bg_canvas.create_oval(width * 0.44, height - 260, width * 0.44 + 360, height + 90, fill="#073142", outline="", tags="bg")
        self.bg_canvas.tag_lower("bg")

    def _build_header(self):
        header = tk.Frame(self.content, bg=PALETTE["bg"])
        header.pack(fill="x", padx=32, pady=(24, 16))

        title_block = tk.Frame(header, bg=PALETTE["bg"])
        title_block.pack(side="left", fill="x", expand=True)

        tk.Label(
            title_block,
            text="Password Strength AI",
            bg=PALETTE["bg"],
            fg=PALETTE["text"],
            font=font(32, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_block,
            text="Local AI password analysis with instant security feedback.",
            bg=PALETTE["bg"],
            fg=PALETTE["muted"],
            font=font(11),
        ).pack(anchor="w", pady=(4, 0))

        pill = tk.Canvas(header, width=188, height=42, bg=PALETTE["bg"], highlightthickness=0)
        pill.pack(side="right")
        rounded_rect(pill, 2, 2, 186, 40, 18, fill="#101827", outline=PALETTE["stroke"], width=1)
        pill.create_oval(18, 15, 28, 25, fill=PALETTE["green"], outline="")
        pill.create_text(98, 21, text="Protected by AI", fill=PALETTE["text"], font=font(10, "bold"))

    def _card(self, parent, title, subtitle=None):
        outer = tk.Frame(parent, bg=PALETTE["bg"])
        canvas = tk.Canvas(outer, bg=PALETTE["bg"], highlightthickness=0, bd=0)
        canvas.pack(fill="both", expand=True)
        body = tk.Frame(canvas, bg=PALETTE["card"])
        window = canvas.create_window(0, 0, anchor="nw", window=body)
        state = {"hover": False}

        def redraw(event):
            canvas.delete("shape")
            width = getattr(event, "width", canvas.winfo_width())
            height = getattr(event, "height", canvas.winfo_height())
            lift = 2 if state["hover"] else 0
            canvas.coords(window, 16, 16 - lift)
            canvas.itemconfigure(window, width=max(1, width - 32), height=max(1, height - 32))
            rounded_rect(canvas, 10, 14, width - 10, height - 6, 18, fill=PALETTE["shadow"], outline="", tags="shape")
            rounded_rect(canvas, 8, 8 - lift, width - 12, height - 12 - lift, 18, fill=PALETTE["card"], outline=PALETTE["stroke"], width=1, tags="shape")
            rounded_rect(canvas, 10, 10 - lift, width - 14, height - 14 - lift, 16, fill="", outline=PALETTE["stroke_soft"], width=1, tags="shape")
            canvas.tag_lower("shape")

        canvas.bind("<Configure>", redraw)
        canvas.bind("<Enter>", lambda event: (state.update({"hover": True}), redraw(event)))
        canvas.bind("<Leave>", lambda event: (state.update({"hover": False}), redraw(event)))

        tk.Label(body, text=title, bg=PALETTE["card"], fg=PALETTE["text"], font=font(17, "bold")).pack(anchor="w", padx=24, pady=(22, 2))
        if subtitle:
            tk.Label(body, text=subtitle, bg=PALETTE["card"], fg=PALETTE["muted"], font=font(10), wraplength=420, justify="left").pack(anchor="w", padx=24, pady=(0, 16))
        return outer, body

    def _build_dashboard(self):
        dashboard = tk.Frame(self.content, bg=PALETTE["bg"])
        dashboard.pack(fill="both", expand=True, padx=24)
        dashboard.columnconfigure(0, weight=3)
        dashboard.columnconfigure(1, weight=2)
        dashboard.rowconfigure(0, weight=1)
        dashboard.rowconfigure(1, weight=1)

        input_card, input_body = self._card(
            dashboard,
            "Password Input",
            "Type a password, generate one, or run an instant model check.",
        )
        input_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        self._build_input_card(input_body)

        guide_card, guide_body = self._card(
            dashboard,
            "Security Intelligence",
            "Guidance calibrated for stronger, less predictable passwords.",
        )
        guide_card.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(8, 0), pady=(0, 8))
        self._build_guide_card(guide_body)

        result_card, result_body = self._card(
            dashboard,
            "Result",
            "Score, crack-time estimate, entropy, and actionable feedback.",
        )
        result_card.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(8, 8))
        self._build_result_card(result_body)

    def _build_input_card(self, parent):
        entry_shell = tk.Frame(parent, bg=PALETTE["card"])
        entry_shell.pack(fill="x", padx=24, pady=(0, 14))

        self.entry_canvas = tk.Canvas(entry_shell, height=64, bg=PALETTE["card"], highlightthickness=0, bd=0)
        self.entry_canvas.pack(fill="x")
        self.entry_canvas.bind("<Configure>", self._draw_entry_shell)

        self.password_entry = tk.Entry(
            self.entry_canvas,
            textvariable=self.password_var,
            show="*",
            bg="#0B1020",
            fg=PALETTE["text"],
            insertbackground=PALETTE["cyan"],
            relief="flat",
            bd=0,
            font=font(16),
        )
        self.entry_window = self.entry_canvas.create_window(22, 32, anchor="w", window=self.password_entry, height=38)
        self.eye_button = tk.Button(
            self.entry_canvas,
            text="◉",
            command=self.toggle_password,
            bg=PALETTE["surface_2"],
            fg=PALETTE["muted"],
            activebackground="#1F2C46",
            activeforeground=PALETTE["text"],
            relief="flat",
            bd=0,
            font=font(12, "bold"),
            cursor="hand2",
        )
        self.eye_window = self.entry_canvas.create_window(0, 32, anchor="e", window=self.eye_button, width=48, height=34)
        self.password_entry.bind("<FocusIn>", self._on_entry_focus)
        self.password_entry.bind("<FocusOut>", self._on_entry_blur)
        self.password_entry.bind("<Return>", lambda _event: self.check_password())
        self.password_var.trace_add("write", lambda *_args: self._update_live_state())

        actions = tk.Frame(parent, bg=PALETTE["card"])
        actions.pack(fill="x", padx=24, pady=(0, 22))

        self.generator_button = PremiumButton(actions, "Generate", self.generate_password, icon="✦", width=176, height=56, variant="secondary")
        self.generator_button.pack(side="left")

        self.check_button = PremiumButton(actions, "Check Password Strength", self.check_password, icon="⌁", width=272, height=56, variant="primary")
        self.check_button.pack(side="right")

        helper = tk.Label(
            parent,
            text="Live feedback updates as you type. Model inference runs locally on your machine.",
            bg=PALETTE["card"],
            fg=PALETTE["subtle"],
            font=font(9),
        )
        helper.pack(anchor="w", padx=24, pady=(0, 22))

        self.breakdown_frame = tk.Frame(parent, bg=PALETTE["card"])
        self.breakdown_frame.pack(fill="x", padx=24, pady=(0, 22))
        self.breakdown_labels = {}
        for index, item in enumerate(["Uppercase", "Lowercase", "Numbers", "Symbols"]):
            tile = tk.Frame(self.breakdown_frame, bg=PALETTE["surface"], padx=14, pady=12)
            tile.grid(row=index // 2, column=index % 2, sticky="ew", padx=4, pady=4)
            self.breakdown_frame.columnconfigure(index % 2, weight=1)
            label = tk.Label(tile, text=f"○ {item}", bg=PALETTE["surface"], fg=PALETTE["muted"], font=font(10, "bold"))
            label.pack(anchor="w")
            self.breakdown_labels[item] = label

        self._set_placeholder()

    def _draw_entry_shell(self, event=None):
        width = self.entry_canvas.winfo_width()
        if width <= 1:
            width = 520
        focused = self.focus_get() == self.password_entry
        self.entry_canvas.delete("shell")
        outline = PALETTE["stroke_focus"] if focused else PALETTE["stroke"]
        if focused:
            rounded_rect(self.entry_canvas, 0, 2, width, 62, 20, fill="", outline="#312E81", width=2, tags="shell")
        rounded_rect(self.entry_canvas, 3, 4, width - 3, 60, 18, fill="#0B1020", outline=outline, width=1, tags="shell")
        self.entry_canvas.create_line(24, 6, width - 24, 6, fill="#1F2D46", width=1, tags="shell")
        self.entry_canvas.coords(self.eye_window, width - 16, 32)
        self.entry_canvas.itemconfigure(self.entry_window, width=max(120, width - 94))
        self.entry_canvas.tag_lower("shell")

    def _set_placeholder(self):
        if not self.password_var.get():
            self.placeholder_active = True
            self.password_entry.configure(show="", fg=PALETTE["subtle"])
            self.password_var.set(self.placeholder)

    def _on_entry_focus(self, _event):
        if self.placeholder_active:
            self.placeholder_active = False
            self.password_var.set("")
            self.password_entry.configure(show="" if self.password_visible else "*", fg=PALETTE["text"])
        self._draw_entry_shell()

    def _on_entry_blur(self, _event):
        self._set_placeholder()
        self._draw_entry_shell()

    def _actual_password(self):
        return "" if self.placeholder_active else self.password_var.get()

    def _build_guide_card(self, parent):
        guide = tk.Frame(parent, bg=PALETTE["card"])
        guide.pack(fill="both", expand=True, padx=24, pady=(0, 22))

        for icon, label, body, color in [
            ("!", "Weak", "Short, common, or predictable.", PALETTE["red"]),
            ("~", "Medium", "Good start, but needs more entropy.", PALETTE["yellow"]),
            ("✓", "Strong", "Long, diverse, and harder to guess.", PALETTE["green"]),
        ]:
            self._info_card(guide, icon, label, body, color)

        self._tip_section(guide, "Security Tips", ["Use 12+ characters", "Avoid dictionary words", "Use a password manager"])
        self._tip_section(guide, "Password Best Practices", ["Prefer unique passphrases", "Mix classes intentionally", "Rotate exposed credentials"])
        self._tip_section(guide, "Common Mistakes", ["Reusing passwords", "Keyboard sequences", "Names and birth years"])

    def _info_card(self, parent, icon, label, body, color):
        row = tk.Frame(parent, bg=PALETTE["surface"], padx=14, pady=12)
        row.pack(fill="x", pady=4)
        row.bind("<Enter>", lambda _event: row.configure(bg=PALETTE["surface_2"]))
        row.bind("<Leave>", lambda _event: row.configure(bg=PALETTE["surface"]))
        dot = tk.Canvas(row, width=32, height=32, bg=PALETTE["surface"], highlightthickness=0)
        dot.pack(side="left", padx=(0, 12))
        rounded_rect(dot, 1, 1, 31, 31, 10, fill=blend(color, "#111827", 0.72), outline=color, width=1)
        dot.create_text(16, 16, text=icon, fill=color, font=font(11, "bold"))
        text = tk.Frame(row, bg=PALETTE["surface"])
        text.pack(side="left", fill="x", expand=True)
        tk.Label(text, text=label, bg=PALETTE["surface"], fg=PALETTE["text"], font=font(10, "bold")).pack(anchor="w")
        tk.Label(text, text=body, bg=PALETTE["surface"], fg=PALETTE["muted"], font=font(9)).pack(anchor="w")

    def _tip_section(self, parent, title, items):
        tk.Label(parent, text=title, bg=PALETTE["card"], fg=PALETTE["text"], font=font(12, "bold")).pack(anchor="w", pady=(18, 6))
        for item in items:
            tk.Label(parent, text=f"✓ {item}", bg=PALETTE["card"], fg=PALETTE["muted"], font=font(9)).pack(anchor="w", pady=3)

    def _build_result_card(self, parent):
        result = tk.Frame(parent, bg=PALETTE["card"])
        result.pack(fill="both", expand=True, padx=24, pady=(0, 22))
        result.columnconfigure(1, weight=1)

        self.meter = CircularMeter(result, size=160)
        self.meter.grid(row=0, column=0, rowspan=5, padx=(0, 24), sticky="n")

        top = tk.Frame(result, bg=PALETTE["card"])
        top.grid(row=0, column=1, sticky="ew")
        tk.Label(top, textvariable=self.result_label_var, bg=PALETTE["card"], fg=PALETTE["text"], font=font(24, "bold")).pack(side="left")
        self.badge = tk.Label(top, textvariable=self.badge_var, bg="#17233A", fg=PALETTE["cyan"], font=font(9, "bold"), padx=12, pady=5)
        self.badge.pack(side="left", padx=(14, 0))

        self.progress_canvas = tk.Canvas(result, height=18, bg=PALETTE["card"], highlightthickness=0, bd=0)
        self.progress_canvas.grid(row=1, column=1, sticky="ew", pady=(12, 10))
        self.progress_canvas.bind("<Configure>", lambda _event: self._draw_progress(0, PALETTE["cyan"]))

        metrics = tk.Frame(result, bg=PALETTE["card"])
        metrics.grid(row=2, column=1, sticky="ew")
        for column in range(4):
            metrics.columnconfigure(column, weight=1)
        for index, var in enumerate([self.confidence_var, self.entropy_var, self.crack_var, self.length_var]):
            tk.Label(
                metrics,
                textvariable=var,
                bg=PALETTE["surface"],
                fg=PALETTE["muted"],
                font=font(8, "bold"),
                padx=8,
                pady=8,
            ).grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 4, 0))

        classes = tk.Frame(result, bg=PALETTE["card"])
        classes.grid(row=3, column=1, sticky="ew", pady=(10, 0))
        for column in range(4):
            classes.columnconfigure(column, weight=1)
        self.result_class_labels = {}
        for index, item in enumerate(["Uppercase", "Lowercase", "Numbers", "Symbols"]):
            label = tk.Label(classes, text=f"○ {item}", bg="#101A2B", fg=PALETTE["muted"], font=font(8, "bold"), padx=8, pady=7)
            label.grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 4, 0))
            self.result_class_labels[item] = label

        lower = tk.Frame(result, bg=PALETTE["card"])
        lower.grid(row=4, column=1, sticky="nsew", pady=(12, 0))
        lower.columnconfigure(0, weight=1)
        lower.columnconfigure(1, weight=1)

        feedback_col = tk.Frame(lower, bg=PALETTE["card"])
        feedback_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        tk.Label(feedback_col, text="Checklist", bg=PALETTE["card"], fg=PALETTE["text"], font=font(11, "bold")).pack(anchor="w", pady=(0, 4))
        self.feedback_frame = tk.Frame(feedback_col, bg=PALETTE["card"])
        self.feedback_frame.pack(fill="x")
        self.feedback_labels = []
        for _ in range(3):
            label = tk.Label(self.feedback_frame, text="", bg=PALETTE["card"], fg=PALETTE["muted"], font=font(9))
            label.pack(anchor="w", pady=3)
            self.feedback_labels.append(label)

        suggestions_col = tk.Frame(lower, bg=PALETTE["card"])
        suggestions_col.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        tk.Label(suggestions_col, text="Suggestions", bg=PALETTE["card"], fg=PALETTE["text"], font=font(11, "bold")).pack(anchor="w", pady=(0, 4))
        self.suggestion_labels = []
        for _ in range(3):
            label = tk.Label(suggestions_col, text="", bg=PALETTE["card"], fg=PALETTE["muted"], font=font(9))
            label.pack(anchor="w", pady=3)
            self.suggestion_labels.append(label)

        self._set_feedback(["Enter a password to begin."], ["Generate a strong password or type your own."])

    def _build_status_bar(self):
        bar = tk.Frame(self.content, bg=PALETTE["bg"], highlightbackground=PALETTE["stroke"], highlightthickness=1)
        bar.pack(fill="x", padx=32, pady=(0, 16))
        self.model_status = tk.Label(bar, text="◈ AI Model: Loading", bg=PALETTE["bg"], fg=PALETTE["muted"], font=font(9, "bold"))
        self.model_status.pack(side="left", padx=14, pady=9)
        tk.Label(bar, textvariable=self.status_var, bg=PALETTE["bg"], fg=PALETTE["green"], font=font(9, "bold")).pack(side="left", padx=14)
        tk.Label(bar, text="Version: v1.0", bg=PALETTE["bg"], fg=PALETTE["muted"], font=font(9, "bold")).pack(side="right", padx=14)
        tk.Label(bar, text="✦ Protected by AI", bg=PALETTE["bg"], fg=PALETTE["cyan"], font=font(9, "bold")).pack(side="right", padx=14)

    def _start_loading(self):
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            if MODEL_PATH.exists():
                artifact = joblib.load(MODEL_PATH)
                self.after(0, self._model_ready, artifact["model"], artifact["vectorizer"], "XGBoost")
                return
            model, vectorizer = train_model("logistic")
            self.after(0, self._model_ready, model, vectorizer, "Logistic Regression")
        except Exception as exc:
            self.after(0, self._model_failed, exc)

    def _model_ready(self, model, vectorizer, model_name):
        self.model = model
        self.vectorizer = vectorizer
        self.model_name = model_name
        self.model_status.configure(text=f"AI Model: {model_name}")
        self.status_var.set("Status: Ready")
        self._set_ready(True)
        self.result_label_var.set("Ready")
        self.badge_var.set("Local inference")
        self.password_entry.focus()

    def _model_failed(self, exc):
        self.status_var.set(f"Status: Model failed - {exc}")
        self.result_label_var.set("Model unavailable")
        self.badge_var.set("Error")
        self._set_ready(False)

    def _set_ready(self, ready):
        state = "normal" if ready else "disabled"
        self.password_entry.configure(state=state)
        self.eye_button.configure(state=state)
        self.generator_button.set_enabled(ready)
        self.check_button.set_enabled(ready)

    def _predict(self, password):
        if self.model_name == "XGBoost":
            tfidf = self.vectorizer.transform([password])
            engineered = build_feature_matrix([password], self.common_passwords)
            features = hstack([tfidf, engineered], format="csr")
            probabilities = self.model.predict_proba(features)[0]
            prediction = int(np.argmax(probabilities))
            confidence = float(probabilities[prediction])
            return prediction, STRENGTH[prediction]["label"], confidence

        prediction, label = predict_password(self.model, self.vectorizer, password)
        return int(prediction), label, 0.82

    def toggle_password(self):
        self.password_visible = not self.password_visible
        if not self.placeholder_active:
            self.password_entry.configure(show="" if self.password_visible else "*")
        self.eye_button.configure(text="◌" if self.password_visible else "◉")

    def generate_password(self):
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*()-_=+"
        generated = [
            random.choice(string.ascii_lowercase),
            random.choice(string.ascii_uppercase),
            random.choice(string.digits),
            random.choice("!@#$%^&*"),
        ]
        generated.extend(random.choice(alphabet) for _ in range(14))
        random.shuffle(generated)
        self.placeholder_active = False
        self.password_entry.configure(show="" if self.password_visible else "*", fg=PALETTE["text"])
        self.password_var.set("".join(generated))
        self.check_password()

    def check_password(self):
        password = self._actual_password()
        if not password:
            self.result_label_var.set("Empty state")
            self.badge_var.set("Waiting")
            self.confidence_var.set("Confidence: --")
            self.crack_var.set("Crack time: --")
            self.entropy_var.set("Entropy: --")
            self.length_var.set("Length: --")
            self.meter.animate_to(0, PALETTE["cyan"])
            self._draw_progress(0, PALETTE["cyan"])
            self._update_class_labels(password)
            self._set_feedback(["Enter a password to begin."], ["Generate a strong password or type your own."])
            return

        prediction, label, confidence = self._predict(password)
        info = STRENGTH[prediction]
        score = self._score_password(password, prediction, confidence)
        entropy = shannon_entropy(password)

        self.result_label_var.set(label)
        self.badge_var.set(f"{confidence * 100:.1f}% confidence")
        self.badge.configure(bg=info["badge"], fg=info["color"])
        self.confidence_var.set(f"Confidence: {confidence * 100:.1f}%")
        self.crack_var.set(f"Crack: {estimate_crack_time(password)}")
        self.entropy_var.set(f"Entropy: {entropy:.2f}")
        self.length_var.set(f"Length: {len(password)}")
        self.meter.animate_to(score, info["color"])
        self._draw_progress(score, info["color"])
        self._update_class_labels(password)
        self._set_feedback(self._feedback(password), self._suggestions(password))

    def _score_password(self, password, prediction, confidence):
        base = [28, 62, 88][prediction]
        features = password_features(password, self.common_passwords)
        length_bonus = min(8, max(0, len(password) - 8))
        diversity_bonus = features[10] * 2
        penalty = 0
        penalty += 12 if features[14] else 0
        penalty += 8 if features[15] else 0
        penalty += 6 if features[11] else 0
        penalty += 5 if features[12] else 0
        confidence_adjustment = int((confidence - 0.5) * 10)
        return max(0, min(100, base + length_bonus + diversity_bonus + confidence_adjustment - penalty))

    def _draw_progress(self, score, color):
        width = self.progress_canvas.winfo_width()
        if width <= 1:
            width = 420
        self.progress_canvas.delete("all")
        rounded_rect(self.progress_canvas, 1, 4, width - 1, 14, 8, fill="#26344E", outline="")
        fill_width = max(12, (width - 2) * score / 100)
        rounded_rect(self.progress_canvas, 1, 4, fill_width, 14, 8, fill=color, outline="")

    def _set_feedback(self, messages, suggestions=None):
        suggestions = suggestions or []
        for index, label in enumerate(self.feedback_labels):
            if index < len(messages):
                text, color = messages[index] if isinstance(messages[index], tuple) else (messages[index], PALETTE["muted"])
                label.configure(text=text, fg=color)
            else:
                label.configure(text="", fg=PALETTE["muted"])
        for index, label in enumerate(self.suggestion_labels):
            if index < len(suggestions):
                text, color = suggestions[index] if isinstance(suggestions[index], tuple) else (suggestions[index], PALETTE["muted"])
                label.configure(text=text, fg=color)
            else:
                label.configure(text="", fg=PALETTE["muted"])

    def _feedback(self, password):
        features = password_features(password, self.common_passwords)
        messages = []
        messages.append(("✓ Great length" if len(password) >= 12 else "⚠ Needs 12+ characters", PALETTE["green"] if len(password) >= 12 else PALETTE["yellow"]))
        messages.append(("✓ Contains special characters" if features[5] else "⚠ Add a symbol", PALETTE["green"] if features[5] else PALETTE["yellow"]))
        if features[14] or features[15]:
            messages.append(("⚠ Avoid common words", PALETTE["yellow"]))
        else:
            messages.append(("✓ No common password match", PALETTE["green"]))
        return messages

    def _suggestions(self, password):
        features = password_features(password, self.common_passwords)
        suggestions = []
        if features[11] or has_sequential_pattern(password):
            suggestions.append(("⚠ Remove keyboard sequences", PALETTE["yellow"]))
        elif features[12]:
            suggestions.append(("⚠ Reduce repeated characters", PALETTE["yellow"]))
        else:
            suggestions.append(("✓ Good randomness signals", PALETTE["green"]))
        if len(password) < 16:
            suggestions.append(("→ Add 4 more characters", PALETTE["muted"]))
        if features[10] < 4:
            suggestions.append(("→ Mix all character classes", PALETTE["muted"]))
        if len(suggestions) < 3:
            suggestions.append(("✓ Suitable for daily use", PALETTE["green"]))
        return suggestions[:3]

    def _update_class_labels(self, password):
        checks = {
            "Uppercase": any(char.isupper() for char in password),
            "Lowercase": any(char.islower() for char in password),
            "Numbers": any(char.isdigit() for char in password),
            "Symbols": any(not char.isalnum() for char in password),
        }
        for name, passed in checks.items():
            text = f"{'✓' if passed else '○'} {name}"
            color = PALETTE["green"] if passed else PALETTE["muted"]
            self.result_class_labels[name].configure(text=text, fg=color)

    def _update_live_state(self):
        password = self._actual_password()
        checks = {
            "Uppercase": any(char.isupper() for char in password),
            "Lowercase": any(char.islower() for char in password),
            "Numbers": any(char.isdigit() for char in password),
            "Symbols": any(not char.isalnum() for char in password),
        }
        for name, passed in checks.items():
            self.breakdown_labels[name].configure(
                text=f"{'✓' if passed else '○'} {name}",
                fg=PALETTE["green"] if passed else PALETTE["muted"],
            )

        if password and self.model is not None:
            self.after(180, self._live_check_if_current, password)

    def _live_check_if_current(self, password):
        if password == self.password_var.get():
            self.check_password()


if __name__ == "__main__":
    app = PremiumPasswordApp()
    app.mainloop()
