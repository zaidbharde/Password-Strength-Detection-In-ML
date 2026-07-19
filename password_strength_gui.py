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
    "bg": "#070A12",
    "bg_2": "#0D1220",
    "card": "#111827",
    "card_2": "#151F32",
    "stroke": "#27344F",
    "stroke_focus": "#7C3AED",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "subtle": "#64748B",
    "blue": "#3B82F6",
    "purple": "#8B5CF6",
    "cyan": "#22D3EE",
    "green": "#34D399",
    "yellow": "#FBBF24",
    "red": "#FB7185",
    "shadow": "#050711",
}

STRENGTH = {
    0: {"label": "Weak", "color": PALETTE["red"], "badge": "#3A1420"},
    1: {"label": "Medium", "color": PALETTE["yellow"], "badge": "#33270F"},
    2: {"label": "Strong", "color": PALETTE["green"], "badge": "#0F3024"},
}

FONT = "Segoe UI"


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


class GradientButton(tk.Canvas):
    def __init__(self, parent, text, command, width=260, height=54):
        super().__init__(parent, width=width, height=height, bg=PALETTE["card"], highlightthickness=0, bd=0)
        self.text = text
        self.command = command
        self.width = width
        self.height = height
        self.enabled = True
        self.hovered = False
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<Button-1>", self._click)
        self.draw()

    def draw(self):
        self.delete("all")
        fill = "#1D4ED8" if self.hovered else "#2563EB"
        if not self.enabled:
            fill = "#263349"
        rounded_rect(self, 2, 2, self.width - 2, self.height - 2, 18, fill=fill, outline="#60A5FA", width=1)
        self.create_text(
            self.width / 2,
            self.height / 2,
            text=self.text,
            fill=PALETTE["text"],
            font=(FONT, 12, "bold"),
        )
        self.create_line(24, 3, self.width - 24, 3, fill="#93C5FD", width=1)

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.draw()

    def _enter(self, _event):
        self.hovered = True
        self.configure(cursor="hand2")
        self.draw()

    def _leave(self, _event):
        self.hovered = False
        self.configure(cursor="")
        self.draw()

    def _click(self, _event):
        if self.enabled:
            self.command()


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
            font=(FONT, 32, "bold"),
        )
        self.create_text(
            self.size / 2,
            self.size / 2 + 28,
            text="score",
            fill=PALETTE["muted"],
            font=(FONT, 10),
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
        self.geometry("1180x780")
        self.minsize(980, 680)
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
            color = "#070A12" if i % 2 else "#080D18"
            self.bg_canvas.create_rectangle(0, i * height / 36, width, (i + 1) * height / 36, fill=color, outline="", tags="bg")
        self.bg_canvas.create_oval(width - 420, -180, width + 120, 340, fill="#172554", outline="", tags="bg")
        self.bg_canvas.create_oval(-180, height - 360, 360, height + 180, fill="#2E1065", outline="", tags="bg")
        self.bg_canvas.create_oval(width * 0.45, height - 260, width * 0.45 + 360, height + 80, fill="#083344", outline="", tags="bg")
        self.bg_canvas.tag_lower("bg")

    def _build_header(self):
        header = tk.Frame(self.content, bg=PALETTE["bg"])
        header.pack(fill="x", padx=32, pady=(28, 18))

        title_block = tk.Frame(header, bg=PALETTE["bg"])
        title_block.pack(side="left", fill="x", expand=True)

        tk.Label(
            title_block,
            text="Password Strength AI",
            bg=PALETTE["bg"],
            fg=PALETTE["text"],
            font=(FONT, 30, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_block,
            text="A premium cybersecurity cockpit for ML-powered password analysis.",
            bg=PALETTE["bg"],
            fg=PALETTE["muted"],
            font=(FONT, 11),
        ).pack(anchor="w", pady=(6, 0))

        pill = tk.Canvas(header, width=188, height=42, bg=PALETTE["bg"], highlightthickness=0)
        pill.pack(side="right")
        rounded_rect(pill, 2, 2, 186, 40, 18, fill="#101827", outline="#2A3A59", width=1)
        pill.create_oval(18, 15, 28, 25, fill=PALETTE["green"], outline="")
        pill.create_text(98, 21, text="Protected by AI", fill=PALETTE["text"], font=(FONT, 10, "bold"))

    def _card(self, parent, title, subtitle=None):
        outer = tk.Frame(parent, bg=PALETTE["bg"])
        canvas = tk.Canvas(outer, bg=PALETTE["bg"], highlightthickness=0, bd=0)
        canvas.pack(fill="both", expand=True)
        body = tk.Frame(canvas, bg=PALETTE["card"])
        window = canvas.create_window(0, 0, anchor="nw", window=body)

        def redraw(event):
            canvas.delete("shape")
            width, height = event.width, event.height
            canvas.coords(window, 18, 18)
            canvas.itemconfigure(window, width=max(1, width - 36), height=max(1, height - 36))
            rounded_rect(canvas, 12, 12, width - 8, height - 8, 24, fill=PALETTE["shadow"], outline="", tags="shape")
            rounded_rect(canvas, 8, 8, width - 12, height - 12, 24, fill=PALETTE["card"], outline=PALETTE["stroke"], width=1, tags="shape")
            canvas.tag_lower("shape")

        canvas.bind("<Configure>", redraw)

        tk.Label(body, text=title, bg=PALETTE["card"], fg=PALETTE["text"], font=(FONT, 16, "bold")).pack(anchor="w", padx=24, pady=(22, 2))
        if subtitle:
            tk.Label(body, text=subtitle, bg=PALETTE["card"], fg=PALETTE["muted"], font=(FONT, 10), wraplength=420, justify="left").pack(anchor="w", padx=24, pady=(0, 18))
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
        input_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        self._build_input_card(input_body)

        guide_card, guide_body = self._card(
            dashboard,
            "Security Intelligence",
            "Guidance calibrated for stronger, less predictable passwords.",
        )
        guide_card.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(10, 0), pady=(0, 10))
        self._build_guide_card(guide_body)

        result_card, result_body = self._card(
            dashboard,
            "Result",
            "Score, crack-time estimate, entropy, and actionable feedback.",
        )
        result_card.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(10, 10))
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
            font=(FONT, 17),
        )
        self.entry_window = self.entry_canvas.create_window(22, 32, anchor="w", window=self.password_entry, height=38)
        self.eye_button = tk.Button(
            self.entry_canvas,
            text="View",
            command=self.toggle_password,
            bg="#111A2C",
            fg=PALETTE["muted"],
            activebackground="#17233A",
            activeforeground=PALETTE["text"],
            relief="flat",
            bd=0,
            font=(FONT, 9, "bold"),
            cursor="hand2",
        )
        self.eye_window = self.entry_canvas.create_window(0, 32, anchor="e", window=self.eye_button, width=58, height=34)
        self.password_entry.bind("<FocusIn>", lambda _event: self._draw_entry_shell())
        self.password_entry.bind("<FocusOut>", lambda _event: self._draw_entry_shell())
        self.password_entry.bind("<Return>", lambda _event: self.check_password())
        self.password_var.trace_add("write", lambda *_args: self._update_live_state())

        actions = tk.Frame(parent, bg=PALETTE["card"])
        actions.pack(fill="x", padx=24, pady=(0, 22))

        self.generator_button = tk.Button(
            actions,
            text="Generate Password",
            command=self.generate_password,
            bg="#151F32",
            fg=PALETTE["text"],
            activebackground="#1F2C46",
            activeforeground=PALETTE["text"],
            relief="flat",
            bd=0,
            font=(FONT, 10, "bold"),
            padx=18,
            pady=13,
            cursor="hand2",
        )
        self.generator_button.pack(side="left")

        self.check_button = GradientButton(actions, "Check Password Strength", self.check_password)
        self.check_button.pack(side="right")

        helper = tk.Label(
            parent,
            text="Live feedback updates as you type. Model inference runs locally on your machine.",
            bg=PALETTE["card"],
            fg=PALETTE["subtle"],
            font=(FONT, 9),
        )
        helper.pack(anchor="w", padx=24, pady=(0, 22))

        self.breakdown_frame = tk.Frame(parent, bg=PALETTE["card"])
        self.breakdown_frame.pack(fill="x", padx=24, pady=(0, 22))
        self.breakdown_labels = {}
        for index, item in enumerate(["Uppercase", "Lowercase", "Numbers", "Symbols"]):
            tile = tk.Frame(self.breakdown_frame, bg="#0D1424", padx=14, pady=12)
            tile.grid(row=index // 2, column=index % 2, sticky="ew", padx=6, pady=6)
            self.breakdown_frame.columnconfigure(index % 2, weight=1)
            label = tk.Label(tile, text=f"- {item}", bg="#0D1424", fg=PALETTE["muted"], font=(FONT, 10, "bold"))
            label.pack(anchor="w")
            self.breakdown_labels[item] = label

    def _draw_entry_shell(self, event=None):
        width = self.entry_canvas.winfo_width()
        if width <= 1:
            width = 520
        focused = self.focus_get() == self.password_entry
        self.entry_canvas.delete("shell")
        outline = PALETTE["stroke_focus"] if focused else PALETTE["stroke"]
        rounded_rect(self.entry_canvas, 2, 4, width - 2, 60, 20, fill="#0B1020", outline=outline, width=2, tags="shell")
        self.entry_canvas.coords(self.eye_window, width - 16, 32)
        self.entry_canvas.itemconfigure(self.entry_window, width=max(120, width - 106))
        self.entry_canvas.tag_lower("shell")

    def _build_guide_card(self, parent):
        guide = tk.Frame(parent, bg=PALETTE["card"])
        guide.pack(fill="both", expand=True, padx=24, pady=(0, 22))

        for label, body, color in [
            ("Weak", "Short, common, or predictable patterns.", PALETTE["red"]),
            ("Medium", "Acceptable, but missing entropy or variety.", PALETTE["yellow"]),
            ("Strong", "Long, diverse, and resistant to guessing.", PALETTE["green"]),
        ]:
            self._guide_row(guide, label, body, color)

        tk.Label(guide, text="Security Tips", bg=PALETTE["card"], fg=PALETTE["text"], font=(FONT, 14, "bold")).pack(anchor="w", pady=(24, 10))
        for tip in [
            "Use 12+ characters",
            "Avoid dictionary words",
            "Don't reuse passwords",
            "Use a password manager",
            "Prefer memorable passphrases",
        ]:
            tk.Label(guide, text=f"+ {tip}", bg=PALETTE["card"], fg=PALETTE["muted"], font=(FONT, 10)).pack(anchor="w", pady=5)

        tk.Label(
            guide,
            text="Premium mode",
            bg=PALETTE["card"],
            fg=PALETTE["cyan"],
            font=(FONT, 11, "bold"),
        ).pack(anchor="w", pady=(30, 4))
        tk.Label(
            guide,
            text="The app combines ML scoring with human-readable security heuristics for clear next steps.",
            bg=PALETTE["card"],
            fg=PALETTE["subtle"],
            font=(FONT, 9),
            wraplength=310,
            justify="left",
        ).pack(anchor="w")

    def _guide_row(self, parent, label, body, color):
        row = tk.Frame(parent, bg="#0D1424", padx=14, pady=12)
        row.pack(fill="x", pady=6)
        dot = tk.Canvas(row, width=18, height=18, bg="#0D1424", highlightthickness=0)
        dot.pack(side="left", padx=(0, 10))
        dot.create_oval(3, 3, 15, 15, fill=color, outline="")
        text = tk.Frame(row, bg="#0D1424")
        text.pack(side="left", fill="x", expand=True)
        tk.Label(text, text=label, bg="#0D1424", fg=PALETTE["text"], font=(FONT, 10, "bold")).pack(anchor="w")
        tk.Label(text, text=body, bg="#0D1424", fg=PALETTE["muted"], font=(FONT, 9)).pack(anchor="w")

    def _build_result_card(self, parent):
        result = tk.Frame(parent, bg=PALETTE["card"])
        result.pack(fill="both", expand=True, padx=24, pady=(0, 22))
        result.columnconfigure(1, weight=1)

        self.meter = CircularMeter(result, size=174)
        self.meter.grid(row=0, column=0, rowspan=4, padx=(0, 24), sticky="n")

        top = tk.Frame(result, bg=PALETTE["card"])
        top.grid(row=0, column=1, sticky="ew")
        tk.Label(top, textvariable=self.result_label_var, bg=PALETTE["card"], fg=PALETTE["text"], font=(FONT, 24, "bold")).pack(side="left")
        self.badge = tk.Label(top, textvariable=self.badge_var, bg="#17233A", fg=PALETTE["cyan"], font=(FONT, 10, "bold"), padx=12, pady=5)
        self.badge.pack(side="left", padx=(14, 0))

        self.progress_canvas = tk.Canvas(result, height=18, bg=PALETTE["card"], highlightthickness=0, bd=0)
        self.progress_canvas.grid(row=1, column=1, sticky="ew", pady=(16, 12))
        self.progress_canvas.bind("<Configure>", lambda _event: self._draw_progress(0, PALETTE["cyan"]))

        metrics = tk.Frame(result, bg=PALETTE["card"])
        metrics.grid(row=2, column=1, sticky="ew")
        metrics.columnconfigure(0, weight=1)
        metrics.columnconfigure(1, weight=1)
        metrics.columnconfigure(2, weight=1)
        for index, var in enumerate([self.crack_var, self.entropy_var, self.length_var]):
            tk.Label(metrics, textvariable=var, bg="#0D1424", fg=PALETTE["muted"], font=(FONT, 9, "bold"), padx=12, pady=10).grid(row=0, column=index, sticky="ew", padx=5)

        tk.Label(result, text="Feedback", bg=PALETTE["card"], fg=PALETTE["text"], font=(FONT, 13, "bold")).grid(row=3, column=1, sticky="w", pady=(20, 8))
        self.feedback_frame = tk.Frame(result, bg=PALETTE["card"])
        self.feedback_frame.grid(row=4, column=0, columnspan=2, sticky="ew")
        self.feedback_labels = []
        for _ in range(4):
            label = tk.Label(self.feedback_frame, text="", bg=PALETTE["card"], fg=PALETTE["muted"], font=(FONT, 10))
            label.pack(anchor="w", pady=3)
            self.feedback_labels.append(label)
        self._set_feedback(["Enter a password to unlock detailed feedback."])

    def _build_status_bar(self):
        bar = tk.Frame(self.content, bg="#090E19")
        bar.pack(fill="x", padx=32, pady=(0, 18))
        self.model_status = tk.Label(bar, text="AI Model: Loading", bg="#090E19", fg=PALETTE["muted"], font=(FONT, 9, "bold"))
        self.model_status.pack(side="left", padx=14, pady=10)
        tk.Label(bar, textvariable=self.status_var, bg="#090E19", fg=PALETTE["muted"], font=(FONT, 9, "bold")).pack(side="left", padx=14)
        tk.Label(bar, text="Version: v1.0", bg="#090E19", fg=PALETTE["muted"], font=(FONT, 9, "bold")).pack(side="right", padx=14)
        tk.Label(bar, text="Protected by AI", bg="#090E19", fg=PALETTE["cyan"], font=(FONT, 9, "bold")).pack(side="right", padx=14)

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
        self.generator_button.configure(state=state)
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
        self.password_entry.configure(show="" if self.password_visible else "*")
        self.eye_button.configure(text="Hide" if self.password_visible else "View")

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
        self.password_var.set("".join(generated))
        self.check_password()

    def check_password(self):
        password = self.password_var.get()
        if not password:
            self.result_label_var.set("Empty state")
            self.badge_var.set("Waiting")
            self.crack_var.set("Estimated crack time: --")
            self.entropy_var.set("Entropy: --")
            self.length_var.set("Length: --")
            self.meter.animate_to(0, PALETTE["cyan"])
            self._draw_progress(0, PALETTE["cyan"])
            self._set_feedback(["Enter a password to unlock detailed feedback."])
            return

        prediction, label, confidence = self._predict(password)
        info = STRENGTH[prediction]
        score = self._score_password(password, prediction, confidence)
        entropy = shannon_entropy(password)

        self.result_label_var.set(label)
        self.badge_var.set(f"{confidence * 100:.1f}% confidence")
        self.badge.configure(bg=info["badge"], fg=info["color"])
        self.crack_var.set(f"Crack time: {estimate_crack_time(password)}")
        self.entropy_var.set(f"Entropy: {entropy:.2f}")
        self.length_var.set(f"Length: {len(password)}")
        self.meter.animate_to(score, info["color"])
        self._draw_progress(score, info["color"])
        self._set_feedback(self._feedback(password))

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

    def _set_feedback(self, messages):
        for index, label in enumerate(self.feedback_labels):
            if index < len(messages):
                text, color = messages[index] if isinstance(messages[index], tuple) else (messages[index], PALETTE["muted"])
                label.configure(text=text, fg=color)
            else:
                label.configure(text="", fg=PALETTE["muted"])

    def _feedback(self, password):
        features = password_features(password, self.common_passwords)
        messages = []
        messages.append(("+ Great length" if len(password) >= 12 else "! Increase length to 12+ characters", PALETTE["green"] if len(password) >= 12 else PALETTE["yellow"]))
        messages.append(("+ Contains special characters" if features[5] else "! Add a symbol for better resistance", PALETTE["green"] if features[5] else PALETTE["yellow"]))
        if features[14] or features[15]:
            messages.append(("! Avoid common words or known passwords", PALETTE["yellow"]))
        else:
            messages.append(("+ No common password match detected", PALETTE["green"]))
        if features[11] or has_sequential_pattern(password):
            messages.append(("! Avoid keyboard or sequential patterns", PALETTE["yellow"]))
        elif features[12]:
            messages.append(("! Reduce repeated character runs", PALETTE["yellow"]))
        else:
            messages.append(("+ Good randomness signals", PALETTE["green"]))
        return messages

    def _update_live_state(self):
        password = self.password_var.get()
        checks = {
            "Uppercase": any(char.isupper() for char in password),
            "Lowercase": any(char.islower() for char in password),
            "Numbers": any(char.isdigit() for char in password),
            "Symbols": any(not char.isalnum() for char in password),
        }
        for name, passed in checks.items():
            self.breakdown_labels[name].configure(
                text=f"{'+' if passed else '-'} {name}",
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
