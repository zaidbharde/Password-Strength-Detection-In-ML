import threading
import tkinter as tk
from tkinter import ttk

from password_strength import predict_password, train_model


COLORS = {
    "bg": "#101820",
    "panel": "#17232f",
    "panel_light": "#203244",
    "text": "#f5f7fb",
    "muted": "#9fb0c3",
    "accent": "#39d98a",
    "weak": "#ff5c7a",
    "medium": "#f9c74f",
    "strong": "#39d98a",
    "button": "#2563eb",
    "button_hover": "#1d4ed8",
}

STRENGTH_COLORS = {
    0: COLORS["weak"],
    1: COLORS["medium"],
    2: COLORS["strong"],
}


class PasswordStrengthApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Password Strength Checker")
        self.geometry("820x560")
        self.minsize(720, 500)
        self.configure(bg=COLORS["bg"])

        self.model = None
        self.vectorizer = None
        self.password_visible = False

        self.password_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Training model...")
        self.result_var = tk.StringVar(value="Model is loading")
        self.score_var = tk.StringVar(value="Waiting")

        self._build_styles()
        self._build_layout()
        self._set_ready_state(False)
        self._start_training()

    def _build_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("TFrame", background=COLORS["bg"])
        self.style.configure("Panel.TFrame", background=COLORS["panel"])
        self.style.configure("Title.TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=("Segoe UI", 28, "bold"))
        self.style.configure("Subtitle.TLabel", background=COLORS["bg"], foreground=COLORS["muted"], font=("Segoe UI", 11))
        self.style.configure("PanelTitle.TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=("Segoe UI", 15, "bold"))
        self.style.configure("Text.TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=("Segoe UI", 11))
        self.style.configure("Muted.TLabel", background=COLORS["panel"], foreground=COLORS["muted"], font=("Segoe UI", 10))
        self.style.configure("Status.TLabel", background=COLORS["bg"], foreground=COLORS["muted"], font=("Segoe UI", 10))
        self.style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), foreground=COLORS["text"], background=COLORS["button"], padding=(16, 10), borderwidth=0)
        self.style.map("Accent.TButton", background=[("active", COLORS["button_hover"]), ("disabled", COLORS["panel_light"])])
        self.style.configure("Ghost.TButton", font=("Segoe UI", 10), foreground=COLORS["text"], background=COLORS["panel_light"], padding=(12, 8), borderwidth=0)
        self.style.map("Ghost.TButton", background=[("active", "#2b4158")])
        self.style.configure("Strength.Horizontal.TProgressbar", troughcolor=COLORS["panel_light"], background=COLORS["accent"], bordercolor=COLORS["panel_light"], lightcolor=COLORS["accent"], darkcolor=COLORS["accent"])

    def _build_layout(self):
        container = ttk.Frame(self, padding=28)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container)
        header.pack(fill="x")
        ttk.Label(header, text="Password Strength Checker", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Machine learning based password scoring with instant feedback.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(6, 0))

        main = ttk.Frame(container)
        main.pack(fill="both", expand=True, pady=(26, 16))
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        input_panel = ttk.Frame(main, style="Panel.TFrame", padding=24)
        input_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        input_panel.columnconfigure(0, weight=1)

        ttk.Label(input_panel, text="Check a password", style="PanelTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(input_panel, text="Enter a password below. The app trains once when it starts.", style="Muted.TLabel").grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 18))

        self.password_entry = tk.Entry(
            input_panel,
            textvariable=self.password_var,
            show="*",
            font=("Segoe UI", 16),
            bg=COLORS["panel_light"],
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
            relief="flat",
            highlightthickness=1,
            highlightbackground="#31465d",
            highlightcolor=COLORS["accent"],
        )
        self.password_entry.grid(row=2, column=0, sticky="ew", ipady=12, padx=(0, 10))
        self.password_entry.bind("<Return>", lambda _event: self.check_password())
        self.password_var.trace_add("write", lambda *_args: self._update_checks())

        self.toggle_button = ttk.Button(input_panel, text="Show", style="Ghost.TButton", command=self.toggle_password)
        self.toggle_button.grid(row=2, column=1, sticky="ew")

        self.check_button = ttk.Button(input_panel, text="Check Strength", style="Accent.TButton", command=self.check_password)
        self.check_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(18, 22))

        self.progress = ttk.Progressbar(input_panel, style="Strength.Horizontal.TProgressbar", maximum=100, value=0)
        self.progress.grid(row=4, column=0, columnspan=2, sticky="ew")

        self.result_label = tk.Label(
            input_panel,
            textvariable=self.result_var,
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=("Segoe UI", 24, "bold"),
            anchor="w",
        )
        self.result_label.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(18, 4))

        ttk.Label(input_panel, textvariable=self.score_var, style="Muted.TLabel").grid(row=6, column=0, columnspan=2, sticky="w")

        self.checks_frame = ttk.Frame(input_panel, style="Panel.TFrame")
        self.checks_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(24, 0))
        self.check_labels = {}
        for row, key in enumerate(["8+ characters", "Upper and lower case", "Number included", "Symbol included"]):
            label = ttk.Label(self.checks_frame, text=f"- {key}", style="Muted.TLabel")
            label.grid(row=row, column=0, sticky="w", pady=3)
            self.check_labels[key] = label

        result_panel = ttk.Frame(main, style="Panel.TFrame", padding=24)
        result_panel.grid(row=0, column=1, sticky="nsew")
        result_panel.columnconfigure(0, weight=1)

        ttk.Label(result_panel, text="Output Guide", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        self._guide_item(result_panel, 1, "Weak", "Easy to guess. Add length and variety.", COLORS["weak"])
        self._guide_item(result_panel, 2, "Medium", "Usable, but can still be improved.", COLORS["medium"])
        self._guide_item(result_panel, 3, "Strong", "Harder to predict based on the model.", COLORS["strong"])

        ttk.Label(result_panel, text="Tip", style="PanelTitle.TLabel").grid(row=4, column=0, sticky="w", pady=(28, 6))
        ttk.Label(
            result_panel,
            text="Use a longer passphrase with mixed character types. Avoid names, dates, and reused passwords.",
            style="Muted.TLabel",
            wraplength=250,
            justify="left",
        ).grid(row=5, column=0, sticky="ew")

        ttk.Label(container, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w")

    def _guide_item(self, parent, row, title, body, color):
        item = ttk.Frame(parent, style="Panel.TFrame")
        item.grid(row=row, column=0, sticky="ew", pady=(16, 0))
        marker = tk.Label(item, text="", bg=color, width=2, height=2)
        marker.pack(side="left", padx=(0, 10))
        text = ttk.Frame(item, style="Panel.TFrame")
        text.pack(side="left", fill="x", expand=True)
        ttk.Label(text, text=title, style="Text.TLabel").pack(anchor="w")
        ttk.Label(text, text=body, style="Muted.TLabel", wraplength=230).pack(anchor="w")

    def _start_training(self):
        thread = threading.Thread(target=self._train_model, daemon=True)
        thread.start()

    def _train_model(self):
        try:
            model, vectorizer = train_model("logistic")
            self.after(0, self._training_complete, model, vectorizer)
        except Exception as exc:
            self.after(0, self._training_failed, exc)

    def _training_complete(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        self.status_var.set("Ready. Model trained with Logistic Regression.")
        self.result_var.set("Ready to check")
        self.score_var.set("Enter a password and press Check Strength")
        self._set_ready_state(True)
        self.password_entry.focus()

    def _training_failed(self, exc):
        self.status_var.set(f"Training failed: {exc}")
        self.result_var.set("Could not train model")
        self.score_var.set("Check the CSV file and installed Python packages.")
        self._set_ready_state(False)

    def _set_ready_state(self, is_ready):
        state = "normal" if is_ready else "disabled"
        self.password_entry.configure(state=state)
        self.check_button.configure(state=state)
        self.toggle_button.configure(state=state)

    def toggle_password(self):
        self.password_visible = not self.password_visible
        self.password_entry.configure(show="" if self.password_visible else "*")
        self.toggle_button.configure(text="Hide" if self.password_visible else "Show")

    def check_password(self):
        password = self.password_var.get()
        if not password:
            self.result_var.set("Enter a password")
            self.score_var.set("The model needs text before it can predict.")
            self.progress.configure(value=0)
            self.result_label.configure(fg=COLORS["muted"])
            return

        prediction, label = predict_password(self.model, self.vectorizer, password)
        self.result_var.set(label)
        self.score_var.set(f"Model output: {prediction} | {label}")
        self.progress.configure(value=(prediction + 1) * 33.34)
        self.style.configure("Strength.Horizontal.TProgressbar", background=STRENGTH_COLORS[prediction], lightcolor=STRENGTH_COLORS[prediction], darkcolor=STRENGTH_COLORS[prediction])
        self.result_label.configure(fg=STRENGTH_COLORS[prediction])

    def _update_checks(self):
        password = self.password_var.get()
        checks = {
            "8+ characters": len(password) >= 8,
            "Upper and lower case": any(ch.islower() for ch in password) and any(ch.isupper() for ch in password),
            "Number included": any(ch.isdigit() for ch in password),
            "Symbol included": any(not ch.isalnum() for ch in password),
        }
        for key, passed in checks.items():
            self.check_labels[key].configure(text=f"{'+' if passed else '-'} {key}", foreground=COLORS["accent"] if passed else COLORS["muted"])


if __name__ == "__main__":
    app = PasswordStrengthApp()
    app.mainloop()
