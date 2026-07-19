import math
import re
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
COMMON_PASSWORDS_PATH = BASE_DIR / "data" / "common_passwords.txt"
REPEATED_PATTERN = re.compile(r"(.)\1{2,}")

FEATURE_NAMES = [
    "length",
    "entropy",
    "has_upper",
    "has_lower",
    "has_digit",
    "has_special",
    "upper_count",
    "lower_count",
    "digit_count",
    "special_count",
    "class_diversity",
    "has_sequential_pattern",
    "has_repeated_chars",
    "max_repeated_run",
    "is_common_password",
    "contains_common_word",
]

KEYBOARD_ROWS = [
    "abcdefghijklmnopqrstuvwxyz",
    "0123456789",
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
]


def load_common_passwords(path=COMMON_PASSWORDS_PATH):
    if not Path(path).exists():
        return set()

    with open(path, "r", encoding="utf-8") as file:
        return {line.strip().lower() for line in file if line.strip()}


def shannon_entropy(password):
    if not password:
        return 0.0

    length = len(password)
    counts = {}
    for char in password:
        counts[char] = counts.get(char, 0) + 1

    entropy = 0.0
    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    return entropy


def has_sequential_pattern(password, min_length=3):
    lowered = password.lower()
    if len(lowered) < min_length:
        return False

    for row in KEYBOARD_ROWS:
        rows = (row, row[::-1])
        for candidate in rows:
            for index in range(len(candidate) - min_length + 1):
                if candidate[index:index + min_length] in lowered:
                    return True
    return False


def max_repeated_run(password):
    if not password:
        return 0

    longest = 1
    current = 1
    previous = password[0]
    for char in password[1:]:
        if char == previous:
            current += 1
            longest = max(longest, current)
        else:
            previous = char
            current = 1
    return longest


def password_features(password, common_passwords=None, common_words=None):
    common_passwords = common_passwords or set()
    common_words = common_words or tuple(word for word in common_passwords if len(word) >= 4)
    password = str(password)
    lowered = password.lower()

    upper_count = sum(char.isupper() for char in password)
    lower_count = sum(char.islower() for char in password)
    digit_count = sum(char.isdigit() for char in password)
    special_count = sum(not char.isalnum() for char in password)
    repeated_run = max_repeated_run(password)

    has_upper = int(upper_count > 0)
    has_lower = int(lower_count > 0)
    has_digit = int(digit_count > 0)
    has_special = int(special_count > 0)
    class_diversity = has_upper + has_lower + has_digit + has_special

    contains_common_word = any(word in lowered for word in common_words)

    return [
        len(password),
        shannon_entropy(password),
        has_upper,
        has_lower,
        has_digit,
        has_special,
        upper_count,
        lower_count,
        digit_count,
        special_count,
        class_diversity,
        int(has_sequential_pattern(password)),
        int(bool(REPEATED_PATTERN.search(password))),
        repeated_run,
        int(lowered in common_passwords),
        int(contains_common_word),
    ]


def build_feature_matrix(passwords, common_passwords=None):
    common_passwords = common_passwords or load_common_passwords()
    common_words = tuple(word for word in common_passwords if len(word) >= 4)
    rows = [password_features(password, common_passwords, common_words) for password in passwords]
    return np.asarray(rows, dtype=np.float32)
