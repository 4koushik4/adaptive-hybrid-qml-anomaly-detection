"""
Helper Utilities

Contains:
- Time utilities
- Formatting helpers
- Data conversion helpers
"""

import numpy as np
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# Timestamp Utility
# ─────────────────────────────────────────────────────────────

def current_timestamp():

    return datetime.utcnow().isoformat()


# ─────────────────────────────────────────────────────────────
# Percentage Formatter
# ─────────────────────────────────────────────────────────────

def format_percentage(value):

    return f"{value * 100:.2f}%"


# ─────────────────────────────────────────────────────────────
# Array Summary
# ─────────────────────────────────────────────────────────────

def summarize_array(arr):

    arr = np.array(arr)

    return {

        "min": float(np.min(arr)),

        "max": float(np.max(arr)),

        "mean": float(np.mean(arr)),

        "std": float(np.std(arr))
    }


# ─────────────────────────────────────────────────────────────
# Safe Division
# ─────────────────────────────────────────────────────────────

def safe_divide(a, b):

    if b == 0:
        return 0

    return a / b


# ─────────────────────────────────────────────────────────────
# Flatten Nested Dictionary
# ─────────────────────────────────────────────────────────────

def flatten_dict(
    dictionary,
    parent_key="",
    separator="_"
):

    items = []

    for key, value in dictionary.items():

        new_key = (
            f"{parent_key}{separator}{key}"
            if parent_key
            else key
        )

        if isinstance(value, dict):

            items.extend(

                flatten_dict(
                    value,
                    new_key,
                    separator
                ).items()
            )

        else:

            items.append(
                (new_key, value)
            )

    return dict(items)