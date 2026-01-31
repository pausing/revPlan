"""
Meeting notes storage: load/save notes by date from meeting_notes.json.
"""
import json
import os
from datetime import datetime

NOTES_FILE = os.path.join(os.path.dirname(__file__), "meeting_notes.json")


def load_notes():
    """Load all notes from the JSON file. Returns dict of date_str -> note text."""
    if not os.path.exists(NOTES_FILE):
        return {}
    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_notes(notes):
    """Write the full notes dict to the JSON file."""
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)


def get_note_for_date(dt):
    """Return the note text for the given date (datetime). Returns empty string if none."""
    notes = load_notes()
    key = dt.strftime("%Y-%m-%d")
    return notes.get(key, "")


def get_all_notes():
    """Return dict of date string (YYYY-MM-DD) -> note text."""
    return load_notes()


def save_note_for_date(dt, text):
    """Save note text for the given date (datetime)."""
    notes = load_notes()
    key = dt.strftime("%Y-%m-%d")
    notes[key] = text or ""
    _save_notes(notes)


def delete_note_for_date(dt):
    """Remove the note for the given date (datetime)."""
    notes = load_notes()
    key = dt.strftime("%Y-%m-%d")
    if key in notes:
        del notes[key]
        _save_notes(notes)
